"""Precompute frozen CR features for LR mini v2.

Run this script once before LR_train-mini-v2.py. It processes the original,
salient, and non-salient images with their corresponding trusted CR models,
then writes memory-mapped float16 feature arrays. TM features and normalized
10-bin labels are cached at the same time, so subsequent training no longer
loads images or executes three Swin-B encoders every epoch.

The generated cache still fully depends on all three images and CR models. If
an image, TM CSV, preprocessing transform, or CR checkpoint changes, regenerate
the corresponding cache.
"""

from __future__ import annotations

import json
import os
import time
from contextlib import nullcontext
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models.swin_transformer import swin_b


# =============================================================================
# Global configuration
# =============================================================================

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

IMAGE_DIR_ORIG = r"E:\2_NNU_experiment\2_aesthetic_dataset\AVA\images"
IMAGE_DIR_SAL = r"E:\2_NNU_experiment\2_aesthetic_dataset\AVA\images_addition_dense\images_salient"
IMAGE_DIR_NS = r"E:\2_NNU_experiment\2_aesthetic_dataset\AVA\images_addition_dense\images_non_salient"

CR_MODEL_ORIG = "./CR_model/My_CR-v3.pt"
CR_MODEL_SAL = "./CR_model/My_CR_S-v3.pt"
CR_MODEL_NS = "./CR_model/My_CR_NS-v3.pt"

SPLITS: Dict[str, Tuple[str, str, str]] = {
    "train": (
        "./TM_feature/train_TM_whole_rd.csv",
        "./TM_feature/train_TM_sal_rd.csv",
        "./TM_feature/train_TM_nonsal_rd.csv",
    ),
    "val": (
        "./TM_feature/val_TM_whole_rd.csv",
        "./TM_feature/val_TM_sal_rd.csv",
        "./TM_feature/val_TM_nonsal_rd.csv",
    ),
    "test": (
        "./TM_feature/test_TM_whole_rd.csv",
        "./TM_feature/test_TM_sal_rd.csv",
        "./TM_feature/test_TM_nonsal_rd.csv",
    ),
}
SPLITS_TO_PROCESS = ("train", "val", "test")

CACHE_ROOT = "./LR_feature"
OVERWRITE_EXISTING_CACHE = True

FEATURE_DIM = 1024
NUM_SCORE_BINS = 10
CR_BATCH_SIZE = 32
NUM_WORKERS = 4
USE_AMP = True
AMP_DTYPE = "float16"  # float16 is efficient and stable for frozen inference.
PRINT_EVERY_BATCHES = 50


class Rank_Swin(nn.Module):
    """Compatibility definition required by trusted full-object CR checkpoints."""

    def __init__(self):
        super().__init__()
        self.encoder_image = swin_b()
        self.encoder_image.head = nn.Sequential()
        self.projector_image = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        image_feature = self.encoder_image(image)
        image_feature_contrast = self.projector_image(image_feature)
        return nn.functional.normalize(image_feature_contrast, dim=1)


def read_branch_csv(
    csv_path: str,
    branch_name: str,
    include_labels: bool,
) -> Tuple[pd.DataFrame, str, List[str]]:
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    header = pd.read_csv(csv_path, nrows=0)
    if header.shape[1] < 12:
        raise ValueError(f"{branch_name} CSV has fewer than 12 columns: {header.shape[1]}")

    id_col = header.columns[0]
    tm_candidates = [
        column for column in header.columns if column.strip().lower() == "tm_features"
    ]
    tm_col = tm_candidates[0] if tm_candidates else header.columns[11]
    label_columns = list(header.columns[1:11]) if include_labels else []
    use_columns = [id_col, *label_columns, tm_col]

    print(f"Reading {branch_name} CSV: {csv_path}")
    df = pd.read_csv(csv_path, usecols=use_columns, dtype=str)
    embedded_header = df[id_col].astype(str).str.strip().str.lower().isin(
        {"image_id", "imageid"}
    )
    if embedded_header.any():
        print(
            f"Warning: removed {int(embedded_header.sum())} embedded header rows "
            f"from {branch_name}"
        )
        df = df.loc[~embedded_header].copy()

    df["_image_id_key"] = (
        df[id_col].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    )
    if df["_image_id_key"].eq("").any():
        raise ValueError(f"{branch_name} contains empty image IDs")
    duplicate_mask = df["_image_id_key"].duplicated(keep=False)
    if duplicate_mask.any():
        examples = df.loc[duplicate_mask, "_image_id_key"].head(5).tolist()
        raise ValueError(
            f"{branch_name} contains duplicate image IDs; examples: {examples}"
        )
    return df.reset_index(drop=True), tm_col, label_columns


class ThreeBranchPrecomputeDataset(Dataset):
    def __init__(
        self,
        csv_orig: str,
        csv_sal: str,
        csv_ns: str,
        transform,
    ):
        self.transform = transform
        df_orig, tm_orig_col, label_columns = read_branch_csv(
            csv_orig, "original", include_labels=True
        )
        df_sal, tm_sal_col, _ = read_branch_csv(
            csv_sal, "salient", include_labels=False
        )
        df_ns, tm_ns_col, _ = read_branch_csv(
            csv_ns, "non-salient", include_labels=False
        )
        if len(label_columns) != NUM_SCORE_BINS:
            raise ValueError(
                f"Expected {NUM_SCORE_BINS} label columns, found {len(label_columns)}"
            )

        orig_ids = set(df_orig["_image_id_key"])
        sal_ids = set(df_sal["_image_id_key"])
        ns_ids = set(df_ns["_image_id_key"])
        if not (orig_ids == sal_ids == ns_ids):
            raise ValueError(
                "Three branch ID sets differ: "
                f"orig={len(orig_ids)}, sal={len(sal_ids)}, ns={len(ns_ids)}, "
                f"orig_minus_sal={len(orig_ids - sal_ids)}, "
                f"orig_minus_ns={len(orig_ids - ns_ids)}"
            )

        orig_keep = df_orig[
            ["_image_id_key", *label_columns, tm_orig_col]
        ].rename(columns={tm_orig_col: "tm_orig"})
        sal_keep = df_sal[["_image_id_key", tm_sal_col]].rename(
            columns={tm_sal_col: "tm_sal"}
        )
        ns_keep = df_ns[["_image_id_key", tm_ns_col]].rename(
            columns={tm_ns_col: "tm_ns"}
        )

        self.df = (
            orig_keep
            .merge(
                sal_keep,
                on="_image_id_key",
                how="inner",
                validate="one_to_one",
                sort=False,
            )
            .merge(
                ns_keep,
                on="_image_id_key",
                how="inner",
                validate="one_to_one",
                sort=False,
            )
            .rename(columns={"_image_id_key": "image_id"})
            .reset_index(drop=True)
        )
        if len(self.df) != len(df_orig):
            raise ValueError(
                f"ID merge changed sample count: before={len(df_orig)}, after={len(self.df)}"
            )
        self.label_columns = label_columns
        self.image_ids = self.df["image_id"].astype(str).tolist()
        print(f"Aligned three-branch dataset size: {len(self.df)}")

    @staticmethod
    def parse_tm_feature(value: object, image_id: str, branch_name: str) -> np.ndarray:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Empty TM feature: image={image_id}, branch={branch_name}")
        feature = np.fromstring(value.strip().strip("[]"), sep=",", dtype=np.float32)
        if feature.size != FEATURE_DIM:
            raise ValueError(
                f"TM dimension error: image={image_id}, branch={branch_name}, "
                f"expected={FEATURE_DIM}, actual={feature.size}"
            )
        if not np.all(np.isfinite(feature)):
            raise ValueError(f"Non-finite TM feature: image={image_id}, branch={branch_name}")
        return feature

    def load_image(self, directory: str, image_id: str) -> torch.Tensor:
        image_path = os.path.join(directory, f"{image_id}.jpg")
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            return self.transform(image)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> Dict[str, object]:
        row = self.df.iloc[index]
        image_id = str(row["image_id"])

        labels = row[self.label_columns].to_numpy(dtype=np.float32)
        label_sum = float(labels.sum())
        if label_sum <= 0.0 or not np.all(np.isfinite(labels)):
            raise ValueError(f"Invalid rating distribution for image {image_id}")
        labels = labels / label_sum

        return {
            "image_id": image_id,
            "image_orig": self.load_image(IMAGE_DIR_ORIG, image_id),
            "image_sal": self.load_image(IMAGE_DIR_SAL, image_id),
            "image_ns": self.load_image(IMAGE_DIR_NS, image_id),
            "tm_orig": torch.from_numpy(
                self.parse_tm_feature(row["tm_orig"], image_id, "original")
            ),
            "tm_sal": torch.from_numpy(
                self.parse_tm_feature(row["tm_sal"], image_id, "salient")
            ),
            "tm_ns": torch.from_numpy(
                self.parse_tm_feature(row["tm_ns"], image_id, "non-salient")
            ),
            "labels": torch.from_numpy(labels),
        }


def load_cr_encoder(checkpoint_path: str, device: torch.device) -> nn.Module:
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"CR checkpoint not found: {checkpoint_path}")
    print(f"Loading trusted CR checkpoint: {checkpoint_path}")
    # These three user-generated files are legacy full-model checkpoints.
    cr_model = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    if not hasattr(cr_model, "encoder_image"):
        raise TypeError(f"CR checkpoint has no encoder_image: {checkpoint_path}")
    encoder = cr_model.encoder_image.to(device)
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    return encoder


def autocast_context(device: torch.device):
    if not USE_AMP or device.type != "cuda":
        return nullcontext()
    dtype = torch.float16 if AMP_DTYPE.lower() == "float16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype, enabled=True)


def create_cache_arrays(output_dir: str, num_samples: int):
    os.makedirs(output_dir, exist_ok=True)
    feature_shape = (num_samples, FEATURE_DIM)
    arrays = {
        "cr_orig": np.lib.format.open_memmap(
            os.path.join(output_dir, "cr_orig.npy"),
            mode="w+",
            dtype=np.float16,
            shape=feature_shape,
        ),
        "cr_sal": np.lib.format.open_memmap(
            os.path.join(output_dir, "cr_sal.npy"),
            mode="w+",
            dtype=np.float16,
            shape=feature_shape,
        ),
        "cr_ns": np.lib.format.open_memmap(
            os.path.join(output_dir, "cr_ns.npy"),
            mode="w+",
            dtype=np.float16,
            shape=feature_shape,
        ),
        "tm_orig": np.lib.format.open_memmap(
            os.path.join(output_dir, "tm_orig.npy"),
            mode="w+",
            dtype=np.float16,
            shape=feature_shape,
        ),
        "tm_sal": np.lib.format.open_memmap(
            os.path.join(output_dir, "tm_sal.npy"),
            mode="w+",
            dtype=np.float16,
            shape=feature_shape,
        ),
        "tm_ns": np.lib.format.open_memmap(
            os.path.join(output_dir, "tm_ns.npy"),
            mode="w+",
            dtype=np.float16,
            shape=feature_shape,
        ),
        "labels": np.lib.format.open_memmap(
            os.path.join(output_dir, "labels.npy"),
            mode="w+",
            dtype=np.float32,
            shape=(num_samples, NUM_SCORE_BINS),
        ),
    }
    return arrays


@torch.inference_mode()
def precompute_split(
    split_name: str,
    csv_paths: Tuple[str, str, str],
    encoders: Tuple[nn.Module, nn.Module, nn.Module],
    device: torch.device,
) -> None:
    output_dir = os.path.join(CACHE_ROOT, split_name)
    metadata_path = os.path.join(output_dir, "metadata.json")
    if os.path.exists(metadata_path) and not OVERWRITE_EXISTING_CACHE:
        raise FileExistsError(
            f"Cache already exists: {output_dir}. Set OVERWRITE_EXISTING_CACHE=True to refresh it."
        )
    if os.path.exists(metadata_path) and OVERWRITE_EXISTING_CACHE:
        # Remove the completion marker before rewriting. If extraction fails,
        # training will reject the incomplete cache instead of reading it.
        os.remove(metadata_path)

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    dataset = ThreeBranchPrecomputeDataset(
        csv_orig=csv_paths[0],
        csv_sal=csv_paths[1],
        csv_ns=csv_paths[2],
        transform=transform,
    )
    loader_kwargs = {
        "batch_size": CR_BATCH_SIZE,
        "shuffle": False,
        "num_workers": NUM_WORKERS,
        "pin_memory": device.type == "cuda",
        "persistent_workers": NUM_WORKERS > 0,
        "drop_last": False,
    }
    if NUM_WORKERS > 0:
        loader_kwargs["prefetch_factor"] = 2
    loader = DataLoader(dataset, **loader_kwargs)

    arrays = create_cache_arrays(output_dir, len(dataset))
    max_id_length = max(1, max(len(image_id) for image_id in dataset.image_ids))
    np.save(
        os.path.join(output_dir, "image_ids.npy"),
        np.asarray(dataset.image_ids, dtype=f"<U{max_id_length}"),
        allow_pickle=False,
    )

    encoder_orig, encoder_sal, encoder_ns = encoders
    cursor = 0
    start_time = time.time()
    for batch_index, batch in enumerate(loader):
        images_orig = batch["image_orig"].to(device, non_blocking=True)
        images_sal = batch["image_sal"].to(device, non_blocking=True)
        images_ns = batch["image_ns"].to(device, non_blocking=True)

        with autocast_context(device):
            cr_orig = encoder_orig(images_orig)
            cr_sal = encoder_sal(images_sal)
            cr_ns = encoder_ns(images_ns)

        for branch_name, feature in (
            ("original", cr_orig),
            ("salient", cr_sal),
            ("non-salient", cr_ns),
        ):
            if feature.ndim != 2 or feature.size(1) != FEATURE_DIM:
                raise ValueError(
                    f"CR output shape error for {branch_name}: {tuple(feature.shape)}, "
                    f"expected [batch, {FEATURE_DIM}]"
                )
            if not torch.isfinite(feature).all():
                raise ValueError(f"CR output contains non-finite values: {branch_name}")

        batch_size = cr_orig.size(0)
        end = cursor + batch_size
        arrays["cr_orig"][cursor:end] = cr_orig.float().cpu().numpy().astype(np.float16)
        arrays["cr_sal"][cursor:end] = cr_sal.float().cpu().numpy().astype(np.float16)
        arrays["cr_ns"][cursor:end] = cr_ns.float().cpu().numpy().astype(np.float16)
        arrays["tm_orig"][cursor:end] = batch["tm_orig"].numpy().astype(np.float16)
        arrays["tm_sal"][cursor:end] = batch["tm_sal"].numpy().astype(np.float16)
        arrays["tm_ns"][cursor:end] = batch["tm_ns"].numpy().astype(np.float16)
        arrays["labels"][cursor:end] = batch["labels"].numpy().astype(np.float32)
        cursor = end

        if (batch_index + 1) % PRINT_EVERY_BATCHES == 0 or cursor == len(dataset):
            elapsed = time.time() - start_time
            rate = cursor / max(elapsed, 1e-6)
            print(
                f"[{split_name}] {cursor}/{len(dataset)} samples, "
                f"{rate:.1f} samples/s"
            )

    if cursor != len(dataset):
        raise RuntimeError(
            f"Cache write incomplete for {split_name}: wrote={cursor}, expected={len(dataset)}"
        )

    for array in arrays.values():
        array.flush()
    del arrays

    metadata = {
        "version": 2,
        "split": split_name,
        "num_samples": len(dataset),
        "feature_dim": FEATURE_DIM,
        "num_bins": NUM_SCORE_BINS,
        "storage_dtype": "float16",
        "image_size": [224, 224],
        "csv_orig": os.path.abspath(csv_paths[0]),
        "csv_sal": os.path.abspath(csv_paths[1]),
        "csv_ns": os.path.abspath(csv_paths[2]),
        "cr_model_orig": os.path.abspath(CR_MODEL_ORIG),
        "cr_model_sal": os.path.abspath(CR_MODEL_SAL),
        "cr_model_ns": os.path.abspath(CR_MODEL_NS),
    }
    metadata_temp_path = metadata_path + ".tmp"
    with open(metadata_temp_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    os.replace(metadata_temp_path, metadata_path)
    print(f"Completed {split_name} cache: {output_dir}")


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for practical Swin-B feature extraction. "
            "Run this script on the RTX 4090 server."
        )

    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    print(f"Using device: {torch.cuda.get_device_name(device)}")

    encoders = (
        load_cr_encoder(CR_MODEL_ORIG, device),
        load_cr_encoder(CR_MODEL_SAL, device),
        load_cr_encoder(CR_MODEL_NS, device),
    )
    for split_name in SPLITS_TO_PROCESS:
        if split_name not in SPLITS:
            raise KeyError(f"Unknown split: {split_name}")
        print("=" * 80)
        print(f"Precomputing split: {split_name}")
        precompute_split(split_name, SPLITS[split_name], encoders, device)

    print("All requested caches were generated successfully.")


if __name__ == "__main__":
    main()
