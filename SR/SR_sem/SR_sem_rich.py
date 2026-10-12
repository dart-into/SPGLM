"""
@DESCRIPTION: AVA generate Semantic Similarity Features using CLIP and save them.
@AUTHOR: lty
"""
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = 'max_split_size_mb:2048'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import clip
import torch
import numpy as np
import pandas as pd
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
from torch.utils.data.dataloader import default_collate
from typing import Optional, Dict, List, Any


# ==============================================================================
# 数据加载部分 (Data Loading Part)
# ==============================================================================

class ImageAVADataset(Dataset):
    """
    AVA 图像数据集加载器
    """

    def __init__(self, csv_file: str, root_dir: str, transform: Optional[transforms.Compose] = None):
        self.image_information = pd.read_csv(csv_file, sep=',')
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_information)

    def __getitem__(self, idx: int) -> Optional[Dict[str, torch.Tensor]]:
        try:
            img_name = str(self.image_information.iloc[idx, 0])
            img_path = os.path.join(self.root_dir, f"{img_name}.jpg")

            image = Image.open(img_path).convert('RGB')

            if self.transform:
                image = self.transform(image)

            sample = {'image': image}
            return sample

        except Exception as e:
            print(f"Warning: Could not read image {self.image_information.iloc[idx, 0]}.jpg. Error: {e}. Skipping.")
            return None


def my_collate(batch: List[Optional[Dict[str, torch.Tensor]]]) -> Optional[Dict[str, torch.Tensor]]:
    """
    自定义 collate_fn，用于过滤掉数据加载过程中返回 None 的坏样本
    """
    # 过滤掉 None 样本
    batch = list(filter(lambda x: x is not None, batch))

    if not batch:
        return None

    # 使用默认的 collate_fn 拼接剩余的有效样本
    return default_collate(batch)


def load_data(batch_size: int = 64) -> DataLoader:
    """
    加载 AVA 训练数据集
    """
    data_train_label_dir = r"../../AVA/Label/Addition_Label/Rich_Label/rich_train.csv"
    data_image_dir = r"/home/user/Data/AVA/images"

    # 定义图像预处理
    transform_pipeline = transforms.Compose([
        transforms.Resize((336, 336)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    transformed_dataset_train = ImageAVADataset(
        csv_file=data_train_label_dir,
        root_dir=data_image_dir,
        transform=transform_pipeline
    )

    data_train = DataLoader(
        transformed_dataset_train,
        batch_size=batch_size,
        shuffle=False,  # 确保顺序一致
        num_workers=4,
        collate_fn=my_collate,
        drop_last=False,
        pin_memory=True
    )

    return data_train


# ==============================================================================
# 特征生成部分 (Feature Generation Part)
# ==============================================================================

def generate_sem_tags() -> np.ndarray:
    """
    加载语义标签
    """
    semantic_csv = pd.read_csv(r'../../TM/All_Tags/Sem_Tags.csv', header=None)
    semantic_tags = np.array(semantic_csv)
    semantic_tags = np.squeeze(semantic_tags)  # (N,)
    print(f"semantic_tags.shape is :{semantic_tags.shape}")

    return semantic_tags


def generate_sem_similarity() -> pd.DataFrame:
    """
    使用 CLIP 计算图像与所有语义标签的相似度特征
    """
    print("=========== generate_sem_similarity Start ===========")

    # 1. 加载 CLIP 模型
    # ViT-L/14@336px 需要匹配 336x336 的图像输入
    Text_Feature_extractor, _ = clip.load('ViT-L/14@336px')
    Text_Feature_extractor.float().cuda()
    model = nn.DataParallel(Text_Feature_extractor).eval()

    # 2. 准备文本标签
    tags = generate_sem_tags()
    tokenized_tags = clip.tokenize(tags).cuda()

    # 3. 准备图像数据
    data_train = load_data(batch_size=64)  # GPU 显存允许的话，可以适当调大 batch_size

    similarity_feature = []

    # 4. 遍历图像
    for data in tqdm(data_train, desc="Generating Semantic Features"):
        if data is None:  # 跳过由 my_collate 产生的空 batch
            continue

        inputs = data['image'].cuda()
        batch_features = []

        # 5. 标签分批计算 (防止 OOM)
        # 每个 batch 的图片与所有标签算相似度，但标签也分批
        tag_batch_size = 1268
        for i in range(0, len(tokenized_tags), tag_batch_size):
            tag_batch = tokenized_tags[i: i + tag_batch_size]

            with torch.no_grad():
                # model 返回 (logits_per_image, logits_per_text)
                logits_per_image, _ = model(inputs, tag_batch)
                batch_features.append(logits_per_image.cpu())

        # 合并当前图片 batch 的所有标签相似度 (dim=1)
        # [ (B, 128), (B, 128), ... ] -> (B, Total_Tags)
        batch_features_tensor = torch.cat(batch_features, dim=1)
        batch_numpy_array = batch_features_tensor.numpy()
        batch_numpy_array_rounded = np.round(batch_numpy_array, decimals=6)
        similarity_feature.append(batch_numpy_array_rounded)

    # 6. 合并所有 batch 的特征
    all_sem_features = np.vstack(similarity_feature)

    print("=========== generate_sem_similarity End ===========")
    return pd.DataFrame(all_sem_features)


# ==============================================================================
# 主函数 (Main Execution)
# ==============================================================================

if __name__ == '__main__':
    torch.cuda.empty_cache()

    # 定义输出文件名
    output_feature_csv = './CR_Sem_Feature_R.csv'

    # 1. 生成语义相似度特征
    Sem_features_df = generate_sem_similarity()

    # 2. 将特征保存到CSV文件
    print(f"Features generated with shape: {Sem_features_df.shape}")
    print(f"Saving semantic features to {output_feature_csv}...")

    Sem_features_df.to_csv(output_feature_csv, index=False, float_format='%.6f')

    print(f"特征已成功保存至{output_feature_csv}。现在可以运行CR_cluster.py")