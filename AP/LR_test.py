"""
@DESCRIPTION: TMCR testing on AVA (Transformer Version)
@AUTHOR: lty
@MODIFIED: 适配 Transformer 版本的训练代码，输出改为CSV格式
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import json
from contextlib import nullcontext
import torch
import warnings
import numpy as np
import csv  # [新增] 引入csv模块
from scipy.stats import spearmanr, pearsonr
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error

use_gpu = True
warnings.filterwarnings("ignore")

# --- LR_precompute-mini-v2.py 生成的测试缓存目录 ---
TEST_CACHE_DIR = './LR_feature/test'

# --- 模型文件读取模式 ---
MODEL_DIR = './LR_model'
# 对应训练代码保存的文件名前缀
MODEL_BASENAME = 'LR_Train'
TOTAL_ROUNDS = 10
BINARY_ACC_THRESHOLD = 5.0
TEST_BATCH_SIZE = 1024
USE_AMP = True
AMP_DTYPE = 'bfloat16'
# --- 输出日志路径 ---
# [修改] 将后缀改为 .csv
SAVE_FILENAME = './Log/LR_Test.csv'
PREDICTION_SAVE_FILENAME = './Log/LR_Test_Predictions.csv'

# ===============================================================================
# 1. 模型定义 (必须与训练代码完全一致)
# ===============================================================================

class GIAA_model(nn.Module):
    def __init__(self):
        super(GIAA_model, self).__init__()

        # 构建 TM 特征降维网络 (1024 -> 128)
        def make_tm_projector():
            return nn.Sequential(
                nn.Linear(1024, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.35),

                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.35)
            )
        self.fc_tm_orig = make_tm_projector()
        self.fc_tm_sal = make_tm_projector()
        self.fc_tm_ns = make_tm_projector()

        # 构建 CR 特征降维网络 (1024 -> 128)
        def make_cr_projector():
            return nn.Sequential(
                nn.Linear(1024, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.15),

                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.15)
            )
        self.fc_cr_orig = make_cr_projector()
        self.fc_cr_sal = make_cr_projector()
        self.fc_cr_ns = make_cr_projector()

        # 定义3个分支独立的 CLS Token (1, 1, 256)
        self.cls_token_orig = nn.Parameter(torch.zeros(1, 1, 256))
        nn.init.trunc_normal_(self.cls_token_orig, std=0.02)
        self.cls_token_sal = nn.Parameter(torch.zeros(1, 1, 256))
        nn.init.trunc_normal_(self.cls_token_sal, std=0.02)
        self.cls_token_ns = nn.Parameter(torch.zeros(1, 1, 256))
        nn.init.trunc_normal_(self.cls_token_ns, std=0.02)
        # 分支位置编码：序列长度为 3 ([CLS_Branch, Image, Text])
        self.pos_embed_branch = nn.Parameter(torch.zeros(1, 3, 256))
        nn.init.trunc_normal_(self.pos_embed_branch, std=0.02)
        # 三路共享 Transformer，并使用分支类型编码保留原图/显著图/非显著图身份。
        self.branch_type_embed = nn.Parameter(torch.zeros(1, 3, 1, 256))
        nn.init.trunc_normal_(self.branch_type_embed, std=0.02)
        branch_encoder_layer = nn.TransformerEncoderLayer(
            d_model=256,
            nhead=8,
            batch_first=True,
            dim_feedforward=256 * 4,
            dropout=0.15
        )
        self.branch_transformer = nn.TransformerEncoder(branch_encoder_layer, num_layers=2)

        # 定义最终的全局 CLS Token
        self.cls_token_final = nn.Parameter(torch.zeros(1, 1, 256))
        nn.init.trunc_normal_(self.cls_token_final, std=0.02)
        # 全局位置编码：序列长度为 4 ([CLS_Final, CLS_Orig_Out, CLS_Sal_Out, CLS_NS_Out])
        self.pos_embed_global = nn.Parameter(torch.zeros(1, 4, 256))
        nn.init.trunc_normal_(self.pos_embed_global, std=0.02)
        # 全局 Transformer Encoder
        global_encoder_layer = nn.TransformerEncoderLayer(
            d_model=256,
            nhead=8,
            batch_first=True,
            dim_feedforward=256 * 4,
            dropout=0.15
        )
        self.global_transformer = nn.TransformerEncoder(global_encoder_layer, num_layers=2)

        # 预测头
        self.regression = nn.Sequential(
            nn.Linear(256, 10),
            nn.Softmax(dim=1)
        )

    def forward(self, cr_orig, cr_sal, cr_non_sal, tm_orig, tm_sal, tm_non_sal):
        # CR/TM 特征均由 LR_precompute-mini-v2.py 预先生成。
        emb_tm_orig = self.fc_tm_orig(tm_orig)
        emb_tm_sal = self.fc_tm_sal(tm_sal)
        emb_tm_ns = self.fc_tm_ns(tm_non_sal)

        emb_cr_orig = self.fc_cr_orig(cr_orig)
        emb_cr_sal = self.fc_cr_sal(cr_sal)
        emb_cr_ns = self.fc_cr_ns(cr_non_sal)

        # ======================================================================
        # Stage 1: Branch Fusion (分别融合三个分支)
        # ======================================================================

        # 扩展 CLS tokens
        B = cr_orig.shape[0]
        cls_o = self.cls_token_orig.expand(B, -1, -1)   # (B, 1, 256)
        cls_s = self.cls_token_sal.expand(B, -1, -1)    # (B, 1, 256)
        cls_ns = self.cls_token_ns.expand(B, -1, -1)    # (B, 1, 256)

        # 三路序列堆叠为 (B, 3 branches, 3 tokens, 256)。
        seq_orig = torch.cat([cls_o, emb_cr_orig.unsqueeze(1), emb_tm_orig.unsqueeze(1)], dim=1)
        seq_sal = torch.cat([cls_s, emb_cr_sal.unsqueeze(1), emb_tm_sal.unsqueeze(1)], dim=1)
        seq_ns = torch.cat([cls_ns, emb_cr_ns.unsqueeze(1), emb_tm_ns.unsqueeze(1)], dim=1)
        branch_sequences = torch.stack([seq_orig, seq_sal, seq_ns], dim=1)
        branch_sequences = (
            branch_sequences
            + self.pos_embed_branch.unsqueeze(1)
            + self.branch_type_embed
        )

        # 合并三路到 batch 维度，只调用一次共享 Transformer。
        shared_input = branch_sequences.reshape(B * 3, 3, 256)
        shared_output = self.branch_transformer(shared_input)
        shared_output = shared_output.reshape(B, 3, 3, 256)
        branch_cls = shared_output[:, :, 0, :]  # (B, 3, 256)

        # ======================================================================
        # Stage 2: Global Fusion (融合三个分支的 CLS)
        # ======================================================================

        # 扩展最终 CLS
        cls_f = self.cls_token_final.expand(B, -1, -1) # (B, 1, 256)
        # 构建全局序列: [Final_CLS, Branch_Orig_CLS, Branch_Sal_CLS, Branch_NS_CLS] -> (B, 4, 256)
        seq_final = torch.cat([cls_f, branch_cls], dim=1)
        # 加入全局位置编码
        seq_final = seq_final + self.pos_embed_global
        # 进入全局 Transformer
        out_seq_final = self.global_transformer(seq_final)
        # 取出最终用于预测的 CLS Token
        final_feature = out_seq_final[:, 0, :] # (B, 256)

        # ======================================================================
        # Stage 3: Prediction
        # ======================================================================
        score = self.regression(final_feature)

        return score

class EMD2Loss(nn.Module):
    def __init__(self):
        super(EMD2Loss, self).__init__()
        self.loss_fn = nn.MSELoss()

    def forward(self, p_pred, p_true):
        cdf_pred = torch.cumsum(p_pred, dim=1)
        cdf_true = torch.cumsum(p_true, dim=1)
        mse_loss = self.loss_fn(cdf_pred, cdf_true)
        loss = torch.sqrt(mse_loss + 1e-6)
        return loss


# ===============================================================================
# 2. 数据加载
# ===============================================================================

CACHE_ARRAY_FILES = {
    'cr_orig': 'cr_orig.npy',
    'cr_sal': 'cr_sal.npy',
    'cr_ns': 'cr_ns.npy',
    'tm_orig': 'tm_orig.npy',
    'tm_sal': 'tm_sal.npy',
    'tm_ns': 'tm_ns.npy',
    'label': 'labels.npy',
    'image_id': 'image_ids.npy',
}


class CachedFeatureDataset(Dataset):
    """使用内存映射按样本读取预计算特征，避免整体加载到内存。"""

    def __init__(self, cache_dir):
        self.cache_dir = os.path.abspath(cache_dir)
        metadata_path = os.path.join(self.cache_dir, 'metadata.json')
        if not os.path.isfile(metadata_path):
            raise FileNotFoundError(
                f"未找到特征缓存元数据: {metadata_path}，请先运行 LR_precompute-mini-v2.py"
            )
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)

        self.num_samples = int(self.metadata['num_samples'])
        self.feature_dim = int(self.metadata['feature_dim'])
        self.num_bins = int(self.metadata.get('num_bins', 10))
        if self.feature_dim != 1024 or self.num_bins != 10:
            raise ValueError(
                f"缓存规格不匹配: feature_dim={self.feature_dim}, num_bins={self.num_bins}"
            )

        self._arrays = None
        for filename in CACHE_ARRAY_FILES.values():
            path = os.path.join(self.cache_dir, filename)
            if not os.path.isfile(path):
                raise FileNotFoundError(f"特征缓存不完整，缺少文件: {path}")

    def _ensure_open(self):
        if self._arrays is not None:
            return
        self._arrays = {
            key: np.load(
                os.path.join(self.cache_dir, filename),
                mmap_mode='r',
                allow_pickle=False,
            )
            for key, filename in CACHE_ARRAY_FILES.items()
        }
        for key, array in self._arrays.items():
            if len(array) != self.num_samples:
                raise ValueError(
                    f"缓存样本数不一致: {key}={len(array)}, metadata={self.num_samples}"
                )

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_arrays'] = None
        return state

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        self._ensure_open()
        sample = {'image_id': str(self._arrays['image_id'][idx])}
        for key in ('cr_orig', 'cr_sal', 'cr_ns', 'tm_orig', 'tm_sal', 'tm_ns', 'label'):
            array = np.array(self._arrays[key][idx], dtype=np.float32, copy=True)
            sample[key] = torch.from_numpy(array)
        return sample


def load_data():
    print(f"正在读取测试特征缓存: {TEST_CACHE_DIR}")
    test_dataset = CachedFeatureDataset(TEST_CACHE_DIR)
    return DataLoader(
        test_dataset,
        batch_size=TEST_BATCH_SIZE,
        shuffle=False,
        num_workers=3,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True,
        drop_last=False,
    )


def get_score(y_pred):
    w = torch.arange(1, 11, dtype=torch.float32, device=y_pred.device)
    w_batch = w.repeat(y_pred.size(0), 1)
    score = (y_pred * w_batch).sum(dim=1)
    score_np = score.detach().float().cpu().numpy()
    return score, score_np


def autocast_context():
    if not USE_AMP or not torch.cuda.is_available():
        return nullcontext()
    dtype = torch.bfloat16 if AMP_DTYPE.lower() == 'bfloat16' else torch.float16
    return torch.autocast(device_type='cuda', dtype=dtype, enabled=True)


def load_fusion_state_dict(model, state_dict):
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            "checkpoint 与当前共享 Branch Transformer 结构不匹配，"
            "请使用修改后的 LR_train.py 重新训练生成 checkpoint。"
        ) from exc


def binary_accuracy(prediction, target, threshold=BINARY_ACC_THRESHOLD):
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if prediction.shape != target.shape:
        raise ValueError(f"预测与标签形状不一致: {prediction.shape} vs {target.shape}")
    return float(np.mean((prediction >= threshold) == (target >= threshold)))


def numpy_emd1(prediction, target):
    pred_cdf = np.cumsum(np.asarray(prediction, dtype=np.float64), axis=1)
    target_cdf = np.cumsum(np.asarray(target, dtype=np.float64), axis=1)
    return float(np.mean(np.abs(pred_cdf - target_cdf)))


def numpy_emd2(prediction, target):
    pred_cdf = np.cumsum(np.asarray(prediction, dtype=np.float64), axis=1)
    target_cdf = np.cumsum(np.asarray(target, dtype=np.float64), axis=1)
    return float(np.sqrt(np.mean((pred_cdf - target_cdf) ** 2) + 1e-8))


def calculate_metrics(prediction_scores, target_scores, prediction_distributions, labels):
    return {
        'srcc': float(spearmanr(prediction_scores, target_scores)[0]),
        'plcc': float(pearsonr(prediction_scores, target_scores)[0]),
        'mse': float(mean_squared_error(target_scores, prediction_scores)),
        'acc': binary_accuracy(prediction_scores, target_scores),
        'emd1': numpy_emd1(prediction_distributions, labels),
        'emd2': numpy_emd2(prediction_distributions, labels),
    }


# ===============================================================================
# 3. 主测试逻辑
# ===============================================================================

def test():
    data_test = load_data()
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision('high')
    log_file_path = SAVE_FILENAME
    log_dir = os.path.dirname(log_file_path)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    with open(log_file_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Model_Name', 'SRCC', 'PLCC', 'MSE', 'ACC', 'EMD1', 'EMD2'])

    model_results = []
    reference_image_ids = None
    reference_target_scores = None
    reference_labels = None

    for i in range(1, TOTAL_ROUNDS + 1):
        model_name = f'{MODEL_BASENAME}_{i}.pt'
        model_path = os.path.join(MODEL_DIR, model_name)

        print(f"=================================================")
        print(f"======> Testing Model: {model_path}")
        print(f"=================================================")

        if not os.path.exists(model_path):
            print(f"Warning: Model file not found {model_path}. Skipping.")
            continue

        model = GIAA_model()

        checkpoint = torch.load(
            model_path,
            map_location='cuda',
            weights_only=False,
        )
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            load_fusion_state_dict(model, checkpoint['model_state_dict'])
            loss_weights = checkpoint.get(
                'loss_weights',
                checkpoint.get('learned_loss_weights'),
            )
            if loss_weights is not None:
                emd2_weight = loss_weights.get('emd2', loss_weights.get('emd'))
                print(
                    "训练使用的损失权重 -> "
                    f"EMD2: {emd2_weight:.4f}, "
                    f"Rank: {loss_weights['rank']:.4f}, "
                    f"Soft-SRCC: {loss_weights.get('soft_spearman', 0.0):.4f}, "
                    f"Score: {loss_weights['score']:.4f}"
                )
            if 'ema_decay' in checkpoint:
                print(
                    f"当前 checkpoint 为 EMA 最优权重，"
                    f"EMA decay={checkpoint['ema_decay']:.4f}, "
                    f"validation SRCC={checkpoint.get('best_validation_srcc', float('nan')):.6f}"
                )
        else:
            # 兼容仅保存 model.state_dict() 的旧 checkpoint。
            load_fusion_state_dict(model, checkpoint)

        model.cuda()
        model.eval()

        prediction_batches = []
        target_batches = []
        distribution_batches = []
        label_batches = []
        image_ids = []

        for data in data_test:
            if data is None: continue

            cr_orig = data['cr_orig'].cuda(non_blocking=True)
            cr_sal = data['cr_sal'].cuda(non_blocking=True)
            cr_non_sal = data['cr_ns'].cuda(non_blocking=True)

            tm_orig = data['tm_orig'].cuda(non_blocking=True)
            tm_sal = data['tm_sal'].cuda(non_blocking=True)
            tm_non_sal = data['tm_ns'].cuda(non_blocking=True)

            labels = data['label'].cuda(non_blocking=True)

            with torch.inference_mode(), autocast_context():
                outputs = model(cr_orig, cr_sal, cr_non_sal, tm_orig, tm_sal, tm_non_sal)
            _, prediction_np = get_score(outputs)
            _, target_np = get_score(labels)
            prediction_batches.append(prediction_np)
            target_batches.append(target_np)
            distribution_batches.append(outputs.detach().float().cpu().numpy())
            label_batches.append(labels.detach().float().cpu().numpy())
            image_ids.extend([str(image_id) for image_id in data['image_id']])

        prediction_scores = np.concatenate(prediction_batches)
        target_scores = np.concatenate(target_batches)
        prediction_distributions = np.concatenate(distribution_batches)
        labels_all = np.concatenate(label_batches)
        metrics = calculate_metrics(
            prediction_scores,
            target_scores,
            prediction_distributions,
            labels_all,
        )

        if reference_image_ids is None:
            reference_image_ids = image_ids
            reference_target_scores = target_scores
            reference_labels = labels_all
        else:
            if image_ids != reference_image_ids:
                raise ValueError('不同 checkpoint 的测试图像顺序不一致')
            if not np.allclose(target_scores, reference_target_scores, atol=1e-6):
                raise ValueError('不同 checkpoint 的测试 MOS 标签不一致')
            if not np.allclose(labels_all, reference_labels, atol=1e-6):
                raise ValueError('不同 checkpoint 的测试评分分布标签不一致')

        model_results.append({
            'model_name': model_name,
            'prediction_scores': prediction_scores,
            'prediction_distributions': prediction_distributions,
            **metrics,
        })

        print(f'--- Model: {model_name} ---')
        print(
            f"SRCC={metrics['srcc']:.6f}, PLCC={metrics['plcc']:.6f}, "
            f"MSE={metrics['mse']:.6f}, ACC={metrics['acc']:.6f}, "
            f"EMD1={metrics['emd1']:.6f}, EMD2={metrics['emd2']:.6f}"
        )

        with open(log_file_path, mode='a', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow([
                model_name,
                metrics['srcc'], metrics['plcc'], metrics['mse'],
                metrics['acc'], metrics['emd1'], metrics['emd2'],
            ])

        del model, checkpoint
        torch.cuda.empty_cache()

    if not model_results:
        raise FileNotFoundError(
            f"未找到可测试模型: {os.path.join(MODEL_DIR, MODEL_BASENAME + '_*.pt')}"
        )

    metric_names = ('srcc', 'plcc', 'mse', 'acc', 'emd1', 'emd2')
    average_metrics = {
        name: float(np.mean([result[name] for result in model_results]))
        for name in metric_names
    }

    # 真正的模型集成：先按图像平均预测分布，再重新计算全部指标。
    ensemble_distribution = np.mean(
        np.stack([result['prediction_distributions'] for result in model_results], axis=0),
        axis=0,
    )
    score_positions = np.arange(1, 11, dtype=np.float64)
    ensemble_prediction = np.sum(ensemble_distribution * score_positions[None, :], axis=1)
    ensemble_metrics = calculate_metrics(
        ensemble_prediction,
        reference_target_scores,
        ensemble_distribution,
        reference_labels,
    )

    prediction_file_path = PREDICTION_SAVE_FILENAME
    with open(prediction_file_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Image_ID', 'Predicted_Score', 'True_Score'])
        writer.writerows(
            [image_id, float(predicted_score), float(true_score)]
            for image_id, predicted_score, true_score in zip(
                reference_image_ids,
                ensemble_prediction,
                reference_target_scores,
            )
        )

    print("-------------------------------------------------")
    print(f"Mean Single-Model Results ({len(model_results)} models):")
    print(
        f"SRCC={average_metrics['srcc']:.6f}, PLCC={average_metrics['plcc']:.6f}, "
        f"MSE={average_metrics['mse']:.6f}, ACC={average_metrics['acc']:.6f}, "
        f"EMD1={average_metrics['emd1']:.6f}, EMD2={average_metrics['emd2']:.6f}"
    )
    print(f"Ensemble Results ({len(model_results)} models):")
    print(
        f"Ensemble SRCC={ensemble_metrics['srcc']:.6f}, "
        f"PLCC={ensemble_metrics['plcc']:.6f}, "
        f"MSE={ensemble_metrics['mse']:.6f}, "
        f"ACC={ensemble_metrics['acc']:.6f}, "
        f"EMD1={ensemble_metrics['emd1']:.6f}, "
        f"EMD2={ensemble_metrics['emd2']:.6f}"
    )
    print("-------------------------------------------------")

    with open(log_file_path, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Mean_Single_Models',
            *[average_metrics[name] for name in metric_names],
        ])
        writer.writerow([
            f'Ensemble_{len(model_results)}_Models',
            *[ensemble_metrics[name] for name in metric_names],
        ])

    print(f"测试完成，日志已保存；逐图预测分数已保存到: {prediction_file_path}")


if __name__ == '__main__':
    test()
