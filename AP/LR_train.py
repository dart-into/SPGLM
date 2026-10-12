"""
@DESCRIPTION: 基于双层 Transformer 融合 (Hierarchical CLS Fusion) 的 GIAA 模型训练脚本
@AUTHOR: lty
@MODIFIED: 双层 Transformer 架构：Branch Transformer (3路) -> Global Transformer -> Regression
@UPDATE: 使用归一化固定损失、SRCC 学习率调度、EMA 验证和固定随机种子训练
"""

import os
import sys
import json
import copy
import random
import warnings
from contextlib import nullcontext
import numpy as np
import csv  # [新增] 引入 csv 模块
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_squared_error
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# --- 超参数设置 ---
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
USE_GPU = torch.cuda.is_available()

NUM_EPOCHS = 30
BATCH_SIZE = 128
LEARNING_RATE = 3e-4
MIN_LEARNING_RATE = 1e-6
WARMUP_EPOCHS = 2
TOTAL_ROUNDS = 10
ROUND_SEEDS = (42, 3407, 2026, 123, 456, 789, 1024, 2024, 2025, 31415)
EARLY_STOPPING_PATIENCE = 5
WEIGHT_DECAY = 2e-2
EMA_DECAY = 0.995
LOSS_SCALE_EMA_DECAY = 0.99
LOSS_SCALE_CALIBRATION_EPOCHS = 2
LOSS_WEIGHTS = (0.25, 0.35, 0.20, 0.20)  # EMD2 / Rank / Soft-SRCC / Score
PAIRWISE_MIN_GAP = 0.18
PAIRWISE_TEMPERATURE = 0.72
SOFT_SPEARMAN_TEMPERATURE = 0.36
USE_AMP = True
AMP_DTYPE = 'bfloat16'

# --- LR_precompute-mini-v2.py 生成的缓存目录 ---
TRAIN_CACHE_DIR = './LR_feature/train'
VALID_CACHE_DIR = './LR_feature/val'

# --- 输出模型路径 ---
SAVE_MODEL_PATH = './LR_model/LR_Train.pt'

# --- 输出日志路径 ---
CSV_LOG_PATH = './Log/LR_Train.csv'

warnings.filterwarnings("ignore")


# ===============================================================================
# 1. 模型定义 (核心区域: 双层 Transformer 融合)
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

        # 合并三路到 batch 维度，只调用一次共享 Transformer，减少参数和 kernel 调用。
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


# ===============================================================================
# 2. 数据加载工具
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
    print(f"正在读取训练特征缓存: {TRAIN_CACHE_DIR}")
    train_dataset = CachedFeatureDataset(TRAIN_CACHE_DIR)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=USE_GPU,
        persistent_workers=True,
    )

    print(f"正在读取验证特征缓存: {VALID_CACHE_DIR}")
    valid_dataset = CachedFeatureDataset(VALID_CACHE_DIR)
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=USE_GPU,
        persistent_workers=True,
    )

    return train_loader, valid_loader


# ===============================================================================
# 3. 辅助函数与验证函数
# ===============================================================================

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

def get_score(y_pred):
    w = torch.arange(1, 11, dtype=torch.float32).to(y_pred.device)
    w_batch = w.repeat(y_pred.size(0), 1)
    score = (y_pred * w_batch).sum(dim=1)
    return score


def autocast_context(device):
    if not USE_AMP or device.type != 'cuda':
        return nullcontext()
    dtype = torch.bfloat16 if AMP_DTYPE.lower() == 'bfloat16' else torch.float16
    return torch.autocast(device_type='cuda', dtype=dtype, enabled=True)


def pairwise_rank_loss(
    pred_score,
    true_score,
    min_gap=PAIRWISE_MIN_GAP,
    temperature=PAIRWISE_TEMPERATURE,
):
    """忽略近似平分样本，并按真实 MOS 差距加权排序监督。"""
    pred_diff = pred_score[:, None] - pred_score[None, :]
    true_diff = true_score[:, None] - true_score[None, :]
    upper_triangle = torch.triu(
        torch.ones_like(true_diff, dtype=torch.bool), diagonal=1
    )
    valid_pairs = upper_triangle & (true_diff.abs() >= min_gap)
    if not valid_pairs.any():
        return pred_score.sum() * 0.0

    selected_true_diff = true_diff[valid_pairs]
    rank_target = selected_true_diff.sign()
    signed_pred_diff = rank_target * pred_diff[valid_pairs]
    pair_weights = selected_true_diff.abs()
    pair_weights = pair_weights / pair_weights.mean().clamp_min(1e-6)
    pair_losses = nn.functional.softplus(-signed_pred_diff / temperature)
    return torch.mean(pair_weights * pair_losses)


def differentiable_soft_rank(values, temperature=SOFT_SPEARMAN_TEMPERATURE):
    differences = values[:, None] - values[None, :]
    return 1.0 + torch.sigmoid(-differences / temperature).sum(dim=1) - 0.5


def negative_pearson_loss(x, y):
    x = x - x.mean()
    y = y - y.mean()
    denominator = torch.sqrt(torch.sum(x * x) * torch.sum(y * y)).clamp_min(1e-8)
    correlation = torch.sum(x * y) / denominator
    return 1.0 - correlation.clamp(-1.0, 1.0)


def soft_spearman_loss(pred_score, true_score):
    if pred_score.numel() < 3:
        return pred_score.sum() * 0.0
    pred_rank = differentiable_soft_rank(pred_score)
    true_rank = differentiable_soft_rank(true_score.detach()).detach()
    return negative_pearson_loss(pred_rank, true_rank)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class NormalizedFixedMultiTaskLoss(nn.Module):
    """先校准损失尺度，再冻结尺度，避免训练后期有效梯度持续放大。"""

    def __init__(self, weights=LOSS_WEIGHTS, scale_decay=LOSS_SCALE_EMA_DECAY):
        super().__init__()
        weights_tensor = torch.tensor(weights, dtype=torch.float32)
        if not torch.isclose(weights_tensor.sum(), torch.tensor(1.0)):
            raise ValueError(f"损失权重之和必须为 1，实际为 {weights}")
        self.scale_decay = float(scale_decay)
        self.register_buffer('weights', weights_tensor)
        self.register_buffer('ema_scales', torch.ones(weights_tensor.numel(), dtype=torch.float32))
        self.register_buffer('initialized', torch.tensor(False, dtype=torch.bool))
        self.register_buffer('scales_frozen', torch.tensor(False, dtype=torch.bool))

    def forward(self, losses):
        if len(losses) != self.weights.numel():
            raise ValueError(f"期望 {self.weights.numel()} 个损失，实际得到 {len(losses)} 个")

        loss_tensor = torch.stack([loss.float() for loss in losses])
        detached_losses = loss_tensor.detach().clamp_min(1e-8)
        if not bool(self.scales_frozen.item()):
            with torch.no_grad():
                if not bool(self.initialized.item()):
                    self.ema_scales.copy_(detached_losses)
                    self.initialized.fill_(True)
                else:
                    self.ema_scales.mul_(self.scale_decay).add_(
                        detached_losses,
                        alpha=1.0 - self.scale_decay,
                    )

        normalized_losses = loss_tensor / self.ema_scales.detach().clamp_min(1e-8)
        total_loss = torch.sum(self.weights * normalized_losses)
        return total_loss, normalized_losses.detach()

    def get_weights(self):
        return self.weights.detach().cpu().tolist()

    def get_scales(self):
        return self.ema_scales.detach().cpu().tolist()

    def freeze_scales(self):
        self.scales_frozen.fill_(True)


class ModelEMA:
    """维护训练模型参数的指数滑动平均，用于验证和保存最优 checkpoint。"""

    def __init__(self, model, decay=EMA_DECAY):
        self.decay = float(decay)
        self.module = copy.deepcopy(model).eval()
        for parameter in self.module.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        model_state = model.state_dict()
        for name, ema_value in self.module.state_dict().items():
            model_value = model_state[name].detach().to(device=ema_value.device)
            if ema_value.is_floating_point():
                ema_value.mul_(self.decay).add_(model_value, alpha=1.0 - self.decay)
            else:
                ema_value.copy_(model_value)

# [修改] 返回所有指标以便写入 CSV
def validate(model, data_loader, criterion_emd2, device):
    model.eval()
    predicts_scores = []
    ratings_scores = []
    total_emd2_loss = 0.0
    num_batches = 0

    for data in data_loader:
        if data is None: continue

        cr_orig = data['cr_orig'].cuda(non_blocking=True)
        cr_sal = data['cr_sal'].cuda(non_blocking=True)
        cr_non_sal = data['cr_ns'].cuda(non_blocking=True)

        tm_orig = data['tm_orig'].cuda(non_blocking=True)
        tm_sal = data['tm_sal'].cuda(non_blocking=True)
        tm_non_sal = data['tm_ns'].cuda(non_blocking=True)

        labels = data['label'].cuda(non_blocking=True)

        with torch.no_grad(), autocast_context(device):
            outputs = model(cr_orig, cr_sal, cr_non_sal, tm_orig, tm_sal, tm_non_sal)
            emd2_loss = criterion_emd2(outputs, labels)
            total_emd2_loss += emd2_loss.item()
            num_batches += 1

            predicted_mean_scores = get_score(outputs)
            true_mean_scores = get_score(labels)

            predicts_scores.extend(predicted_mean_scores.cpu().numpy())
            ratings_scores.extend(true_mean_scores.cpu().numpy())

    if len(predicts_scores) < 2:
        return 0.0, 0.0, 0.0, 0.0

    plcc = pearsonr(predicts_scores, ratings_scores)[0]
    srcc = spearmanr(predicts_scores, ratings_scores)[0]
    mse = mean_squared_error(predicts_scores, ratings_scores)
    avg_emd2_loss = total_emd2_loss / max(1, num_batches)

    print(f'验证结果 -> EMD2: {avg_emd2_loss:.4f}, SRCC: {srcc:.4f}, PLCC: {plcc:.4f}, MSE: {mse:.4f}')
    return srcc, plcc, mse, avg_emd2_loss


# ===============================================================================
# 4. 主训练逻辑与执行
# ===============================================================================

def main():
    print("正在初始化模型和数据加载器...")
    if len(ROUND_SEEDS) != TOTAL_ROUNDS:
        raise ValueError(
            f"ROUND_SEEDS 数量必须等于 TOTAL_ROUNDS: "
            f"{len(ROUND_SEEDS)} vs {TOTAL_ROUNDS}"
        )
    train_loader, valid_loader = load_data()
    device = torch.device("cuda" if USE_GPU else "cpu")
    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision('high')

    # [新增] 初始化记录数据的 CSV
    log_dir = os.path.dirname(CSV_LOG_PATH)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    with open(CSV_LOG_PATH, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Round', 'Seed', 'Epoch', 'Learning_Rate',
            'SRCC', 'PLCC', 'MSE', 'EMD2',
            'Loss_Weight_EMD2', 'Loss_Weight_Rank',
            'Loss_Weight_Soft_Spearman', 'Loss_Weight_Score',
            'Loss_Scale_EMA_EMD2', 'Loss_Scale_EMA_Rank',
            'Loss_Scale_EMA_Soft_Spearman', 'Loss_Scale_EMA_Score'
        ])

    for round_num in range(1, TOTAL_ROUNDS + 1):
        round_seed = ROUND_SEEDS[round_num - 1]
        set_random_seed(round_seed)
        print(
            f"\n{'=' * 25} 开始第 {round_num}/{TOTAL_ROUNDS} 轮训练 "
            f"(seed={round_seed}) {'=' * 25}\n"
        )

        model = GIAA_model().to(device)
        ema = ModelEMA(model, decay=EMA_DECAY)

        criterion_emd2 = EMD2Loss().to(device)
        loss_balancer = NormalizedFixedMultiTaskLoss().to(device)

        optimizer_parameters = [
            {
                'params': [p for p in model.parameters() if p.requires_grad],
                'weight_decay': WEIGHT_DECAY
            }
        ]
        try:
            optimizer = torch.optim.AdamW(
                optimizer_parameters,
                lr=LEARNING_RATE,
                fused=device.type == 'cuda',
            )
        except (TypeError, RuntimeError):
            optimizer = torch.optim.AdamW(optimizer_parameters, lr=LEARNING_RATE)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=0,
            threshold=1e-4,
            threshold_mode='abs',
            cooldown=1,
            min_lr=MIN_LEARNING_RATE,
        )

        print("\n开始训练...")
        best_srcc = -1.0

        # --- 早停设置 ---
        patience = EARLY_STOPPING_PATIENCE
        early_stopping_counter = 0

        for epoch in range(NUM_EPOCHS):
            if epoch < WARMUP_EPOCHS:
                warmup_lr = LEARNING_RATE * float(epoch + 1) / float(WARMUP_EPOCHS)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = warmup_lr

            model.train()
            running_loss = 0.0

            for i, data in enumerate(train_loader):
                if data is None: continue

                cr_orig = data['cr_orig'].cuda(non_blocking=True)
                cr_sal = data['cr_sal'].cuda(non_blocking=True)
                cr_non_sal = data['cr_ns'].cuda(non_blocking=True)

                tm_orig = data['tm_orig'].cuda(non_blocking=True)
                tm_sal = data['tm_sal'].cuda(non_blocking=True)
                tm_non_sal = data['tm_ns'].cuda(non_blocking=True)

                labels = data['label'].cuda(non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                with autocast_context(device):
                    outputs = model(cr_orig, cr_sal, cr_non_sal, tm_orig, tm_sal, tm_non_sal)

                    predicted_mean_scores = get_score(outputs)
                    true_mean_scores = get_score(labels)

                    loss_emd2 = criterion_emd2(outputs, labels)
                    loss_rank = pairwise_rank_loss(predicted_mean_scores, true_mean_scores)
                    loss_soft_spearman = soft_spearman_loss(
                        predicted_mean_scores,
                        true_mean_scores,
                    )
                    loss_score = nn.functional.smooth_l1_loss(
                        predicted_mean_scores, true_mean_scores
                    )
                    loss, normalized_losses = loss_balancer(
                        [loss_emd2, loss_rank, loss_soft_spearman, loss_score]
                    )
                loss.backward()
                optimizer.step()
                ema.update(model)

                running_loss += loss.item()

                if (i + 1) % 250 == 0:
                    average_loss = running_loss / (i + 1)
                    current_weights = loss_balancer.get_weights()
                    current_scales = loss_balancer.get_scales()
                    print(
                        f'Epoch [{epoch + 1}/{NUM_EPOCHS}], '
                        f'Step [{i + 1}/{len(train_loader)}], '
                        f'Avg Normalized Loss: {average_loss:.4f}, '
                        f'Weights(EMD2/Rank/SoftSRCC/Score): '
                        f'{current_weights[0]:.2f}/{current_weights[1]:.2f}/'
                        f'{current_weights[2]:.2f}/{current_weights[3]:.2f}, '
                        f'EMA Scales: '
                        f'{current_scales[0]:.4f}/{current_scales[1]:.4f}/'
                        f'{current_scales[2]:.4f}/{current_scales[3]:.4f}'
                    )

            if epoch + 1 == LOSS_SCALE_CALIBRATION_EPOCHS:
                loss_balancer.freeze_scales()
                print(
                    f"损失尺度校准完成，将从 Epoch {epoch + 2} 起固定 EMA scales: "
                    f"{loss_balancer.get_scales()}"
                )

            print(f"\n--- Epoch {epoch + 1} EMA 模型验证 ---")
            current_srcc, current_plcc, current_mse, current_emd2 = validate(
                ema.module, valid_loader, criterion_emd2, device
            )
            fixed_weights = loss_balancer.get_weights()
            loss_scales = loss_balancer.get_scales()
            print(
                "固定归一化损失权重 -> "
                f"EMD2: {fixed_weights[0]:.2f}, "
                f"Rank: {fixed_weights[1]:.2f}, "
                f"Soft-SRCC: {fixed_weights[2]:.2f}, "
                f"Score: {fixed_weights[3]:.2f}"
            )

            current_lr = optimizer.param_groups[0]['lr']

            with open(CSV_LOG_PATH, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    round_num, round_seed, epoch + 1, current_lr,
                    current_srcc, current_plcc, current_mse, current_emd2,
                    *fixed_weights, *loss_scales
                ])

            if epoch + 1 > WARMUP_EPOCHS:
                scheduler.step(current_srcc)
            next_lr = optimizer.param_groups[0]['lr']
            print(
                f"Epoch {epoch + 1} 完成。验证时学习率: {current_lr:.2e}，"
                f"下一 Epoch 学习率: {next_lr:.2e}"
            )

            if current_srcc > best_srcc + 1e-4:
                best_srcc = current_srcc
                early_stopping_counter = 0 # 重置计数器
                base_path, extension = os.path.splitext(SAVE_MODEL_PATH)
                round_save_path = f"{base_path}_{round_num}{extension}"
                print(f"新最优 SRCC: {best_srcc:.4f}。保存模型到 {round_save_path}")
                torch.save({
                    # 测试默认加载 EMA 权重；同时保留原始模型权重便于后续恢复训练。
                    'model_state_dict': ema.module.state_dict(),
                    'raw_model_state_dict': model.state_dict(),
                    'loss_balancer_state_dict': loss_balancer.state_dict(),
                    'loss_weights': {
                        'emd2': fixed_weights[0],
                        'rank': fixed_weights[1],
                        'soft_spearman': fixed_weights[2],
                        'score': fixed_weights[3]
                    },
                    'loss_scales': {
                        'emd2': loss_scales[0],
                        'rank': loss_scales[1],
                        'soft_spearman': loss_scales[2],
                        'score': loss_scales[3]
                    },
                    'ema_decay': EMA_DECAY,
                    'best_validation_srcc': best_srcc,
                    'round': round_num,
                    'seed': round_seed,
                    'epoch': epoch + 1,
                    'training_config': {
                        'learning_rate': LEARNING_RATE,
                        'min_learning_rate': MIN_LEARNING_RATE,
                        'warmup_epochs': WARMUP_EPOCHS,
                        'weight_decay': WEIGHT_DECAY,
                        'early_stopping_patience': EARLY_STOPPING_PATIENCE,
                        'ema_decay': EMA_DECAY,
                        'loss_scale_ema_decay': LOSS_SCALE_EMA_DECAY,
                        'loss_scale_calibration_epochs': LOSS_SCALE_CALIBRATION_EPOCHS,
                        'pairwise_min_gap': PAIRWISE_MIN_GAP,
                        'pairwise_temperature': PAIRWISE_TEMPERATURE,
                        'soft_spearman_temperature': SOFT_SPEARMAN_TEMPERATURE,
                        'batch_size': BATCH_SIZE,
                    }
                }, round_save_path)
            else:
                early_stopping_counter += 1
                print(f"早停计数: {early_stopping_counter} / {patience}")
                if early_stopping_counter >= patience:
                    print(f"早停触发: SRCC 在连续 {patience} 个 Epoch 内未提升，结束本轮训练。")
                    break

            print("-------------------------------------------\n")

        print(f"\n{'=' * 25} 第 {round_num}/{TOTAL_ROUNDS} 轮训练结束 {'=' * 25}\n")


if __name__ == '__main__':
    save_dir = os.path.dirname(SAVE_MODEL_PATH)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    main()
