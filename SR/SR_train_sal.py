"""
@DESCRIPTION: Contrastive Ranking on AVA (Fixed for 12 clusters)
@AUTHOR: lty (Updated by Gemini)
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import pandas as pd
import numpy as np
import warnings
from torch.utils.data.sampler import BatchSampler
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import copy
from torch.autograd import Variable
from torch import nn
from PIL import Image
from torchvision.models.swin_transformer import swin_b, Swin_B_Weights
from torch.utils.data.dataloader import default_collate
warnings.filterwarnings("ignore")


# ================== 全局路径配置 ==================
# 训练CSV文件路径 (包含语义和美学得分)
SEMANTIC_TRAIN_CSV_PATH = r"./CR_cluster/train_CR_by_Sem_GAS.csv"

# AVA 图像数据集的根目录
IMAGE_ROOT_DIR = r'/home/user/Data/AVA/images_addition_dense/images_salient'

# 对比训练epoch
CONTRAST_EPOCH = 50

# 训练后模型的保存路径
OUTPUT_MODEL_PATH = r'./CR_model/My_CR_S-v3.pt'
# ================================================


def my_collate(batch):
    # 过滤掉批次中可能是None的项 (例如，如果__getitem__中发生错误)
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        return None # 如果整个批次都是None，返回None
    return default_collate(batch)


class Rank_Swin(nn.Module):
    def __init__(self):
        super(Rank_Swin, self).__init__()
        self.encoder_image = swin_b()
        self.encoder_image.load_state_dict(torch.load(r'../Model/swin_b-68c6b09e.pth'))
        self.encoder_image.head = nn.Sequential() # 移除分类头

        self.projector_image = nn.Sequential(nn.Linear(1024, 1024), nn.BatchNorm1d(1024), nn.ReLU(inplace=True),
                                             nn.Linear(1024, 1024))

    def forward(self, image):
        image_feature = self.encoder_image(image)
        image_feature_contrast = self.projector_image(image_feature)
        feature = nn.functional.normalize(image_feature_contrast, dim=1)  # L2归一化

        return feature

class ImageAVADataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.image_information = pd.read_csv(csv_file, sep=',')
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_information)

    def __getitem__(self, idx):
        try:
            # 检查索引是否越界 (新增加的保护)
            if idx >= len(self.image_information):
                print(f"Index {idx} out of bounds.") # 保留错误/警告
                return None

            img_id = str(self.image_information.iloc[idx, 0])
            img_path = str(os.path.join(self.root_dir, img_id + '.jpg'))
            image = Image.open(img_path).convert('RGB')

            label = self.image_information.iloc[idx, 1:11]

            semantic_label = self.image_information.iloc[idx, 14]

            y = np.array([k for k in label])

            if y.sum() == 0:
                print(f"Skipping item at index {idx} (Image: {img_id}): label sum is zero.") # 保留错误/警告
                return None

            p = y / y.sum()
            if self.transform:
                image = self.transform(image)
            sample = {'image': image, 'label': torch.from_numpy(np.float64(p)).double(), 'semantic_label': semantic_label}
            return sample

        except Exception as e:
            try:
                failed_id = str(self.image_information.iloc[idx, 0])
            except:
                failed_id = "UNKNOWN (error during .iloc lookup)"

            print(f"Error loading item at index {idx} (Image ID: {failed_id}): {e}") # 保留错误/警告
            return None


def generate_semantic_score_dic():  # 这个函数创建一个字典，按语义标签和美学得分区间组织训练数据，用于后续的平衡批次采样，在ContrastBatch中被使用
    all_data = pd.read_csv(SEMANTIC_TRAIN_CSV_PATH, sep=',')  # <--- 修改点

    semantic_label = np.array(all_data['Semantic'])
    aesthetic_score = np.array(all_data['Aesthetic'])
    score_level = [2, 3, 4, 5, 6, 7, 8, 9]
    all_data_dic = {}

    num_semantic_clusters = 12

    for semantic in range(num_semantic_clusters):
        all_data_dic[str(semantic)] = {}
        for i in range(1, len(score_level)):
            all_data_dic[str(semantic)][score_level[i - 1]] = np.array([])

    for j in range(num_semantic_clusters):
        semantic_idx = np.where(semantic_label == j)[0]
        if len(semantic_idx) == 0:
            print(f"Warning: No data found for semantic cluster {j}")
            continue

        semantic_idx_score = aesthetic_score[semantic_idx]
        for i in range(1, len(score_level)):
            semantic_idx_score_idx = np.where(
                (semantic_idx_score > score_level[i - 1])
                & (semantic_idx_score < score_level[i])
            )[0]
            if len(semantic_idx_score_idx) == 0:
                continue
            else:
                all_data_dic[str(j)][score_level[i - 1]] = semantic_idx[semantic_idx_score_idx]

    return all_data_dic


class BalancedBatchSampler(BatchSampler):  # 这个采样器确保每个批次包含来自不同语义类别和美学得分区间的平衡样本，在ContrastBatch中被使用
    def __init__(self, labels, n_samplers, sampling_num):
        self.labels = labels  # 这是一个字典，作为采样器的数据源
        self.n_samplers = n_samplers  # 定义了从每个得分区间中采样多少个样本
        self.sampling_num = sampling_num  # 决定了总共要生成多少个批次
        self.keys_with_data = [k for k, v in labels.items() if v.any()] # 找出有数据的得分区间

    def __iter__(self):
        count = 0
        while count < self.sampling_num:
            indices = []
            for key in self.keys_with_data: # 仅遍历有数据的key
                if not self.labels[key].any(): # 检查是否为空
                    continue

                if len(self.labels[key]) < self.n_samplers:  # 如果某个得分区间中的样本数量少于预期的采样数，进行有放回采样
                    sample = np.random.choice(self.labels[key], self.n_samplers, replace=True)
                else:  # 如果样本数量足够，进行无放回采样
                    sample = np.random.choice(self.labels[key], self.n_samplers, replace=False)
                indices.extend(sample)

            if indices: # 只有在indices不为空时才 yield
                yield indices  # 使函数变成一个生成器
            count += 1

    def __len__(self):
        return self.sampling_num


def ContrastBatch(semantic='0', bin_size=5, sampling_num=15):  # 平衡数据加载器
    all_data_dic = generate_semantic_score_dic()
    semantic_label = all_data_dic[semantic] # semantic 是 str

    # 检查该语义类别是否有数据
    has_data = any(len(v) > 0 for v in semantic_label.values())
    if not has_data:
        # print(f"No data for semantic cluster {semantic}, skipping batch creation.")
        return None # 如果没有数据，返回None

    train_batch_sampler = BalancedBatchSampler(semantic_label, bin_size, sampling_num)
    data_image_dir = os.path.join(IMAGE_ROOT_DIR)  # <--- 修改点

    data_image_train_label_dir = SEMANTIC_TRAIN_CSV_PATH  # <--- 修改点

    transformed_dataset_train = ImageAVADataset(
        csv_file=data_image_train_label_dir,
        root_dir=data_image_dir,
        transform=transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.RandomCrop((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
    )

    data_train = DataLoader(transformed_dataset_train, batch_sampler=train_batch_sampler, num_workers=4, # num_workers可以适当调整
                            collate_fn=my_collate)

    return data_train


# aesthetic contrast
def make_criterion(batch_size, temperature):  # 这个函数创建了一个对比损失函数，用于计算监督对比学习损失，给aesthetic_contrast函数使用

    def criterion(features, labels):  # 真正的对比损失函数

        # ==========logits================================================================================
        anchor_feature = torch.cat(torch.unbind(features, dim=1), dim=0)  # 将多视图特征展开为一维
        contrast_feature = anchor_feature
        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T), temperature)  # 计算所有特征对之间的点积相似度，并除以温度参数缩放
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)  # 找到每行的最大值，用于数值稳定性
        logits = anchor_dot_contrast - logits_max.detach()  # 减去最大值，防止指数运算溢出
        # =================================================================================================

        # ==========mask=================================================================================
        labels = labels.contiguous().view(-1, 1) # 确保labels形状正确
        mask = torch.eq(labels, labels.T).float().cuda()  # 创建布尔掩码，标记哪些样本对具有相同标签

        # 优化操作：创建负样本对的布尔掩码
        mask_rev = (1 - mask).float().cuda()

        contrast_count = features.shape[1]  # 获取对比视图的数量
        anchor_count = contrast_count  # 锚点数量等于对比视图数量

        # 扩展掩码以匹配扩展后的特征矩阵维度
        # mask = mask.repeat(anchor_count, contrast_count) # 这行在view=1时是多余的，但保留

        logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(batch_size * anchor_count).view(-1, 1).cuda(), 0)  # 创建对数掩码，将自身对比的位置设为0
        mask = mask * logits_mask  # 结合标签掩码和对数掩码，得到最终的掩码矩阵
        mask_rev = mask_rev * logits_mask # 负样本掩码也排除自身
        # =================================================================================================

        # ==========loss=================================================================================
        exp_logits = torch.exp(logits) * logits_mask  # 计算指数化的对数几率，并应用掩码
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))  # 计算对数概率（log softmax）

        mask_sum = mask.sum(1)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_sum.masked_fill(mask_sum == 0, 1) # 计算正样本对的对数概率平均值, 避免除以0

        # 优化操作：计算负样本对的对数概率平均值
        mask_rev_sum = mask_rev.sum(1)
        mean_log_prob_neg = (mask_rev * log_prob).sum(1) / mask_rev_sum.masked_fill(mask_rev_sum == 0, 1) # 计算负样本对的对数概率平均值, 避免除以0

        # α加权正样本损失 + β加权负样本损失（α/β可调，如α=1.0, β=0.5）
        alpha = 1.0
        beta = 0.5
        loss = alpha * (-mean_log_prob_pos) + beta * (mean_log_prob_neg)  # 负样本项取原值（越小越好）

        loss = loss.view(anchor_count, batch_size).mean()  # 重新调整损失形状并计算平均值
        # =================================================================================================

        return loss

    return criterion


def aesthetic_contrast(features, bin_num=5):  # 这个函数实现了美学对比学习，通过组织不同美学得分的样本进行对比学习。

    contrast_feature = []
    contrast_label = []

    if features.shape[0] % bin_num != 0:
        print(f"Feature shape {features.shape[0]} not divisible by bin_num {bin_num}. Truncating.") # 保留警告
        # 截断到最近的bin_num倍数
        new_size = (features.shape[0] // bin_num) * bin_num
        if new_size == 0:
            return 0.0 # 如果连一个bin都凑不齐，不计算损失
        features = features[:new_size]

    num_bin = int(features.shape[0] / bin_num)
    if num_bin <= 1:
        # print("Only one bin, cannot perform contrastive loss.")
        return 0.0 # 至少需要2个bin才能对比

    bin_size = [bin_num] * num_bin # [9, 9, 9, ...]

    # ====================== from left to right ======================
    for anchor in range(num_bin - 1):
        for pos in range(anchor, num_bin - 1):
            old_label = [bin_size[i] for i in range(anchor)]
            pos_label = [1] * (bin_size[anchor] + bin_size[pos])
            neg_bins = [bin_size[i] for i in range(pos + 1, num_bin)]
            neg_label = [2] * sum(neg_bins)

            if not neg_label: # 如果没有负样本，跳过
                continue

            pos_features = torch.cat([features[sum(old_label): sum(old_label) + bin_size[anchor]],
                                      features[
                                      sum(bin_size) - sum(neg_bins) - bin_size[pos]: sum(bin_size) - sum(neg_bins)]],
                                     dim=0)
            neg_features = features[sum(bin_size) - sum(neg_bins):]
            anchor_contrast_label = pos_label + neg_label
            anchor_contrast_feature = torch.cat([pos_features, neg_features], dim=0)
            contrast_feature.append(anchor_contrast_feature)
            contrast_label.append(anchor_contrast_label)

    # ====================== from right to left ======================
    for anchor in range(num_bin - 1, 0, -1):
        for pos in range(anchor, 0, -1):
            old_label = [bin_size[i] for i in range(num_bin - 1, anchor, -1)]
            pos_label = [1] * (bin_size[anchor] + bin_size[pos])
            neg_bins = [bin_size[i] for i in range(pos - 1, -1, -1)]
            neg_label = [2] * sum(neg_bins)

            if not neg_label: # 如果没有负样本，跳过
                continue

            pos_features = torch.cat(
                [
                    features[sum(bin_size) - sum(old_label) - bin_size[anchor]: sum(bin_size) - sum(old_label)],
                    features[sum(neg_bins): sum(neg_bins) + bin_size[pos]]
                ],
                dim=0
            )
            neg_features = features[0: sum(neg_bins)]
            anchor_contrast_label = pos_label + neg_label
            anchor_contrast_feature = torch.cat([pos_features, neg_features], dim=0)
            contrast_feature.append(anchor_contrast_feature)
            contrast_label.append(anchor_contrast_label)

    supCR_loss = 0.0
    if not contrast_feature:
        return supCR_loss # 如果没有生成任何对比组，返回0

    for i, (feature, label) in enumerate(zip(contrast_feature, contrast_label)):
        label = torch.from_numpy(np.array(label)).double()
        label = Variable(label.cuda())
        batch_size, _ = feature.shape

        feature = torch.unsqueeze(feature, 1)

        criterion = make_criterion(batch_size, 0.15)
        supCR_loss += criterion(feature, label)

    return supCR_loss / len(contrast_feature) # 返回平均损失，这里与旧版本的累加损失不同


def train_all():
    # parameter
    aesthetic_contrast_epoch_num = CONTRAST_EPOCH  # 原50
    best_loss = float('inf')

    # model
    model = Rank_Swin()
    model.cuda()
    model = torch.nn.DataParallel(model) # 推荐使用DataParallel
    print("Model loaded to GPU(s).")

    # optimizer
    optimizer = optim.AdamW(model.parameters(), lr=0.00001, weight_decay=5E-2)

    # save_fig = 1

    num_semantic_clusters = 12

    # aesthetic contrast
    for epoch in range(aesthetic_contrast_epoch_num):
        epoch_loss = 0.0
        print(f"\n--- Epoch {epoch+1}/{aesthetic_contrast_epoch_num} ---")
        print('*****************contrast train*****************')
        model.train()

        for semantic in range(num_semantic_clusters):
            epoch_semantic_loss = 0.0
            data_contrast = ContrastBatch(semantic=str(semantic))

            if data_contrast is None: # 检查是否成功创建了DataLoader
                continue

            for batch_idx, data in enumerate(data_contrast):
                if data is None: # 检查collate_fn是否返回了None
                    continue

                inputs = data['image']
                inputs = Variable(inputs.float().cuda())

                features = model(inputs)

                loss = aesthetic_contrast(features)

                if loss == 0.0: # 如果损失为0 (例如，批次中bin不够)
                    continue

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_semantic_loss += loss.item() # .item() 释放显存

            epoch_loss += epoch_semantic_loss

        print(f"--- Epoch {epoch+1} Total Loss: {epoch_loss:.4f} ---")

        if epoch_loss < best_loss and epoch_loss > 0: # 确保损失有效
            best_loss = epoch_loss
            # --- 以下是您要求保留的打印信息 ---
            print('**************************************************')
            print(f"NEW BEST LOSS! Epoch: {epoch+1}, Best Loss: {best_loss:.4f}")
            print('**************************************************')

            # 保存完整的CR模型
            best_model_full = copy.deepcopy(model.module)
            torch.save(best_model_full, OUTPUT_MODEL_PATH)
            print(f"Best model saved to {OUTPUT_MODEL_PATH}")


if __name__ == '__main__':
    train_all()