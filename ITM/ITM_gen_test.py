"""
@DESCRIPTION: AVA generate TM_feature (Memory-Efficient Version)
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

# ================== 全局路径配置 ==================
DATA_IMAGE_DIR = r"/home/user/Data/AVA/images"

ORIGINAL_LABEL_FILE = r"../AVA/Label/Regular_Label/test.csv"

OUTPUT_CSV_FILE = './TM_feature/test_TM_whole.csv'
# ================================================


class ImageAVADataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.image_information = pd.read_csv(csv_file, sep=',')
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_information)

    def __getitem__(self, idx):
        # 添加 try-except 块来处理可能损坏的图像文件
        try:
            img_path = str(os.path.join(self.root_dir, (str(self.image_information.iloc[idx, 0]) + '.jpg')))
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            # 注意：这里我们返回图像和它的索引，以便后续匹配
            return {'image': image, 'index': idx}
        except Exception as e:
            print(f"Warning: Skipping image at index {idx} due to error: {e}")
            return None # 如果图像加载失败，返回 None

def my_collate(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if not batch: # 如果整个批次都是坏数据，返回一个空字典
        return {}
    return default_collate(batch)


def load_data(batch_size=32):
    data_test_label_dir = ORIGINAL_LABEL_FILE # ================修改点
    data_image_dir = DATA_IMAGE_DIR

    transformed_dataset_train = ImageAVADataset(
        csv_file=data_test_label_dir,
        root_dir=data_image_dir,
        transform=transforms.Compose(
            [
                transforms.Resize((336, 336)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
    )
    # 关键：shuffle=False 保证了 DataLoader 的输出顺序和原始 CSV 文件一致
    data_train = DataLoader(transformed_dataset_train, batch_size=batch_size, shuffle=False, num_workers=0,
                            collate_fn=my_collate, drop_last=False)
    return data_train


def generate_tags():
    semantic_csv = pd.read_csv(r'./All_Tags/Sem_Tags.csv', header=None)
    aesthetic_csv = pd.read_csv(r'./All_Tags/Attr_Tags.csv', header=None)
    semantic_tags = np.array(semantic_csv)
    aesthetic_tags = np.array(aesthetic_csv)
    all_tags = np.concatenate((semantic_tags, aesthetic_tags))
    all_tags = np.squeeze(all_tags)
    print(f"all_tags.shape is :{all_tags.shape}")
    return all_tags

# 将特征生成和CSV写入合并，实现批处理，避免内存爆炸
def generate_and_write_features_in_batches():
    # 1. 准备工作：加载模型、标签和原始数据
    Text_Feature_extractor, preprocess = clip.load('ViT-L/14@336px')
    Text_Feature_extractor.float().cuda()
    model = nn.DataParallel(Text_Feature_extractor).eval()

    tags = generate_tags()
    tags = clip.tokenize(tags).cuda()

    batch_size = 64 # 和 DataLoader 的 batch_size 保持一致
    data_train = load_data(batch_size=batch_size)

    # 读取原始的 label CSV 文件，我们将逐批次地为其添加特征
    original_label_file = ORIGINAL_LABEL_FILE # ================修改点
    all_data_df = pd.read_csv(original_label_file)
    output_csv_file = OUTPUT_CSV_FILE # ================修改点

    # 2. 初始化输出文件：创建新文件并写入表头
    print(f"Initializing output file: {output_csv_file}")
    # 计算新的列顺序
    columns = list(all_data_df.columns)
    col1_index = columns.index('tag1') # 找到'tag1'的位置
    columns.insert(col1_index, 'TM_features') # 在'tag1'前插入新列
    # 创建一个空的DataFrame并写入表头
    pd.DataFrame(columns=columns).to_csv(output_csv_file, index=False)

    # 3. 循环处理每个批次
    # 使用 all_data_df.iloc 来确保我们总是能拿到当前批次对应的正确元数据
    processed_rows = 0
    for data in tqdm(data_train, desc="Generating and writing features"):
        if not data: # 如果 collate_fn 返回空字典，则跳过
            continue

        inputs = data['image'].cuda()
        indices = data['index'] # 获取当前批次的原始索引

        batch_features = []
        tag_batch_size = 1268
        for i in range(0, len(tags), tag_batch_size):
            tag_batch = tags[i:i + tag_batch_size]
            with torch.no_grad():
                logits_per_image, _ = model(inputs, tag_batch)
                batch_features.append(logits_per_image.cpu())

        # 合并当前图片批次的所有标签相似度
        batch_features = torch.cat(batch_features, dim=1) # [batch_size, num_tags]

        # 1. 先转换为 numpy 数组
        batch_numpy_array = batch_features.numpy()
        
        # 2. 使用 numpy.round 对数组中的所有元素保留6位小数
        batch_numpy_array_rounded = np.round(batch_numpy_array, decimals=6)
        
        # 3. 最后再转换为列表
        features_as_list = batch_numpy_array_rounded.tolist()

        # 4. 合并数据并追加写入
        # 从原始DataFrame中获取当前批次对应的行
        # 使用 .loc[indices] 来精确匹配索引，即使有损坏的图像被跳过
        batch_df_to_write = all_data_df.loc[indices].copy()

        # 添加新特征列
        # 直接为筛选后的 DataFrame 添加新列，它们的行数是保证一致的
        batch_df_to_write['TM_features'] = features_as_list

        batch_df_to_write['TM_features'] = batch_df_to_write['TM_features'].apply(
            lambda feature_list: '[' + ', '.join(['%.6f' % f for f in feature_list]) + ']'
        )

        # 调整列顺序以匹配我们在第2步中定义的顺序
        batch_df_to_write = batch_df_to_write[columns]

        # 以追加模式(mode='a')写入CSV，并且不写入表头(header=False)
        batch_df_to_write.to_csv(output_csv_file, mode='a', header=False, index=False)

        processed_rows += len(inputs)

    print(f"处理完成。已处理行数：{processed_rows}。特征已保存至{output_csv_file}")
    

if __name__ == '__main__':
    torch.cuda.empty_cache()
    generate_and_write_features_in_batches()