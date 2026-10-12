"""
@DESCRIPTION: Load semantic features from two files, perform clustering on each, 
             offset labels for the second file, and generate a final combined label file.
@AUTHOR: lty
"""
import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import IncrementalPCA

# ==============================================================================
# 分数与聚类部分 (Scoring and Clustering Part)
# ==============================================================================
def generate_GAS(label_csv_path): # 生成通用美学评分
    print(f"===========generate_GAS 开始 {label_csv_path}===========")

    all_data = pd.read_csv(label_csv_path)
    # 假设分数分布总是在第2到第11列 (索引1到10)
    score_distribution_df = all_data.iloc[:, 1:11]
    scores = np.arange(1, 11)
    
    numerator = score_distribution_df.multiply(scores, axis='columns').sum(axis=1)
    denominator = score_distribution_df.sum(axis=1)
    # 避免除以零
    weighted_scores = numerator / denominator.replace(0, 1)

    GAS = pd.DataFrame(weighted_scores)
    print(f"===========generate_GAS 结束 {label_csv_path}===========")

    return GAS

# ========== 代码的核心部分 ==========
def My_Cluster(feature_csv_path): # K-means聚类, 聚为6类
    # 优化操作：使用 IncrementalPCA 降低聚类前的特征维度
    print(f"===========IncrementalPCA降维 开始 {feature_csv_path}===========")
    # 1. 初始化 IncrementalPCA
    # n_components 仍然是您想要降到的维度
    # batch_size 是每次处理的数据行数，可以根据您的内存大小调整
    # 注意：536可能是一个特定的数字，这里保留它。如果特征维度小于536，它将保留所有维度。
    pca = IncrementalPCA(n_components=536, batch_size=8192) # 2048 8192

    # 2. 第一次遍历数据：分块训练 PCA 模型
    print("正在训练 IncrementalPCA 模型...")
    # 使用 chunksize 分块读取，避免一次性加载到内存
    original_shape_info = 0
    try:
        for chunk in pd.read_csv(feature_csv_path, chunksize=8192): # 2048 8192
            if original_shape_info == 0:
                original_shape_info = chunk.shape[1]
            pca.partial_fit(chunk.values)
    except Exception as e:
        print(f"PCA 训练过程中出错 {feature_csv_path}: {e}")
        return None

    print("PCA 模型训练完毕。正在转换数据...")

    # 3. 第二次遍历数据：分块转换数据
    reduced_features_list = []
    try:
        for chunk in pd.read_csv(feature_csv_path, chunksize=8192): # 2048 8192
            reduced_chunk = pca.transform(chunk.values)
            reduced_features_list.append(reduced_chunk)
    except Exception as e:
        print(f"PCA 转换数据过程中出错 {feature_csv_path}: {e}")
        return None

    # 将转换后的所有分块合并成一个大的Numpy数组
    reduced_features = np.vstack(reduced_features_list)
    
    print(f"原始特征维度: {original_shape_info}")
    print(f"PCA降维后特征维度: {reduced_features.shape[1]}")
    print("===========IncrementalPCA降维 结束===========")

    print("===========K-Means 开始 (k=6)===========")
    print("正在对降维后的特征进行K-Means聚类...")
    # 按照要求，将 n_clusters 修改为 6
    kmeans = KMeans(n_clusters=6, random_state=42, n_init='auto')
    cluster_labels = kmeans.fit_predict(reduced_features)
    print("===========K-Means 结束============")
    return cluster_labels


def process_and_combine(original_csv_path, semantic_labels, aesthetic_scores):
    """
    Reads an original CSV, adds semantic and aesthetic labels, and returns a DataFrame.
    """
    print(f"===========正在处理结果 {original_csv_path}===========")
    train_df = pd.read_csv(original_csv_path)

    # 确保长度匹配
    if len(train_df) != len(semantic_labels) or len(train_df) != len(aesthetic_scores):
        print(f"错误: {original_csv_path} 长度不匹配!")
        print(f"原始CSV长度: {len(train_df)}")
        print(f"语义标签长度: {len(semantic_labels)}")
        print(f"美学分数长度: {len(aesthetic_scores)}")
        return None
    
    # 注意：aesthetic_scores 是 DataFrame，需要转换为 Series 或取其列
    train_df['Semantic'] = semantic_labels
    train_df['Aesthetic'] = aesthetic_scores.iloc[:, 0] # 取DataFrame的第一列

    return train_df

# ==============================================================================
# 主函数 (Main Execution)
# ==============================================================================
if __name__ == '__main__':
    # 定义输入和输出文件路径
    input_feature_csv1 = '../CR_sem/CR_Sem_Feature_R.csv'
    original_csv1_path = r"../../AVA/Label/Addition_Label/Rich_Label/rich_train.csv"
    
    input_feature_csv2 = '../CR_sem/CR_Sem_Feature_S.csv'
    original_csv2_path = r"../../AVA/Label/Addition_Label/Sparse_Label/sparse_train.csv"
    
    output_csv_path = './train_CR_by_Sem_GAS.csv'

    # 检查所有输入文件是否存在
    files_to_check = [input_feature_csv1, original_csv1_path, input_feature_csv2, original_csv2_path]
    files_exist = True
    for f in files_to_check:
        if not os.path.exists(f):
            print(f"错误: 输入文件 '{f}' 未找到。")
            files_exist = False
    
    if not files_exist:
        print("一个或多个输入文件缺失。正在退出。")
    else:
        print("所有输入文件均已找到。开始处理...")
        
        # 1. 处理第一组数据 (Rich)
        print("\n--- 正在处理第 1 组 (Rich) ---")
        Aesthetic_scores1 = generate_GAS(original_csv1_path)
        semantic_labels1 = My_Cluster(input_feature_csv1)
        
        if semantic_labels1 is not None:
            df1 = process_and_combine(original_csv1_path, semantic_labels1, Aesthetic_scores1)
        else:
            df1 = None
            print("因聚类错误，跳过第 1 组。")

        # 2. 处理第二组数据 (Sparse)
        print("\n--- 正在处理第 2 组 (Sparse) ---")
        Aesthetic_scores2 = generate_GAS(original_csv2_path)
        semantic_labels2 = My_Cluster(input_feature_csv2)
        
        if semantic_labels2 is not None:
            # 关键步骤：将第二组的标签 (0-5) 偏移到 (6-11)
            print("正在偏移第 2 组的语义标签 (加 6)...")
            semantic_labels2 = semantic_labels2 + 6
            
            df2 = process_and_combine(original_csv2_path, semantic_labels2, Aesthetic_scores2)
        else:
            df2 = None
            print("因聚类错误，跳过第 2 组。")

        # 3. 合并所有数据并保存
        if df1 is not None and df2 is not None:
            print("\n--- 正在合并结果 ---")
            combined_df = pd.concat([df1, df2], ignore_index=True)
            combined_df.to_csv(output_csv_path, index=False, float_format='%.6f')
            print(f"最终合并的 CSV 已保存至 {output_csv_path}")
            print(f"最终文件总行数: {len(combined_df)}")
        else:
            print("\n一组或两组数据处理失败。未生成最终文件。")