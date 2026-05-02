"""
计算精简版RDKit 3D特征
保留：分子质量、电子性质、表面积等核心3D特征
"""
import os
import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from tqdm import tqdm

RDLogger.DisableLog('rdApp.*')

# 核心3D特征列表（精简版）
CORE_3D_FEATURES = {
    # 分子质量相关
    'MolWt': '分子量',
    'HeavyAtomMolWt': '重原子分子量',
    'ExactMolWt': '精确分子量',
    
    # 电子性质
    'NumValenceElectrons': '价电子数',
    'NumRadicalElectrons': '自由基电子数',
    'MaxPartialCharge': '最大部分电荷',
    'MinPartialCharge': '最小部分电荷',
    'MaxAbsPartialCharge': '最大绝对部分电荷',
    'MinAbsPartialCharge': '最小绝对部分电荷',
    
    # 表面积相关
    'LabuteASA': 'Labute表面积',
    'TPSA': '拓扑极性表面积',
    
    # 分子体积/密度相关
    'MolLogP': 'LogP',
    'MolMR': '摩尔折射率',
    
    # 原子计数
    'HeavyAtomCount': '重原子数',
    'NHOHCount': 'N+O+H计数',
    'NOCount': 'N+O计数',
    'NumHAcceptors': '氢键受体数',
    'NumHDonors': '氢键供体数',
    'NumHeteroatoms': '杂原子数',
    'NumRotatableBonds': '可旋转键数',
    
    # 环相关
    'RingCount': '环数',
    'NumAliphaticRings': '脂肪环数',
    'NumAromaticRings': '芳香环数',
    
    # 其他3D相关
    'FractionCSP3': 'sp3碳比例',
    'qed': 'QED药物相似性',
}

def compute_rdkit_features(smiles_list, feature_names):
    """
    计算指定RDKit特征
    
    Args:
        smiles_list: SMILES字符串列表
        feature_names: 特征名称列表
    
    Returns:
        features: numpy数组，形状为 (n_samples, n_features)
        valid_indices: 有效样本的索引
    """
    features = []
    valid_indices = []
    
    for i, smiles in enumerate(tqdm(smiles_list, desc="计算RDKit特征")):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            
            # 计算所有特征
            feat_vector = []
            for feat_name in feature_names:
                try:
                    feat_func = getattr(Descriptors, feat_name)
                    value = feat_func(mol)
                    # 处理NaN和Inf
                    if np.isnan(value) or np.isinf(value):
                        value = 0.0
                except:
                    value = 0.0
                feat_vector.append(value)
            
            features.append(feat_vector)
            valid_indices.append(i)
        except Exception as e:
            print(f"处理SMILES {smiles} 时出错: {e}")
            continue
    
    return np.array(features, dtype=np.float32), valid_indices

def main():
    print("="*60)
    print("RDKit 3D 特征计算 (精简版)")
    print("="*60)
    
    feature_names = list(CORE_3D_FEATURES.keys())
    print(f"\n将计算 {len(feature_names)} 个核心3D特征:")
    for i, (name, desc) in enumerate(CORE_3D_FEATURES.items(), 1):
        print(f"  {i:2d}. {name:<25} ({desc})")
    
    # 读取训练集和测试集
    print("\n" + "="*60)
    print("读取数据集...")
    train_df = pd.read_csv("data/raw/multimodal_train.csv")
    test_df = pd.read_csv("data/raw/multimodal_test.csv")
    
    train_smiles = train_df['SMILES'].tolist()
    test_smiles = test_df['SMILES'].tolist()
    
    print(f"训练集样本数: {len(train_smiles)}")
    print(f"测试集样本数: {len(test_smiles)}")
    
    # 计算训练集特征
    print("\n" + "="*60)
    print("计算训练集RDKit特征...")
    train_features, train_valid_idx = compute_rdkit_features(train_smiles, feature_names)
    print(f"训练集特征形状: {train_features.shape}")
    print(f"有效样本数: {len(train_valid_idx)}/{len(train_smiles)}")
    
    # 计算测试集特征
    print("\n" + "="*60)
    print("计算测试集RDKit特征...")
    test_features, test_valid_idx = compute_rdkit_features(test_smiles, feature_names)
    print(f"测试集特征形状: {test_features.shape}")
    print(f"有效样本数: {len(test_valid_idx)}/{len(test_smiles)}")
    
    # 保存特征
    print("\n" + "="*60)
    print("保存特征文件...")
    
    # 保存为npy格式
    train_npy_path = "data/processed/rdkit3d_train.npy"
    test_npy_path = "data/processed/rdkit3d_test.npy"
    
    np.save(train_npy_path, train_features)
    np.save(test_npy_path, test_features)
    
    print(f"训练集特征已保存: {train_npy_path}")
    print(f"测试集特征已保存: {test_npy_path}")
    
    # 保存为pth格式
    train_pth_path = "data/processed/rdkit3d_train.pth"
    test_pth_path = "data/processed/rdkit3d_test.pth"
    
    train_data = {
        'features': torch.FloatTensor(train_features),
        'smiles': [train_smiles[i] for i in train_valid_idx],
        'feature_names': feature_names
    }
    test_data = {
        'features': torch.FloatTensor(test_features),
        'smiles': [test_smiles[i] for i in test_valid_idx],
        'feature_names': feature_names
    }
    
    torch.save(train_data, train_pth_path)
    torch.save(test_data, test_pth_path)
    
    print(f"训练集特征已保存: {train_pth_path}")
    print(f"测试集特征已保存: {test_pth_path}")
    
    # 统计信息
    print("\n" + "="*60)
    print("特征统计信息:")
    print("="*60)
    print(f"\n训练集:")
    print(f"  样本数: {train_features.shape[0]}")
    print(f"  特征维度: {train_features.shape[1]}")
    print(f"  特征值范围: {train_features.min():.4f} ~ {train_features.max():.4f}")
    print(f"  特征均值: {train_features.mean():.4f}")
    print(f"  特征标准差: {train_features.std():.4f}")
    
    print(f"\n测试集:")
    print(f"  样本数: {test_features.shape[0]}")
    print(f"  特征维度: {test_features.shape[1]}")
    print(f"  特征值范围: {test_features.min():.4f} ~ {test_features.max():.4f}")
    print(f"  特征均值: {test_features.mean():.4f}")
    print(f"  特征标准差: {test_features.std():.4f}")
    
    # 各特征统计
    print("\n" + "="*60)
    print("各特征统计:")
    print("="*60)
    for i, feat_name in enumerate(feature_names):
        train_feat = train_features[:, i]
        print(f"{feat_name:<25}: 训练集 {train_feat.min():8.2f} ~ {train_feat.max():8.2f}, "
              f"均值 {train_feat.mean():8.2f}, 标准差 {train_feat.std():8.2f}")

if __name__ == "__main__":
    main()
