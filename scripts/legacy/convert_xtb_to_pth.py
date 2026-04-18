import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

def convert_xtb_to_pth():
    """
    将XTB数据按SMILES匹配，分为训练集和验证集，保存为pth格式
    """
    print("="*60)
    print("XTB 数据转换 - CSV 转 PTH")
    print("="*60)
    
    # 1. 读取XTB数据
    print("\n1. 读取XTB数据...")
    xtb_df = pd.read_csv("/home/liutao/pxf/data_clean/xtb_processed.csv")
    print(f"   XTB数据总行数: {len(xtb_df)}")
    
    # 创建SMILES到特征的映射
    feature_cols = [
        'N_Atoms', 'N_Heavy_Atoms', 'Molecular_Mass_amu',
        'Electronic_Energy_AU', 'Electronic_Energy_kcal_mol',
        'HOMO_eV', 'LUMO_eV', 'HOMO_LUMO_Gap_eV',
        'Dipole_Total_Debye', 'Dipole_Theta_deg', 'Dipole_Phi_deg',
        'Charge_Min', 'Charge_Max', 'Charge_Mean', 'Charge_STD', 'Charge_Range',
        'Molecular_Volume_cm3_mol'
    ]
    
    # 2. 读取训练集和测试集的SMILES
    print("\n2. 读取训练集和测试集SMILES...")
    train_df = pd.read_csv("/home/liutao/pxf/MP_new/data/multimodal_train.csv")
    test_df = pd.read_csv("/home/liutao/pxf/MP_new/data/multimodal_test.csv")
    
    # 创建SMILES到MP的映射（用于快速查找）
    train_smiles_to_mp = dict(zip(train_df['SMILES'], train_df['MP']))
    test_smiles_to_mp = dict(zip(test_df['SMILES'], test_df['MP']))
    
    train_smiles_set = set(train_df['SMILES'])
    test_smiles_set = set(test_df['SMILES'])
    
    print(f"   训练集样本数: {len(train_smiles_set)}")
    print(f"   测试集样本数: {len(test_smiles_set)}")
    
    # 3. 直接通过DataFrame操作进行匹配（更高效）
    print("\n3. 创建SMILES到XTB特征的映射...")
    smiles_to_features = {}
    for _, row in tqdm(xtb_df.iterrows(), total=len(xtb_df), desc="处理XTB数据"):
        smiles = row['SMILES']
        features = row[feature_cols].values.astype(np.float32)
        smiles_to_features[smiles] = features
    
    # 4. 匹配训练集
    print("\n4. 匹配训练集...")
    train_features = []
    train_targets = []
    train_smiles_matched = []
    
    # 找出同时在XTB和训练集中的SMILES
    common_train_smiles = train_smiles_set & set(smiles_to_features.keys())
    print(f"   训练集交集大小: {len(common_train_smiles)}")
    
    for smiles in tqdm(common_train_smiles, desc="匹配训练集"):
        train_features.append(smiles_to_features[smiles])
        train_targets.append(train_smiles_to_mp[smiles])
        train_smiles_matched.append(smiles)
    
    print(f"   训练集匹配成功: {len(train_features)}/{len(train_smiles_set)}")
    
    # 5. 匹配测试集
    print("\n5. 匹配测试集...")
    test_features = []
    test_targets = []
    test_smiles_matched = []
    
    # 找出同时在XTB和测试集中的SMILES
    common_test_smiles = test_smiles_set & set(smiles_to_features.keys())
    print(f"   测试集交集大小: {len(common_test_smiles)}")
    
    for smiles in tqdm(common_test_smiles, desc="匹配测试集"):
        test_features.append(smiles_to_features[smiles])
        test_targets.append(test_smiles_to_mp[smiles])
        test_smiles_matched.append(smiles)
    
    print(f"   测试集匹配成功: {len(test_features)}/{len(test_smiles_set)}")
    
    # 6. 转换为tensor并保存
    print("\n6. 保存为PTH格式...")
    
    # 训练集
    train_features_tensor = torch.tensor(np.array(train_features), dtype=torch.float32)
    train_targets_tensor = torch.tensor(np.array(train_targets), dtype=torch.float32)
    
    train_data = {
        'features': train_features_tensor,
        'targets': train_targets_tensor,
        'smiles': train_smiles_matched,
        'feature_names': feature_cols
    }
    
    train_save_path = "/home/liutao/pxf/MP_new/data/XTB_train.pth"
    torch.save(train_data, train_save_path)
    print(f"   训练集已保存: {train_save_path}")
    print(f"   特征形状: {train_features_tensor.shape}")
    print(f"   目标形状: {train_targets_tensor.shape}")
    
    # 测试集
    test_features_tensor = torch.tensor(np.array(test_features), dtype=torch.float32)
    test_targets_tensor = torch.tensor(np.array(test_targets), dtype=torch.float32)
    
    test_data = {
        'features': test_features_tensor,
        'targets': test_targets_tensor,
        'smiles': test_smiles_matched,
        'feature_names': feature_cols
    }
    
    test_save_path = "/home/liutao/pxf/MP_new/data/XTB_test.pth"
    torch.save(test_data, test_save_path)
    print(f"   测试集已保存: {test_save_path}")
    print(f"   特征形状: {test_features_tensor.shape}")
    print(f"   目标形状: {test_targets_tensor.shape}")
    
    # 7. 统计信息
    print("\n" + "="*60)
    print("转换完成!")
    print("="*60)
    print(f"\n特征维度: {len(feature_cols)}")
    print(f"特征列表: {feature_cols}")
    print(f"\n训练集:")
    print(f"  - 样本数: {len(train_features)}")
    print(f"  - 特征范围: {train_features_tensor.min():.4f} ~ {train_features_tensor.max():.4f}")
    print(f"  - MP范围: {train_targets_tensor.min():.4f} ~ {train_targets_tensor.max():.4f}")
    print(f"\n测试集:")
    print(f"  - 样本数: {len(test_features)}")
    print(f"  - 特征范围: {test_features_tensor.min():.4f} ~ {test_features_tensor.max():.4f}")
    print(f"  - MP范围: {test_targets_tensor.min():.4f} ~ {test_targets_tensor.max():.4f}")

if __name__ == "__main__":
    convert_xtb_to_pth()
