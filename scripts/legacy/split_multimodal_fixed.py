import csv
import numpy as np
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import RDLogger
import pandas as pd
import os

# 完全屏蔽RDKit日志
RDLogger.DisableLog('rdApp.*')

def get_scaffold(smiles, include_chirality=False):
    """获取分子的骨架（Murcko骨架）"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        # 使用正确的 RDKit 骨架提取 API
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        if scaffold is None:
            return smiles
            
        # 致命修复：RDKit 中控制手性/异构的参数名是 isomericSmiles
        scaffold_smiles = Chem.MolToSmiles(scaffold, isomericSmiles=include_chirality)
        
        # 处理无环直链分子返回空字符串的情况，回退到原SMILES
        if not scaffold_smiles:
            return smiles
            
        return scaffold_smiles
    except Exception as e:
        # 如果依然有无法提取的极端分子，回退到原SMILES
        return smiles

def scaffold_split_for_multimodal(input_file, output_dir, test_size=4000, num_bins=10, seed=42):
    """
    严格的骨架分割：确保test中的所有骨架都不在train中出现
    同时使用分位数分层抽样，确保 MP 分布高度一致，且测试集大小可控
    """
    np.random.seed(seed)
    
    # 读取数据
    print(f"读取数据: {input_file}")
    data = pd.read_csv(input_file)
    print(f"总数据量: {len(data)} 条")
    
    smiles_list = data['SMILES'].tolist()
    
    # 提取骨架
    print("提取分子骨架...")
    scaffold_to_indices = defaultdict(list)
    
    for i, smi in enumerate(smiles_list):
        if i % 50000 == 0 and i > 0:
            print(f"  已处理 {i}/{len(smiles_list)} 个分子...")
        scaffold = get_scaffold(smi)
        if scaffold is not None:
            scaffold_to_indices[scaffold].append(i)
    
    unique_scaffolds = len(scaffold_to_indices)
    print(f"提取完成! 共有 {unique_scaffolds} 个唯一骨架")
    
    # ---------------- 核心分布控制逻辑 ----------------
    
    # 1. 计算全局数据的 MP 分布和分位数边界
    valid_mps = data['MP'].dropna()
    total_valid = len(valid_mps)
    
    # 根据全局数据划分为 num_bins 个真实的数值区间
    bin_edges = np.percentile(valid_mps, np.linspace(0, 100, num_bins + 1))
    bin_edges[0] -= 1e-5  # 确保包含最小值
    bin_edges[-1] += 1e-5 # 确保包含最大值
    
    # 计算每个区间中包含的全局分子总数
    mol_bins = np.digitize(valid_mps, bin_edges) - 1
    mol_bins = np.clip(mol_bins, 0, num_bins - 1)
    global_bin_counts = np.bincount(mol_bins, minlength=num_bins)
    
    # 根据全局分布比例，计算测试集在每个区间的目标分子数
    test_ratio = test_size / total_valid
    target_bin_counts = global_bin_counts * test_ratio
    
    # 2. 计算每个骨架的信息，并将其分配到对应的区间
    print("计算骨架的MP分布并归类...")
    bins_dict = {i: [] for i in range(num_bins)}
    
    for scaffold, indices in scaffold_to_indices.items():
        median_mp = data.iloc[indices]['MP'].median()
        
        # 处理可能的 NaN
        if pd.isna(median_mp):
            bin_idx = np.random.randint(0, num_bins)
        else:
            bin_idx = np.digitize(median_mp, bin_edges) - 1
            bin_idx = max(0, min(bin_idx, num_bins - 1))
            
        bins_dict[bin_idx].append({
            'scaffold': scaffold,
            'indices': indices,
            'size': len(indices)
        })
    
    # 3. 在每个区间内进行受限背包采样
    test_scaffold_sets = []
    train_scaffold_sets = []
    
    for i in range(num_bins):
        scaffolds_in_bin = bins_dict[i]
        # 在区间内随机打乱，避免选择偏差
        np.random.shuffle(scaffolds_in_bin)
        
        target_size_for_this_bin = target_bin_counts[i]
        current_bin_test_size = 0
        
        for info in scaffolds_in_bin:
            # 如果当前区间还没有达到目标大小，考虑加入测试集
            if current_bin_test_size < target_size_for_this_bin:
                # 溢出保护：如果是该区间第一个骨架，或者加入后不超过目标大小的1.5倍，则允许加入
                # 避免单一巨大骨架毁掉整个区间的分布比例
                if current_bin_test_size == 0 or (current_bin_test_size + info['size'] <= target_size_for_this_bin * 1.5):
                    test_scaffold_sets.append(info['indices'])
                    current_bin_test_size += info['size']
                    continue
            
            # 不满足条件或配额已满，全部进入训练集
            train_scaffold_sets.append(info['indices'])
    
    # 展开索引
    test_indices = [idx for scaffold_set in test_scaffold_sets for idx in scaffold_set]
    train_indices = [idx for scaffold_set in train_scaffold_sets for idx in scaffold_set]
    
    print(f"\n分割结果:")
    print(f"  测试集: {len(test_indices)} 个分子 ({len(test_scaffold_sets)} 个骨架)")
    print(f"  训练集: {len(train_indices)} 个分子 ({len(train_scaffold_sets)} 个骨架)")
    print(f"  总计: {len(test_indices) + len(train_indices)} 个分子")
    
    # 验证：确保没有骨架重叠
    print("\n验证骨架分割互斥性...")
    test_scaffolds = set()
    train_scaffolds = set()
    
    for scaffold_set in test_scaffold_sets:
        rep_idx = scaffold_set[0]
        test_scaffolds.add(get_scaffold(smiles_list[rep_idx]))
    
    for scaffold_set in train_scaffold_sets:
        rep_idx = scaffold_set[0]
        train_scaffolds.add(get_scaffold(smiles_list[rep_idx]))
    
    overlap = test_scaffolds & train_scaffolds
    if len(overlap) > 0:
        print(f"  ✗ 错误: 发现 {len(overlap)} 个骨架同时存在于train和test中!")
        raise ValueError("骨架分割失败，存在重叠!")
    else:
        print(f"  ✓ 验证通过: train和test骨架完全互斥，0重叠")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存测试集
    test_data = data.iloc[test_indices][['SMILES', 'MP']]
    test_file = os.path.join(output_dir, 'multimodal_test.csv')
    test_data.to_csv(test_file, index=False)
    
    # 保存训练集
    train_data = data.iloc[train_indices][['SMILES', 'MP']]
    train_file = os.path.join(output_dir, 'multimodal_train.csv')
    train_data.to_csv(train_file, index=False)
    
    # 统计分布信息，用于直观检验
    print(f"\n======== 分布一致性统计 ========")
    print(f" [测试集]")
    print(f"   分子数: {len(test_data)} 条")
    print(f"   MP 均值: {test_data['MP'].mean():.2f} °C")
    print(f"   MP 中位数: {test_data['MP'].median():.2f} °C")
    print(f"   MP 标准差: {test_data['MP'].std():.2f} °C")
    print(f"   MP 范围: {test_data['MP'].min():.1f} ~ {test_data['MP'].max():.1f} °C")
    
    print(f"\n [训练集]")
    print(f"   分子数: {len(train_data)} 条")
    print(f"   MP 均值: {train_data['MP'].mean():.2f} °C")
    print(f"   MP 中位数: {train_data['MP'].median():.2f} °C")
    print(f"   MP 标准差: {train_data['MP'].std():.2f} °C")
    print(f"   MP 范围: {train_data['MP'].min():.1f} ~ {train_data['MP'].max():.1f} °C")
    print(f"================================\n")
    
    print(f"多模态测试集已保存: {test_file}")
    print(f"多模态训练集已保存: {train_file}")
    
    return test_file, train_file

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(current_dir, "cleaned.csv")
    output_dir = current_dir
    
    # 分割数据：目标4000，利用10个分位数区间控制分布
    test_file, train_file = scaffold_split_for_multimodal(
        input_file=input_file,
        output_dir=output_dir,
        test_size=4000,
        num_bins=10, 
        seed=42
    )
    
    print("\n运行完成!")