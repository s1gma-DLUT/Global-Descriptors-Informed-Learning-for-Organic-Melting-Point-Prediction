import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import RDLogger
from collections import defaultdict

# 完全屏蔽RDKit日志
RDLogger.DisableLog('rdApp.*')

def get_scaffold(smiles, include_chirality=False):
    """获取分子的骨架（Murcko骨架）"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        if scaffold is None:
            return smiles
        scaffold_smiles = Chem.MolToSmiles(scaffold, isomericSmiles=include_chirality)
        if not scaffold_smiles:
            return smiles
        return scaffold_smiles
    except Exception:
        return smiles

def canonicalize_smiles(smiles):
    """标准化SMILES"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, isomericSmiles=False)
    except Exception:
        return None

def check_data_leakage(train_file, test_file):
    """检查数据泄露问题"""
    print("=" * 60)
    print("数据泄露检查报告")
    print("=" * 60)

    # 读取数据
    print("\n[1] 读取数据...")
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)

    print(f"  训练集: {len(train_df)} 条记录")
    print(f"  测试集: {len(test_df)} 条记录")

    # ========== 检查1: 完全重复的SMILES ==========
    print("\n[2] 检查完全重复的SMILES...")

    # 标准化SMILES
    train_df['canonical_smiles'] = train_df['SMILES'].apply(canonicalize_smiles)
    test_df['canonical_smiles'] = test_df['SMILES'].apply(canonicalize_smiles)

    # 移除无法解析的SMILES
    train_valid = train_df[train_df['canonical_smiles'].notna()]
    test_valid = test_df[test_df['canonical_smiles'].notna()]

    train_smiles_set = set(train_valid['canonical_smiles'])
    test_smiles_set = set(test_valid['canonical_smiles'])

    duplicate_smiles = train_smiles_set & test_smiles_set

    if len(duplicate_smiles) > 0:
        print(f"  ✗ 发现 {len(duplicate_smiles)} 个完全重复的SMILES!")
        print(f"    示例重复SMILES:")
        for i, smi in enumerate(list(duplicate_smiles)[:5]):
            print(f"      {i+1}. {smi}")
    else:
        print(f"  ✓ 没有发现完全重复的SMILES")

    # ========== 检查2: 骨架泄露 ==========
    print("\n[3] 检查骨架泄露...")

    print("  正在提取训练集骨架...")
    train_scaffolds = {}
    train_scaffold_counts = defaultdict(int)
    for idx, row in train_valid.iterrows():
        scaffold = get_scaffold(row['canonical_smiles'])
        if scaffold:
            train_scaffolds[row['canonical_smiles']] = scaffold
            train_scaffold_counts[scaffold] += 1

    print("  正在提取测试集骨架...")
    test_scaffolds = {}
    test_scaffold_counts = defaultdict(int)
    for idx, row in test_valid.iterrows():
        scaffold = get_scaffold(row['canonical_smiles'])
        if scaffold:
            test_scaffolds[row['canonical_smiles']] = scaffold
            test_scaffold_counts[scaffold] += 1

    # 检查骨架重叠
    train_scaffold_set = set(train_scaffold_counts.keys())
    test_scaffold_set = set(test_scaffold_counts.keys())
    overlapping_scaffolds = train_scaffold_set & test_scaffold_set

    if len(overlapping_scaffolds) > 0:
        print(f"  ✗ 发现 {len(overlapping_scaffolds)} 个骨架同时存在于训练集和测试集!")

        # 统计泄露的分子数量
        leaked_train_mols = sum(train_scaffold_counts[s] for s in overlapping_scaffolds)
        leaked_test_mols = sum(test_scaffold_counts[s] for s in overlapping_scaffolds)

        print(f"    - 涉及训练集分子数: {leaked_train_mols}")
        print(f"    - 涉及测试集分子数: {leaked_test_mols}")
        print(f"    - 训练集泄露比例: {leaked_train_mols/len(train_valid)*100:.2f}%")
        print(f"    - 测试集泄露比例: {leaked_test_mols/len(test_valid)*100:.2f}%")

        print(f"\n    示例泄露骨架:")
        for i, scaffold in enumerate(list(overlapping_scaffolds)[:5]):
            train_count = train_scaffold_counts[scaffold]
            test_count = test_scaffold_counts[scaffold]
            print(f"      {i+1}. {scaffold}")
            print(f"         训练集出现: {train_count} 次, 测试集出现: {test_count} 次")
    else:
        print(f"  ✓ 没有发现骨架泄露，训练集和测试集骨架完全互斥")

    # ========== 统计信息 ==========
    print("\n[4] 骨架统计信息...")
    print(f"  训练集唯一骨架数: {len(train_scaffold_set)}")
    print(f"  测试集唯一骨架数: {len(test_scaffold_set)}")
    print(f"  训练集平均每个骨架的分子数: {len(train_valid)/len(train_scaffold_set):.2f}")
    print(f"  测试集平均每个骨架的分子数: {len(test_valid)/len(test_scaffold_set):.2f}")

    # 统计单分子骨架（singleton）
    train_singletons = sum(1 for v in train_scaffold_counts.values() if v == 1)
    test_singletons = sum(1 for v in test_scaffold_counts.values() if v == 1)
    print(f"  训练集单分子骨架数: {train_singletons} ({train_singletons/len(train_scaffold_set)*100:.1f}%)")
    print(f"  测试集单分子骨架数: {test_singletons} ({test_singletons/len(test_scaffold_set)*100:.1f}%)")

    print("\n" + "=" * 60)
    if len(duplicate_smiles) == 0 and len(overlapping_scaffolds) == 0:
        print("结论: ✓ 数据分割正常，无泄露问题")
    else:
        print("结论: ✗ 发现数据泄露问题，需要重新分割数据!")
    print("=" * 60)

if __name__ == "__main__":
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    train_file = os.path.join(current_dir, "multimodal_train.csv")
    test_file = os.path.join(current_dir, "multimodal_test.csv")

    check_data_leakage(train_file, test_file)
