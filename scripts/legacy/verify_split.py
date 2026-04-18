import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import RDLogger
from collections import defaultdict

# 完全屏蔽RDKit日志
RDLogger.DisableLog('rdApp.*')

def get_scaffold(smiles, include_chirality=False):
    """获取分子的骨架（Murcko骨架）- 与split_multimodal_fixed.py完全一致"""
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

def verify_split():
    """使用与split_multimodal_fixed.py相同的逻辑验证分割结果"""
    print("=" * 60)
    print("使用 split_multimodal_fixed.py 相同逻辑验证数据")
    print("=" * 60)

    # 读取数据
    train_file = "multimodal_train.csv"
    test_file = "multimodal_test.csv"

    print(f"\n读取训练集: {train_file}")
    train_data = pd.read_csv(train_file)
    print(f"读取测试集: {test_file}")
    test_data = pd.read_csv(test_file)

    print(f"\n训练集: {len(train_data)} 条")
    print(f"测试集: {len(test_data)} 条")

    # 提取所有SMILES
    train_smiles_list = train_data['SMILES'].tolist()
    test_smiles_list = test_data['SMILES'].tolist()

    # ========== 方法1: 与split_multimodal_fixed.py完全相同的验证逻辑 ==========
    print("\n" + "-" * 60)
    print("[验证1] 使用split_multimodal_fixed.py相同逻辑验证骨架互斥性")
    print("-" * 60)

    # 构建scaffold_to_indices映射（模拟分割时的结构）
    train_scaffold_to_indices = defaultdict(list)
    test_scaffold_to_indices = defaultdict(list)

    print("  提取训练集骨架...")
    for i, smi in enumerate(train_smiles_list):
        scaffold = get_scaffold(smi)
        if scaffold is not None:
            train_scaffold_to_indices[scaffold].append(i)

    print("  提取测试集骨架...")
    for i, smi in enumerate(test_smiles_list):
        scaffold = get_scaffold(smi)
        if scaffold is not None:
            test_scaffold_to_indices[scaffold].append(i)

    # 验证：确保没有骨架重叠（与split_multimodal_fixed.py第140-157行完全一致）
    print("\n  验证骨架分割互斥性...")
    test_scaffolds = set()
    train_scaffolds = set()

    # 获取测试集骨架
    for scaffold, indices in test_scaffold_to_indices.items():
        test_scaffolds.add(scaffold)

    # 获取训练集骨架
    for scaffold, indices in train_scaffold_to_indices.items():
        train_scaffolds.add(scaffold)

    overlap = test_scaffolds & train_scaffolds
    if len(overlap) > 0:
        print(f"  ✗ 错误: 发现 {len(overlap)} 个骨架同时存在于train和test中!")

        # 统计泄露的分子
        leaked_train = sum(len(train_scaffold_to_indices[s]) for s in overlap)
        leaked_test = sum(len(test_scaffold_to_indices[s]) for s in overlap)

        print(f"    - 涉及训练集分子: {leaked_train} 个")
        print(f"    - 涉及测试集分子: {leaked_test} 个")

        print(f"\n    示例泄露骨架:")
        for i, s in enumerate(list(overlap)[:5]):
            print(f"      {i+1}. {s}")
            print(f"         训练集: {len(train_scaffold_to_indices[s])} 个分子")
            print(f"         测试集: {len(test_scaffold_to_indices[s])} 个分子")
    else:
        print(f"  ✓ 验证通过: train和test骨架完全互斥，0重叠")

    # ========== 方法2: 检查完全重复的SMILES ==========
    print("\n" + "-" * 60)
    print("[验证2] 检查完全重复的SMILES")
    print("-" * 60)

    # 标准化SMILES
    def canonicalize(smi):
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                return None
            return Chem.MolToSmiles(mol, isomericSmiles=False)
        except:
            return None

    train_canonical = [canonicalize(s) for s in train_smiles_list]
    test_canonical = [canonicalize(s) for s in test_smiles_list]

    train_set = set(s for s in train_canonical if s is not None)
    test_set = set(s for s in test_canonical if s is not None)

    duplicate = train_set & test_set
    if len(duplicate) > 0:
        print(f"  ✗ 发现 {len(duplicate)} 个完全重复的SMILES!")
        print(f"\n    示例重复SMILES:")
        for i, s in enumerate(list(duplicate)[:5]):
            print(f"      {i+1}. {s}")
    else:
        print(f"  ✓ 没有发现完全重复的SMILES")

    # ========== 统计信息 ==========
    print("\n" + "-" * 60)
    print("[统计信息]")
    print("-" * 60)
    print(f"  训练集唯一骨架数: {len(train_scaffolds)}")
    print(f"  测试集唯一骨架数: {len(test_scaffolds)}")
    print(f"  训练集平均每个骨架分子数: {len(train_data)/len(train_scaffolds):.2f}")
    print(f"  测试集平均每个骨架分子数: {len(test_data)/len(test_scaffolds):.2f}")

    print("\n" + "=" * 60)
    if len(overlap) == 0 and len(duplicate) == 0:
        print("结论: ✓ 数据分割正常，无泄露问题")
    else:
        print("结论: ✗ 发现数据泄露问题!")
    print("=" * 60)

if __name__ == "__main__":
    verify_split()
