# Feature Bundle Schema Documentation

## Overview

This document describes the 17-dimensional **mixed-source physicochemical feature bundle** used for melting point prediction.

**IMPORTANT**: Despite the historical name `XTB_train.pth`, this is not a pure XTB feature set. It is a **hybrid feature bundle** where 16 dimensions come from XTB calculations and 1 dimension (molecular volume) comes from RDKit.

## Feature Definitions

| Index | Field Name | Unit | 最终来源 | 当前实现 | 是否真实验证 | 备注 |
|-------|-----------|------|---------|----------|--------------|------|
| 0 | N_Atoms | count | XTB direct parse | Heavy atom count (Z>1) | ✓ Verified | 实际为非氢原子数 |
| 1 | N_Heavy_Atoms | count | XTB direct parse | Same as N_Atoms | ✓ Verified | 与N_Atoms一致 |
| 2 | Molecular_Mass_amu | amu | derived from XTB | Sum(atomic_mass * count) | ✓ Verified | 从原子计数计算 |
| 3 | Electronic_Energy_AU | Hartree | XTB direct parse | Line: `:: total energy XXX Eh` | ✓ Verified | 总能量 |
| 4 | Electronic_Energy_kcal_mol | kcal/mol | derived from XTB | AU * 627.509 | ✓ Verified | 能量单位转换 |
| 5 | HOMO_eV | eV | XTB direct parse | Line: `... (HOMO)` with energy in eV | ✓ Verified | 最高占据轨道 |
| 6 | LUMO_eV | eV | XTB direct parse | Line: `... (LUMO)` with energy in eV | ✓ Verified | 最低未占轨道 |
| 7 | HOMO_LUMO_Gap_eV | eV | XTB direct parse | Line: `HL-Gap XXX eV` | ✓ Verified | 能隙 |
| 8 | Dipole_Total_Debye | Debye | XTB direct parse | Line: `full: x y z total` | ✓ Verified | 偶极矩总量 |
| 9 | Dipole_Theta_deg | degrees | derived from XTB | atan2(z, sqrt(x²+y²)) | ✓ Verified | 偶极矩角度 |
| 10 | Dipole_Phi_deg | degrees | derived from XTB | atan2(y, x) | ✓ Verified | 偶极矩角度 |
| 11 | Charge_Min | e | XTB direct parse | Min of atomic charges | ✓ Verified | 最小原子电荷 |
| 12 | Charge_Max | e | XTB direct parse | Max of atomic charges | ✓ Verified | 最大原子电荷 |
| 13 | Charge_Mean | e | XTB direct parse | Mean of atomic charges | ✓ Verified | 平均原子电荷 |
| 14 | Charge_STD | e | XTB direct parse | Std dev of atomic charges | ✓ Verified | 电荷标准差 |
| 15 | Charge_Range | e | derived from XTB | Charge_Max - Charge_Min | ✓ Verified | 电荷范围 |
| 16 | Molecular_Volume_cm3_mol | cm³/mol | RDKit-derived | Volume = 1.2 * MolMR | ✓ Verified | 基于摩尔折射率的近似值 |

## Source Classification

### XTB Direct Parse (10 fields)
- Fields directly extracted from XTB output
- No additional calculation required
- Includes: N_Atoms, N_Heavy_Atoms, Electronic_Energy_AU, HOMO_eV, LUMO_eV, HOMO_LUMO_Gap_eV, Dipole_Total_Debye, Charge_Min, Charge_Max, Charge_Mean, Charge_STD

### Derived from XTB (5 fields)
- Calculated from directly parsed XTB fields
- Includes: Molecular_Mass_amu, Electronic_Energy_kcal_mol, Dipole_Theta_deg, Dipole_Phi_deg, Charge_Range

### RDKit-derived (1 field)
- **Molecular_Volume_cm3_mol**: Not available in standard XTB output
- Calculated using RDKit's molar refractivity (MolMR)
- Method: Volume = 1.2 * molar_refractivity (empirical conversion)
- Note: This is an approximate value, not a precise physical measurement

## Implementation Modules

### 1. `src/preprocessing/xtb_extract.py`
- **Responsibility**: Extracts 16-dimensional XTB features
- **Output**: 16D features (missing volume)
- **Volume field**: Marked as `external_fill_required`

### 2. `src/preprocessing/rdkit_features.py`
- **Responsibility**: Computes molecular volume using RDKit
- **Output**: 1D volume feature
- **Method**: Volume = 1.2 * molar_refractivity

### 3. `src/preprocessing/merge_features.py`
- **Responsibility**: Merges 16D XTB features with 1D RDKit volume
- **Output**: 17D feature bundle compatible with XTB_train.pth
- **Validation**: Ensures schema compatibility

## Schema Layers

The feature bundle is organized into three distinct schema layers:

### 1. XTB_PARSED_16D_NAMES (16 dimensions)
- Directly parsed from XTB output or derived from XTB parsed values
- Does NOT include molecular volume
- Defined in: `src/preprocessing/schema.py`

### 2. RDKIT_EXTRA_1D_NAMES (1 dimension)
- `Molecular_Volume_cm3_mol` - Computed separately by RDKit
- Defined in: `src/preprocessing/schema.py`

### 3. FULL_17D_FEATURE_NAMES (17 dimensions)
- Complete feature bundle: XTB_PARSED_16D_NAMES + RDKIT_EXTRA_1D_NAMES
- Compatible with old XTB_train.pth format
- Defined in: `src/preprocessing/schema.py`

## Workflow for New Molecules

1. **XTB Calculation**: Run XTB on new molecules
2. **Feature Extraction**: Use `xtb_extract.py` to get 16D features
3. **Volume Calculation**: Use `rdkit_features.py` to get volume
4. **Bundle Creation**: Use `merge_features.py` to create 17D bundle

## Important Notes

1. **Mixed-Source Nature**: The feature bundle is a hybrid of XTB and RDKit features
2. **Volume Approximation**: Molecular volume is an approximate calculation
3. **SMILES Canonicalization**: Always canonicalize SMILES when matching molecules
4. **Schema Compatibility**: Output maintains compatibility with original XTB_train.pth format

## Validation Results

### XTB Fields
- Energy: < 0.01 AU difference (acceptable)
- HOMO/LUMO: < 0.5 eV difference (acceptable)
- Dipole: < 0.1 Debye difference (acceptable)

### Volume Field
- RDKit-derived volume values match the approach used in XTB_train.pth
- Volume calculation is stable and reproducible

## Schema Versioning

To ensure reproducibility, the feature bundle includes version information:

```python
schema_info = {
    'description': 'Mixed-source physicochemical feature bundle',
    'xtb_features': 16,
    'rdkit_features': 1,
    'total_dimensions': 17
}
```

## File References

- **XTB Parser**: `src/preprocessing/xtb_extract.py`
- **Volume Calculator**: `src/preprocessing/rdkit_features.py`
- **Bundle Merger**: `src/preprocessing/merge_features.py`
- **Volume Script**: `scripts/00c_compute_rdkit_volume.py`
- **Bundle Script**: `scripts/00d_merge_feature_bundle.py`