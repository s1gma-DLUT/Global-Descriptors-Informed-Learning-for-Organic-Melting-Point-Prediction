# XTB Feature Schema

The files named `XTB_train.pth` and `XTB_test.pth` contain a 17-dimensional
feature bundle:

- 16 XTB parsed or derived features.
- 1 RDKit-derived molecular volume feature.

## Feature Order

1. `N_Atoms`
2. `N_Heavy_Atoms`
3. `Molecular_Mass_amu`
4. `Electronic_Energy_AU`
5. `Electronic_Energy_kcal_mol`
6. `HOMO_eV`
7. `LUMO_eV`
8. `HOMO_LUMO_Gap_eV`
9. `Dipole_Total_Debye`
10. `Dipole_Theta_deg`
11. `Dipole_Phi_deg`
12. `Charge_Min`
13. `Charge_Max`
14. `Charge_Mean`
15. `Charge_STD`
16. `Charge_Range`
17. `Molecular_Volume_cm3_mol`

The schema constants are defined in `src/preprocessing/schema.py`.

## Notes

The feature bundle is intentionally stored as a fixed-order numeric vector for
training speed. When regenerating features, validate both the shape and feature
order before starting a training run.

`Molecular_Volume_cm3_mol` is computed by RDKit and appended to the 16 parsed or
derived XTB fields to form the 17-dimensional bundle.
