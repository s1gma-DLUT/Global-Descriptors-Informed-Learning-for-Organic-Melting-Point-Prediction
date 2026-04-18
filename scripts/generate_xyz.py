#!/usr/bin/env python3
"""
Generate 3D XYZ files from SMILES for XTB calculations.
"""

import os
from rdkit import Chem
from rdkit.Chem import AllChem

def generate_xyz(smiles, output_dir, mol_id):
    """Generate XYZ file from SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    # Add hydrogens
    mol = Chem.AddHs(mol)
    
    # Generate 3D coordinates
    AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    AllChem.UFFOptimizeMolecule(mol)
    
    # Write XYZ file
    output_file = os.path.join(output_dir, f"mol_{mol_id:06d}_{smiles.replace('/', '_')[:50]}.xyz")
    
    conf = mol.GetConformer()
    with open(output_file, 'w') as f:
        f.write(f"{mol.GetNumAtoms()}\n")
        f.write(f"Generated from SMILES: {smiles}\n")
        for atom in mol.GetAtoms():
            pos = conf.GetAtomPosition(atom.GetIdx())
            f.write(f"{atom.GetSymbol():<2} {pos.x:10.6f} {pos.y:10.6f} {pos.z:10.6f}\n")
    
    return output_file

def main():
    output_dir = "data/external/xtb_jobs/smoke_test/inputs"
    os.makedirs(output_dir, exist_ok=True)
    
    smiles_list = ["CCC", "c1ccccc1", "CCCC"]
    
    for i, smiles in enumerate(smiles_list):
        xyz_file = generate_xyz(smiles, output_dir, i)
        if xyz_file:
            print(f"Generated: {xyz_file}")
        else:
            print(f"Failed: {smiles}")

if __name__ == "__main__":
    main()
