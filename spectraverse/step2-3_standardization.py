import numpy as np
import os, glob, re
import pandas as pd
import sys
from rdkit import Chem

STANDARDIZE_TAUTOMERS = os.environ.get("SPECTRAVERSE_STANDARDIZE_TAUTOMERS", "true").strip().lower() == "true"
print(f"[step2-3 config] standardize_tautomers={STANDARDIZE_TAUTOMERS} (env SPECTRAVERSE_STANDARDIZE_TAUTOMERS)")
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem import rdMolDescriptors
from tqdm import tqdm
from multiprocessing import Pool

print("Step2-3: Standardize SMILES")

metadata_csv_dir = sys.argv[1]
input_mgf_dir = sys.argv[2]

benchmark_dir = os.path.dirname(metadata_csv_dir)

temp_mgf_dir = benchmark_dir + '/test-temp.mgf'
temp_csv_dir = benchmark_dir + '/test-temp.csv'

output_csv_dir = sys.argv[3]
output_mgf_dir = sys.argv[4]


removed_index = []

metadata = pd.read_csv(metadata_csv_dir)

deuterated = metadata[metadata['SMILES'].fillna('').str.contains('[2H]', regex=False)].index
metadata = metadata[~metadata['SMILES'].fillna('').str.contains('[2H]', regex=False)]

tritiated = metadata[metadata['SMILES'].fillna('').str.contains('[3H]', regex=False)].index
metadata = metadata[~metadata['SMILES'].fillna('').str.contains('[3H]', regex=False)]

c13_smiles = metadata[metadata['SMILES'].fillna('').str.contains('[13C]', regex=False)].index
metadata = metadata[~metadata['SMILES'].fillna('').str.contains('[13C]', regex=False)]

precursor_mz_1000 = metadata[metadata['PRECURSOR_MZ'] > 1000]
metadata = metadata[metadata['PRECURSOR_MZ'] <= 1000]

metadata_columns = list(metadata.columns)

smiles_na_cleaned_file = os.environ.get("SPECTRAVERSE_SMILES_REPAIR_CSV", "")
if smiles_na_cleaned_file and os.path.exists(smiles_na_cleaned_file):
    ref = pd.read_csv(smiles_na_cleaned_file)
    smiles_na_unique = ref['COMPOUND_NAME'].unique()
    for i, compound in enumerate(smiles_na_unique):
        metadata.loc[(metadata['SMILES'].isna()) & (metadata['COMPOUND_NAME'] == compound), 'SMILES'] = ref['SMILES'][i]
else:
    print(f"Skipping SMILES-NA repair (SPECTRAVERSE_SMILES_REPAIR_CSV not set or file missing)")

metadata = metadata[metadata['SMILES'].notna()]
metadata_index = metadata.index

def extract_info(filename, indices):
    info_list = []
    record = False
    with open(filename, 'r') as file:
        current_info = []
        for line in file:
            line = line.strip()
            if 'BEGIN IONS' in line:
                record = True
                current_info = [line]
            elif 'END IONS' in line:
                current_info.append(line)
                record = False
                info_list.append(current_info)
            elif record:
                current_info.append(line)
    return [info_list[i] for i in indices]

info = extract_info(input_mgf_dir, metadata_index)

with open(temp_mgf_dir, 'w') as file:
    for sublist in info:
        for item in sublist:
            file.write("%s\n" % item)
        file.write("\n") 

metadata.to_csv(temp_csv_dir, index=False)

# Molecule standardization is shared from a single reusable module
# (spectraverse/standardization.py) so the exact same chemistry is used by the
# curation pipeline and by DreaMS-Mol's standardize_smiles.
try:
    from spectraverse.standardization import preprocess_mol
except ImportError:  # run as a script: standardization.py is a sibling on sys.path
    from standardization import preprocess_mol

def process_smiles(smiles):
    try:
        smiles = smiles.split(' |')[0]
        mol, canonical_sm, inchikey, mol_charge, canonical_sm_charge, inchikey_charge, duplicate = preprocess_mol(smiles, standardize_tautomers=STANDARDIZE_TAUTOMERS)
        return mol, canonical_sm, inchikey, mol_charge, canonical_sm_charge, inchikey_charge, duplicate
    except ValueError as e:
        return None, None, None, None, None, None, None

metadata = pd.read_csv(temp_csv_dir)

uniq_smiles = metadata.drop_duplicates('SMILES')

with Pool() as p:
    results = p.map(process_smiles, uniq_smiles['SMILES'].values)

mols, canonical_smiles, inchikeys, charge_cols, charge_canonical_smiles, charge_inchikey, duplicate = zip(*results)
# add to data frame
uniq_smiles = uniq_smiles.assign(CANONICAL_SMILES=canonical_smiles, CANONICAL_INCHIKEY=inchikeys,
                                 CANONICAL_SMILES_CHARGE=charge_canonical_smiles, CANONICAL_INCHIKEY_CHARGE=charge_inchikey,
                                 DUPLICATE=duplicate)
uniq_smiles = uniq_smiles[['SMILES', 'CANONICAL_SMILES', 'CANONICAL_INCHIKEY', 'CANONICAL_SMILES_CHARGE', 'CANONICAL_INCHIKEY_CHARGE', 'DUPLICATE']]
metadata = metadata.merge(uniq_smiles, how='left', on='SMILES')

metadata1 = metadata[metadata_columns + ['CANONICAL_SMILES', 'CANONICAL_INCHIKEY', 'DUPLICATE']]
metadata2 = metadata[metadata_columns + ['CANONICAL_SMILES_CHARGE', 'CANONICAL_INCHIKEY_CHARGE', 'DUPLICATE']]

metadata2 = metadata2.rename(columns={'CANONICAL_SMILES_CHARGE': 'CANONICAL_SMILES', 'CANONICAL_INCHIKEY_CHARGE': 'CANONICAL_INCHIKEY'})

metadata1 = metadata1[~metadata1['CANONICAL_SMILES'].isna()]
metadata2 = metadata2[~metadata2['CANONICAL_SMILES'].isna()]

metadata = pd.concat([metadata1, metadata2])
metadata = metadata.sort_index()
metadata_index = metadata.index

def calculate_exact_mass(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        exact_mass = Descriptors.ExactMolWt(mol)
    except:
        exact_mass = None    

    return exact_mass

metadata['EXACT_MASS_ORIGINAL'] = metadata['SMILES'].apply(calculate_exact_mass)
metadata['EXACT_MASS_CANONICAL'] = metadata['CANONICAL_SMILES'].apply(calculate_exact_mass)

for i in range(metadata.shape[0]):
    smiles = metadata['CANONICAL_SMILES'].iloc[i]
    mol = Chem.MolFromSmiles(smiles)
    charge = Chem.GetFormalCharge(mol)
    if charge != 0:
        parent_mass = metadata['PARENT_MASS'].iloc[i]
        exact_mass = metadata['EXACT_MASS_CANONICAL'].iloc[i]
        if metadata['ADDUCT'].iloc[i] == '[M+H]+':
            if abs(parent_mass - exact_mass) <= 0.1 and charge == 1:
                metadata['ADDUCT'].iloc[i] = '[M]+'
        if metadata['ADDUCT'].iloc[i] == '[M-H]-':
            if abs(parent_mass - exact_mass) <= 0.1 and charge == -1:
                metadata['ADDUCT'].iloc[i] = '[M]-'

# ref: https://github.com/matchms/matchms/blob/1f904e0d469aef35dbba8b7b2d7b52886f3f75cc/matchms/data/known_adducts_table.csv#L56
adduct_mass_adjustments = {
    '[M]+': 0,
    '[M+K]+': 38.963158,
    '[M+CH3COO]-': 59.013851,
    '[M+H+HCOOH]+': 46.00548,
    '[M]-': 0,
    '[M+FA-H]-': (46.005477 - 1.007276),
    '[M+Cl]-' : 34.969402,
    '[M+NH4]+': 18.033823,
    '[M+Na]+' : 22.989218,
    '[M-H]-': -1.007276,
    '[M+H]+': 1.007276,
    '[M]': 0,
    '[M+K]': 38.963158,
    '[M+CH3COO]': 59.013851,
    '[M+H+HCOOH]': 46.00548,
    '[M+FA-H]': (46.005477 - 1.007276),
    '[M+Cl]' : 34.969402,
    '[M+NH4]': 18.033823,
    '[M+Na]' : 22.989218,
    '[M-H]': -1.007276,
    '[M+H]': 1.007276,
}

def normalize_adduct(adduct):
    if pd.isna(adduct):
        return adduct    
    return adduct.rstrip('+-')

if 'ORIG_ADDUCT' in metadata.columns:
    for i in range(metadata.shape[0]):
        orig_adduct = metadata['ORIG_ADDUCT'].iloc[i]
        adduct = metadata['ADDUCT'].iloc[i]
        if normalize_adduct(orig_adduct) != normalize_adduct(adduct) and orig_adduct in adduct_mass_adjustments and adduct in adduct_mass_adjustments:
            parent_mass = metadata['EXACT_MASS_CANONICAL'].iloc[i]
            precusor_mz = metadata['PRECURSOR_MZ'].iloc[i]

            mass_diff = precusor_mz - parent_mass

            orig_adduct_value = adduct_mass_adjustments.get(orig_adduct)
            adduct_value = adduct_mass_adjustments.get(adduct)

            orig_adduct_diff = abs(mass_diff - orig_adduct_value)
            adduct_diff = abs(mass_diff - adduct_value)

            if orig_adduct_diff < adduct_diff:
                metadata['ADDUCT'].iloc[i] = orig_adduct
else:
    print("Skipping ORIG_ADDUCT-based correction (column absent)")

info = extract_info(temp_mgf_dir, metadata_index)

with open(output_mgf_dir, 'w') as file:
    for sublist in info:
        for item in sublist:
            file.write("%s\n" % item)
        file.write("\n") 

metadata.to_csv(output_csv_dir, index=False)

