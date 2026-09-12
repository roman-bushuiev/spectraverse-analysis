"""Molecule standardization chemistry for the spectraverse curation pipeline.

Extracted verbatim from ``step2-3_standardization.py`` so that a single, validated
implementation can be reused both by that step and by external consumers
(DreaMS-Mol's ``standardize_smiles``). rdkit-only and side-effect-free at import
(no env reads, no pandas/tqdm), so importing it stays cheap.

``preprocess_mol`` runs at spectraverse defaults: tautomer standardization ON,
charge neutralization ON, stereochemistry stripped (``stereochem=False``).
"""
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.MolStandardize import rdMolStandardize

# spectraverse default when a caller does not specify (env-free; step2-3 passes its
# own SPECTRAVERSE_STANDARDIZE_TAUTOMERS explicitly to preserve its behaviour).
DEFAULT_STANDARDIZE_TAUTOMERS = True

uncharger = rdMolStandardize.Uncharger()
te = rdMolStandardize.TautomerEnumerator()

# define a function to check for aromatic sulfoxides and correct bond/charges
def check_aromatic_sulfoxides(parent_clean_mol):
    S_pos = []
    O_pos = []
    S_charge = False
    O_charge = False
    single_bond = False
    for atom in parent_clean_mol.GetAtoms():
        if atom.GetSymbol() == 'S':
            charge = atom.GetFormalCharge()
            if charge == 1:
                S_charge = True
                S_pos.append(atom.GetIdx())
        if atom.GetSymbol() == 'O':
            charge = atom.GetFormalCharge()
            degree = atom.GetDegree()
            if charge == -1 and degree == 1:
                O_charge = True    
                O_pos.append(atom.GetIdx())
    bonds = []
    SO_comb = []
    SO_comb_idx = []
    for i in range(len(S_pos)):
        for j in range(len(O_pos)):
            bonds.append(parent_clean_mol.GetBondBetweenAtoms(S_pos[i], O_pos[j]))
            SO_comb.append((S_pos[i], O_pos[j]))
    none_indices = [i for i, x in enumerate(bonds) if x is None]
    bonds = [x for x in bonds if x is not None]
    SO_comb = [x for i, x in enumerate(SO_comb) if i not in none_indices]        
    for i in range(len(bonds)):
        if bonds[i].GetBondType() == Chem.rdchem.BondType.SINGLE:
            single_bond = True
            SO_comb_idx.append(i)
    
    if S_charge and O_charge and single_bond:
        for i in range(len(SO_comb_idx)):
            atom1 = parent_clean_mol.GetAtomWithIdx(SO_comb[SO_comb_idx[i]][0])
            atom1.SetFormalCharge(0)
            atom2 = parent_clean_mol.GetAtomWithIdx(SO_comb[SO_comb_idx[i]][1])
            atom2.SetFormalCharge(0)
            bond = parent_clean_mol.GetBondBetweenAtoms(SO_comb[SO_comb_idx[i]][0], SO_comb[SO_comb_idx[i]][1])
            bond.SetBondType(Chem.rdchem.BondType.DOUBLE)
    
    return parent_clean_mol

def check_P_O_charge(parent_clean_mol):
    P_pos = []
    O_pos = []
    P_charge = False
    O_charge = False
    single_bond = False
    for atom in parent_clean_mol.GetAtoms():
        if atom.GetSymbol() == 'P':
            charge = atom.GetFormalCharge()
            if charge == 1:
                P_charge = True
                P_pos.append(atom.GetIdx())
        if atom.GetSymbol() == 'O':
            charge = atom.GetFormalCharge()
            degree = atom.GetDegree()
            if charge == -1 and degree == 1:
                O_charge = True    
                O_pos.append(atom.GetIdx())
    bonds = []
    PO_comb = []
    PO_comb_idx = []
    for i in range(len(P_pos)):
        for j in range(len(O_pos)):
            bonds.append(parent_clean_mol.GetBondBetweenAtoms(P_pos[i], O_pos[j]))
            PO_comb.append((P_pos[i], O_pos[j]))
    none_indices = [i for i, x in enumerate(bonds) if x is None]
    bonds = [x for x in bonds if x is not None]
    PO_comb = [x for i, x in enumerate(PO_comb) if i not in none_indices]
    for i in range(len(bonds)):
        if bonds[i].GetBondType() == Chem.rdchem.BondType.SINGLE:
            single_bond = True
            PO_comb_idx.append(i)

    if P_charge and O_charge and single_bond:
        for i in range(len(PO_comb_idx)):
            atom1 = parent_clean_mol.GetAtomWithIdx(PO_comb[PO_comb_idx[i]][0])
            atom1.SetFormalCharge(0)
            atom2 = parent_clean_mol.GetAtomWithIdx(PO_comb[PO_comb_idx[i]][1])
            atom2.SetFormalCharge(0)
            bond = parent_clean_mol.GetBondBetweenAtoms(PO_comb[PO_comb_idx[i]][0], PO_comb[PO_comb_idx[i]][1])
            bond.SetBondType(Chem.rdchem.BondType.DOUBLE)

    return parent_clean_mol

"""
rdkit contributed code to neutralize charged molecules;
obtained from:
    https://www.rdkit.org/docs/Cookbook.html
    http://www.mail-archive.com/rdkit-discuss@lists.sourceforge.net/msg02669.html
"""
def _InitialiseNeutralisationReactions():
    patts= (
        # Imidazoles
        ('[n+;H]','n'),
        # Amines
        ('[N+;!H0]','N'),
        # Carboxylic acids and alcohols
        ('[$([O-]);!$([O-][#7])]','O'),
        # Thiols
        ('[S-;X1]','S'),
        # Sulfonamides
        ('[$([N-;X2]S(=O)=O)]','N'),
        # Enamines
        ('[$([N-;X2][C,N]=C)]','N'),
        # Tetrazoles
        ('[n-]','[nH]'),
        # Sulfoxides
        ('[$([S-]=O)]','S'),
        # Amides
        ('[$([N-]C=O)]','N'),
        )
    return [(Chem.MolFromSmarts(x),Chem.MolFromSmiles(y,False)) for x,y in patts]

_reactions=None
def NeutraliseCharges(mol, reactions=None):
    global _reactions
    if reactions is None:
        if _reactions is None:
            _reactions=_InitialiseNeutralisationReactions()
        reactions=_reactions
    for i,(reactant, product) in enumerate(reactions):
        while mol.HasSubstructMatch(reactant):
            rms = AllChem.ReplaceSubstructs(mol, reactant, product)
            mol = rms[0]
    return mol

def preprocess_mol(smiles, stereochem=False, standardize_tautomers=None):
    if standardize_tautomers is None:
        standardize_tautomers = DEFAULT_STANDARDIZE_TAUTOMERS
    """
    SMILES preprocessing and standardization pipeline:
    1. Load molecule in RDKit
    2. Optionally remove stereochemistry
    3. Sanitize and remove hydrogens
    4. Run 'Cleanup' (disconnect metal atoms, reionize)
    5. Select parent molecule if multiple fragments
    6. Neutralize charges, using two different approaches
    7. Optionally, standardize tautomers
    8. Return the cleaned molecule, canonical SMILES, and inchikey.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        # raise ValueError("invalid SMILES: " + str(smiles))
        print("invalid SMILES: " + str(smiles))
        return None, None, None, None, None, None, None, None 
    
    # optionally, remove stereochemistry
    if not stereochem:
        Chem.RemoveStereochemistry(mol)
    
    # sanitize molecule and remove hydrogens
    try:
    # sanitize molecule and remove hydrogens
        Chem.SanitizeMol(mol)
    except:
        print("Unable to kekulize molecule: " + str(smiles))
        return None, None, None, None, None, None, None, None 
    mol = Chem.RemoveHs(mol)
    
    # from https://bitsilla.com/blog/2021/06/standardizing-a-molecule-using-rdkit/
    # removeHs, disconnect metal atoms, normalize the molecule, reionize the molecule
    try:
        clean_mol = rdMolStandardize.Cleanup(mol) 
    except:
        print("Unable to clean molecule: " + str(smiles))
        return None, None, None, None, None, None, None, None    
    # if many fragments, get the "parent" (the actual mol we are interested in) 
    try:
        parent_clean_mol = rdMolStandardize.FragmentParent(clean_mol)
    except:
        print("Unable to fragment parent molecule: " + str(smiles))
        return None, None, None, None, None, None, None, None
    
    parent_clean_mol = check_aromatic_sulfoxides(parent_clean_mol)
    parent_clean_mol = check_P_O_charge(parent_clean_mol)
    
    charge_before = sum(atom.GetFormalCharge() for atom in parent_clean_mol.GetAtoms())
    
    # two different approaches to neutralizing charges
    uncharged_parent_clean_mol = uncharger.uncharge(parent_clean_mol)
    # manual double-check
    uncharged_parent_clean_mol2 = NeutraliseCharges(uncharged_parent_clean_mol)
    # run uncharger again
    uncharged_parent_clean_mol3 = uncharger.uncharge(uncharged_parent_clean_mol2)
    
    charge_after = sum(atom.GetFormalCharge() for atom in uncharged_parent_clean_mol3.GetAtoms())
    
    # tautomer enumerator
    if standardize_tautomers:
        try:
            Chem.SanitizeMol(uncharged_parent_clean_mol3)
            mol = te.Canonicalize(uncharged_parent_clean_mol3)
        except:
            print("Unable to canonicalize tautomers: " + str(smiles))
            mol = smiles = inchikey = None   
    else:
        mol = uncharged_parent_clean_mol3
        try:
            Chem.SanitizeMol(mol)
        except:
            print("Unable to sanitize molecule: " + str(smiles))
            mol = smiles = inchikey = None    
    
    """
    Not used: 
        Chem.GetSymmSSSR(mol)  # forces RDKit to find rings and perceive aromaticity
        Chem.SanitizeMol(mol, Chem.SANITIZE_SETAROMATICITY | Chem.SANITIZE_KEKULIZE)
    """
    
    # get SMILES and inchikey
    if mol is not None:
        try:
            smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=stereochem)

            mol = Chem.MolFromSmiles(smiles)
            mol = Chem.RemoveHs(mol)
            smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=stereochem)

            inchikey = Chem.inchi.MolToInchiKey(mol)
        except:
            print("Unable to convert molecule to SMILES or InChIKey: " + str(smiles))
            smiles = inchikey = None
    else:
        smiles = inchikey = None
    
    if charge_before != charge_after:
        duplicate = 'duplicate'
        # tautomer enumerator
        if standardize_tautomers:
            try:
                Chem.SanitizeMol(parent_clean_mol)
                mol_charge = te.Canonicalize(parent_clean_mol)
            except:
                print("Unable to canonicalize tautomers: " + str(smiles))
                mol_charge = smiles_charge = inchikey_charge = None    
        else:
            mol_charge = parent_clean_mol
            try:
                Chem.SanitizeMol(mol_charge)
            except:
                print("Unable to sanitize molecule: " + str(smiles))
                mol_charge = smiles_charge = inchikey_charge = None  
        
        if mol_charge is not None:
            try:
                smiles_charge = Chem.MolToSmiles(mol_charge, canonical=True, isomericSmiles=stereochem)

                mol_charge = Chem.MolFromSmiles(smiles_charge)
                mol_charge = Chem.RemoveHs(mol_charge)
                smiles_charge = Chem.MolToSmiles(mol_charge, canonical=True, isomericSmiles=stereochem)

                inchikey_charge = Chem.inchi.MolToInchiKey(mol_charge)  
            except:
                print("Unable to convert molecule to SMILES or InChIKey: " + str(smiles))
                smiles_charge = inchikey_charge = None    
    else:
        duplicate = 'unique'
        mol_charge = smiles_charge = inchikey_charge = None
    
    return mol, smiles, inchikey, mol_charge, smiles_charge, inchikey_charge, duplicate
