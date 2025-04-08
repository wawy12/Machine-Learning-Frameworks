import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from rdkit.ML.Descriptors import MoleculeDescriptors

RU_smiles = pd.read_csv('./polymer_content_data.csv')

if not os.path.exists('./Repeat_Units1'):
    os.mkdir('./Repeat_Units1')

for i, row in RU_smiles.iterrows():
    RU_name = row['Repeat_Unit']
    smiles = row['smiles']
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    w = Chem.SDWriter(f'./Repeat_Units1/{RU_name}.sdf')
    w.write(mol)
    w.close()

    descriptors_dict = {}

for i, row in RU_smiles.iterrows():
    RU_name = row['Repeat_Unit']
    smiles = row['smiles']
    sdf_path = f'./Repeat_Units1/{RU_name}.sdf'
    sdf_supplier = Chem.SDMolSupplier(sdf_path)

    for mol in sdf_supplier:
        if mol is not None:
            descriptor_calculator = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)
            descriptors = descriptor_calculator.CalcDescriptors(mol)
            descriptors_dict[RU_name] = descriptors 

descriptors_df = pd.DataFrame.from_dict(descriptors_dict, orient='index', columns=descriptor_names)
descriptors_df.index.name = 'Repeat_Unit'  
descriptors_df.to_csv('polymer_content_molecule_descriptors_final.csv')
