import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors

# 从ChEMBL数据库获取已知活性数据
def load_chembl_data(filepath):
    df = pd.read_csv(filepath)
    # 示例字段：['molecule_chembl_id', 'smiles', 'target_chembl_id', 'pIC50']
    return df[['smiles', 'pIC50']].dropna()

# 分子特征工程
def featurize_molecules(df):
    df['mol'] = df['smiles'].apply(Chem.MolFromSmiles)
    df = df[df['mol'].notnull()]
    
    # 生成分子指纹
    df['fp'] = df['mol'].apply(lambda x: AllChem.GetMorganFingerprintAsBitVect(x, 2, 2048))
    
    # 计算描述符
    df['mw'] = df['mol'].apply(Descriptors.MolWt)
    df['logp'] = df['mol'].apply(Descriptors.MolLogP)
    return df
