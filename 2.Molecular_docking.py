import subprocess
import numpy as np

def prepare_receptor(protein_pdb):
    """使用AutoDockTools转换受体格式"""
    subprocess.run(['prepare_receptor -r {protein_pdb} -o protein.pdbqt'])

def run_vina_docking(ligand_smiles, output_dir):
    """运行分子对接"""
    # 配体准备
    mol = Chem.MolFromSmiles(ligand_smiles)
    Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol)
    ligand_path = f"{output_dir}/ligand.pdbqt"
    Chem.MolToPDBFile(mol, ligand_path)
    
    # 运行对接
    vina_cmd = f"vina --receptor protein.pdbqt --ligand {ligand_path} \
              --center_x {cx} --center_y {cy} --center_z {cz} \
              --size_x {sx} --size_y {sy} --size_z {sz}"
    result = subprocess.run(vina_cmd.split(), capture_output=True)
    
    # 解析结合能
    affinity = float(result.stdout.split()[33])
    return affinity
