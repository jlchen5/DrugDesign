import shutil

import pytest
from rdkit import Chem

from drugdesign.docking import (DockingBox, DockingError, dock_smiles, docking_available,
                                embed_ligand, parse_vina_output, prepare_receptor)

VINA_STDOUT = """\
Performing docking (random seed: 42) ... done.

mode |   affinity | dist from best mode
     | (kcal/mol) | rmsd l.b.| rmsd u.b.
-----+------------+----------+----------
   1       -7.512          0          0
   2       -7.104      1.843      2.611
   3        -6.9       2.004      6.115
"""


def test_parse_vina_output():
    poses = parse_vina_output(VINA_STDOUT)
    assert [p.mode for p in poses] == [1, 2, 3]
    assert poses[0].affinity == -7.512
    assert poses[2].rmsd_ub == pytest.approx(6.115)
    assert parse_vina_output("no table here") == []


def test_box_args_and_validation():
    box = DockingBox((1, 2, 3), (10, 10, 10))
    args = box.vina_args()
    assert args[:4] == ["--center_x", "1.000", "--size_x", "10.000"]
    with pytest.raises(ValueError):
        DockingBox((0, 0, 0), (0, 10, 10))


def test_box_from_ligand(tmp_path):
    mol = embed_ligand("c1ccccc1CCO")
    path = tmp_path / "ref.sdf"
    Chem.MolToMolFile(mol, str(path))
    box = DockingBox.from_ligand(path, padding=5.0)
    assert all(s > 10 for s in box.size)


def test_embed_ligand_adds_hydrogens():
    mol = embed_ligand("CCO")
    assert mol.GetNumAtoms() == 9 and mol.GetNumConformers() == 1
    with pytest.raises(DockingError):
        embed_ligand("bad")


def test_prepare_receptor_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        prepare_receptor(tmp_path / "missing.pdb")


@pytest.mark.skipif(not (docking_available() and shutil.which("obabel")),
                    reason="AutoDock Vina / Open Babel not installed")
def test_dock_smiles_end_to_end(tmp_path):
    from rdkit.Chem import AllChem

    peptide = Chem.AddHs(Chem.MolFromSequence("WKFYHDLRW"))
    AllChem.EmbedMolecule(peptide, randomSeed=1)
    pdb = tmp_path / "receptor.pdb"
    Chem.MolToPDBFile(peptide, str(pdb))
    receptor = prepare_receptor(pdb, tmp_path / "receptor.pdbqt")
    center = tuple(peptide.GetConformer().GetPositions().mean(axis=0))
    result = dock_smiles("c1ccccc1O", receptor, DockingBox(center, (24, 24, 24)), tmp_path,
                         exhaustiveness=1, num_modes=3)
    assert result.poses and result.best_affinity < 0
    assert result.output_path.exists()
