"""Structure-based scoring with AutoDock Vina.

External tools (all optional, looked up on ``PATH``):

* ``vina`` – AutoDock Vina command-line program (or the ``vina`` Python package)
* ``mk_prepare_ligand.py`` / ``meeko`` Python package – ligand PDBQT preparation
* ``prepare_receptor`` (ADFR suite) or ``obabel`` (Open Babel) – receptor PDBQT preparation
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from .data import mol_from_smiles
from .validation import finite_number, positive_int

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]


class DockingError(RuntimeError):
    """Raised when a docking step cannot be performed."""


@dataclass
class DockingBox:
    center: Tuple[float, float, float]
    size: Tuple[float, float, float] = (20.0, 20.0, 20.0)

    def __post_init__(self) -> None:
        if len(self.center) != 3 or len(self.size) != 3:
            raise ValueError("center and size must both have three components")
        self.center = tuple(float(c) for c in self.center)
        self.size = tuple(float(s) for s in self.size)
        if not np.isfinite(self.center + self.size).all():
            raise ValueError("Box coordinates and size must be finite")
        if any(s <= 0 for s in self.size):
            raise ValueError(f"Box size must be positive, got {self.size}")

    @classmethod
    def from_ligand(cls, path: PathLike, padding: float = 8.0) -> "DockingBox":
        """Box enclosing a reference (co-crystallised) ligand plus ``padding`` Å per side."""
        path = Path(path)
        finite_number(padding, "padding", minimum=0)
        if not path.is_file():
            raise FileNotFoundError(path)
        suffix = path.suffix.lower()
        if suffix == ".pdb":
            mol = Chem.MolFromPDBFile(str(path), removeHs=False, sanitize=False)
        elif suffix in {".sdf", ".mol"}:
            mol = Chem.MolFromMolFile(str(path), removeHs=False, sanitize=False)
        elif suffix == ".mol2":
            mol = Chem.MolFromMol2File(str(path), removeHs=False, sanitize=False)
        else:
            raise ValueError(f"Unsupported reference ligand format: {suffix}")
        if mol is None or mol.GetNumAtoms() == 0 or mol.GetNumConformers() == 0:
            raise DockingError(f"Could not read 3D coordinates from {path}")
        if not mol.GetConformer().Is3D():
            raise DockingError(f"Reference ligand must have 3D coordinates: {path}")
        coords = mol.GetConformer().GetPositions()
        lo, hi = coords.min(axis=0), coords.max(axis=0)
        center = (lo + hi) / 2
        size = (hi - lo) + 2 * padding
        return cls(tuple(center), tuple(size))

    def vina_args(self) -> List[str]:
        args = []
        for axis, c, s in zip("xyz", self.center, self.size):
            args += [f"--center_{axis}", f"{c:.3f}", f"--size_{axis}", f"{s:.3f}"]
        return args


@dataclass
class DockingPose:
    mode: int
    affinity: float  # kcal/mol, more negative = stronger binding
    rmsd_lb: float
    rmsd_ub: float


@dataclass
class DockingResult:
    smiles: str
    poses: List[DockingPose] = field(default_factory=list)
    output_path: Optional[Path] = None

    @property
    def best_affinity(self) -> float:
        return min(p.affinity for p in self.poses) if self.poses else float("nan")


def _require(executable: str) -> str:
    path = shutil.which(executable)
    if path is None:
        raise DockingError(
            f"Required program '{executable}' was not found on PATH. "
            "See README for installing AutoDock Vina / Open Babel / Meeko."
        )
    return path


def _run(cmd: Sequence[str], timeout: Optional[float] = None) -> subprocess.CompletedProcess:
    logger.debug("Running: %s", " ".join(map(str, cmd)))
    try:
        result = subprocess.run(
            [str(c) for c in cmd], capture_output=True, text=True, timeout=timeout
        )
    except subprocess.TimeoutExpired as exc:
        raise DockingError(f"Command timed out after {timeout}s: {cmd[0]}") from exc
    except OSError as exc:
        raise DockingError(f"Cannot run {cmd[0]}: {exc}") from exc
    if result.returncode != 0:
        raise DockingError(
            f"Command failed ({result.returncode}): {' '.join(map(str, cmd))}\n"
            f"{result.stderr.strip() or result.stdout.strip()}"
        )
    return result


# --- Receptor -----------------------------------------------------------------


def prepare_receptor(
    protein_pdb: PathLike, output_pdbqt: Optional[PathLike] = None, ph: float = 7.4
) -> Path:
    """Convert a protein PDB file to PDBQT (ADFR ``prepare_receptor`` or Open Babel)."""
    protein_pdb = Path(protein_pdb)
    if not protein_pdb.exists():
        raise FileNotFoundError(f"Receptor file not found: {protein_pdb}")
    output = Path(output_pdbqt) if output_pdbqt else protein_pdb.with_suffix(".pdbqt")
    if output.resolve() == protein_pdb.resolve():
        raise ValueError("Receptor output must not overwrite the input")
    output.parent.mkdir(parents=True, exist_ok=True)

    if shutil.which("prepare_receptor"):
        _run(["prepare_receptor", "-r", protein_pdb, "-o", output, "-A", "hydrogens"])
    elif shutil.which("obabel"):
        _run(["obabel", protein_pdb, "-xr", "-p", str(ph), "-O", output])
    else:
        raise DockingError("Neither 'prepare_receptor' (ADFR) nor 'obabel' found on PATH")
    return output


# --- Ligand -------------------------------------------------------------------


def embed_ligand(smiles: str, random_seed: int = 42, optimize: bool = True) -> Chem.Mol:
    """Add explicit hydrogens and embed a 3D conformer (ETKDGv3 + MMFF94).

    Does not choose a pH-dependent protonation state or enumerate tautomers.
    """
    positive_int(random_seed, "random_seed", minimum=0)
    if random_seed > 2**31 - 1:
        raise ValueError("random_seed must fit a signed 32-bit integer")
    mol = mol_from_smiles(smiles)
    if mol is None:
        raise DockingError(f"Invalid SMILES: {smiles}")
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = random_seed
    if AllChem.EmbedMolecule(mol, params) != 0:
        params.useRandomCoords = True
        if AllChem.EmbedMolecule(mol, params) != 0:
            raise DockingError(f"3D embedding failed for {smiles}")
    if optimize and AllChem.MMFFHasAllMoleculeParams(mol):
        AllChem.MMFFOptimizeMolecule(mol, maxIters=2000)
    return mol


def prepare_ligand(smiles: str, output_pdbqt: PathLike, random_seed: int = 42) -> Path:
    """Write a dockable PDBQT file for ``smiles`` using Meeko, or Open Babel as fallback."""
    output = Path(output_pdbqt)
    output.parent.mkdir(parents=True, exist_ok=True)
    mol = embed_ligand(smiles, random_seed=random_seed)

    try:
        from meeko import MoleculePreparation, PDBQTWriterLegacy
    except ImportError:
        MoleculePreparation = None

    if MoleculePreparation is not None:
        try:
            setups = MoleculePreparation().prepare(mol)
            if not setups:
                raise DockingError(f"Meeko produced no ligand setups for {smiles}")
            pdbqt, ok, error = PDBQTWriterLegacy.write_string(setups[0])
        except Exception as exc:
            raise DockingError(f"Meeko preparation failed for {smiles}: {exc}") from exc
        if not ok:
            raise DockingError(f"Meeko failed for {smiles}: {error}")
        output.write_text(pdbqt, encoding="utf-8")
        return output

    _require("obabel")
    sdf_path = output.with_suffix(".sdf")
    Chem.MolToMolFile(mol, str(sdf_path))
    _run(["obabel", sdf_path, "-O", output, "--partialcharge", "gasteiger"])
    return output


# --- Vina ---------------------------------------------------------------------

_VINA_ROW = re.compile(r"^\s*(\d+)\s+(-?\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s*$")


def parse_vina_output(text: str) -> List[DockingPose]:
    """Parse the result table printed by the ``vina`` command-line program."""
    poses = []
    in_table = False
    for line in text.splitlines():
        if line.lstrip().startswith("-----+"):
            in_table = True
            continue
        if in_table:
            match = _VINA_ROW.match(line)
            if match:
                mode, aff, lb, ub = match.groups()
                poses.append(DockingPose(int(mode), float(aff), float(lb), float(ub)))
            elif poses:
                break
    return poses


def run_vina(
    receptor_pdbqt: PathLike,
    ligand_pdbqt: PathLike,
    box: DockingBox,
    output_pdbqt: Optional[PathLike] = None,
    exhaustiveness: int = 8,
    num_modes: int = 9,
    seed: int = 42,
    cpu: int = 0,
    timeout: Optional[float] = None,
) -> List[DockingPose]:
    """Dock one prepared ligand and return its poses (best first)."""
    positive_int(exhaustiveness, "exhaustiveness")
    positive_int(num_modes, "num_modes")
    positive_int(cpu, "cpu", minimum=0)
    positive_int(seed, "seed", minimum=0)
    if timeout is not None:
        finite_number(timeout, "timeout", minimum=0.001)
    receptor_pdbqt, ligand_pdbqt = Path(receptor_pdbqt), Path(ligand_pdbqt)
    for p in (receptor_pdbqt, ligand_pdbqt):
        if not p.exists():
            raise FileNotFoundError(p)
    output = (
        Path(output_pdbqt)
        if output_pdbqt
        else ligand_pdbqt.with_name(ligand_pdbqt.stem + "_out.pdbqt")
    )
    if output.resolve() in {receptor_pdbqt.resolve(), ligand_pdbqt.resolve()}:
        raise ValueError("Vina output must not overwrite a receptor or ligand input")
    output.parent.mkdir(parents=True, exist_ok=True)

    if shutil.which("vina"):
        cmd = [
            "vina",
            "--receptor",
            receptor_pdbqt,
            "--ligand",
            ligand_pdbqt,
            *box.vina_args(),
            "--exhaustiveness",
            str(exhaustiveness),
            "--num_modes",
            str(num_modes),
            "--seed",
            str(seed),
            "--out",
            output,
        ]
        if cpu:
            cmd += ["--cpu", str(cpu)]
        poses = parse_vina_output(_run(cmd, timeout=timeout).stdout)
    else:
        if timeout is not None:
            raise DockingError("timeout requires the Vina CLI; Python Vina cannot enforce it")
        try:
            poses = _run_vina_python(
                receptor_pdbqt, ligand_pdbqt, box, output, exhaustiveness, num_modes, seed, cpu
            )
        except (RuntimeError, ValueError) as exc:
            raise DockingError(f"Vina failed: {exc}") from exc
    if not poses:
        raise DockingError(f"Vina produced no poses for {ligand_pdbqt}")
    if not output.is_file() or output.stat().st_size == 0:
        raise DockingError(f"Vina did not write poses to {output}")
    return poses


def _run_vina_python(receptor, ligand, box, output, exhaustiveness, num_modes, seed, cpu):
    try:
        from vina import Vina
    except ImportError as exc:
        raise DockingError(
            "AutoDock Vina not found: install the 'vina' program or Python package"
        ) from exc
    v = Vina(sf_name="vina", cpu=cpu, seed=seed, verbosity=0)
    v.set_receptor(str(receptor))
    v.set_ligand_from_file(str(ligand))
    v.compute_vina_maps(center=list(box.center), box_size=list(box.size))
    v.dock(exhaustiveness=exhaustiveness, n_poses=num_modes)
    v.write_poses(str(output), n_poses=num_modes, overwrite=True)
    energies = np.atleast_2d(v.energies(n_poses=num_modes))
    # The Python API does not report RMSD to the best pose.
    return [
        DockingPose(i + 1, float(e[0]), float("nan"), float("nan")) for i, e in enumerate(energies)
    ]


def dock_smiles(
    smiles: str,
    receptor_pdbqt: PathLike,
    box: DockingBox,
    output_dir: PathLike,
    name: Optional[str] = None,
    **vina_kwargs,
) -> DockingResult:
    """Prepare and dock a single SMILES. Files go to ``output_dir/<name>*.pdbqt``."""
    output_dir = Path(output_dir)
    name = name or "ligand"
    if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        raise ValueError("Ligand name may contain only letters, digits, underscores and hyphens")
    receptor_path = Path(receptor_pdbqt).resolve()
    if receptor_path in {
        (output_dir / f"{name}.pdbqt").resolve(),
        (output_dir / f"{name}_out.pdbqt").resolve(),
    }:
        raise ValueError("Ligand output must not overwrite the receptor")
    ligand = prepare_ligand(
        smiles, output_dir / f"{name}.pdbqt", random_seed=vina_kwargs.get("seed", 42)
    )
    out = output_dir / f"{name}_out.pdbqt"
    poses = run_vina(receptor_pdbqt, ligand, box, out, **vina_kwargs)
    return DockingResult(smiles=smiles, poses=poses, output_path=out)


def docking_available() -> bool:
    """True if Vina and a ligand-preparation route are both available."""
    has_vina = shutil.which("vina") is not None or _module_available("vina")
    has_ligprep = _module_available("meeko") or shutil.which("obabel") is not None
    return has_vina and has_ligprep


def _module_available(name: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(name) is not None
