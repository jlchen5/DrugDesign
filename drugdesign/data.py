"""Loading, standardising and cleaning bioactivity data (e.g. ChEMBL exports)."""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]

# Multipliers that convert a concentration unit to molar.
_UNIT_TO_MOLAR = {"m": 1.0, "mm": 1e-3, "um": 1e-6, "µm": 1e-6, "nm": 1e-9, "pm": 1e-12}


def quiet_rdkit(quiet: bool = True) -> None:
    """Silence (or re-enable) RDKit's parser warnings on stderr."""
    if quiet:
        RDLogger.DisableLog("rdApp.*")
    else:
        RDLogger.EnableLog("rdApp.*")


def to_pactivity(values: Union[float, Sequence[float], np.ndarray], unit: str = "nM") -> np.ndarray:
    """Convert IC50/Ki/EC50 concentrations to p-values: pX = -log10(X [M]).

    Non-positive values become NaN instead of raising.
    """
    factor = _UNIT_TO_MOLAR.get(unit.strip().lower())
    if factor is None:
        raise ValueError(f"Unsupported unit '{unit}'. Use one of {sorted(_UNIT_TO_MOLAR)}")
    arr = np.asarray(values, dtype=float) * factor
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(arr > 0, -np.log10(arr), np.nan)


def mol_from_smiles(smiles: Optional[str]) -> Optional[Chem.Mol]:
    """Parse a SMILES string, returning None for empty or invalid input."""
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    return Chem.MolFromSmiles(smiles.strip())


def mols_from_smiles(smiles: Iterable[Optional[str]]) -> List[Optional[Chem.Mol]]:
    return [mol_from_smiles(s) for s in smiles]


@lru_cache(maxsize=None)
def _fragment_chooser() -> rdMolStandardize.LargestFragmentChooser:
    return rdMolStandardize.LargestFragmentChooser()


@lru_cache(maxsize=None)
def _uncharger() -> rdMolStandardize.Uncharger:
    return rdMolStandardize.Uncharger()


def standardize_smiles(smiles: Optional[str], keep_largest_fragment: bool = True,
                       neutralize: bool = True) -> Optional[str]:
    """Return a canonical SMILES (salts stripped, charges neutralised) or None if invalid."""
    mol = mol_from_smiles(smiles)
    if mol is None:
        return None
    try:
        if keep_largest_fragment:
            mol = _fragment_chooser().choose(mol)
        if neutralize:
            mol = _uncharger().uncharge(mol)
        return Chem.MolToSmiles(mol)
    except Exception:  # RDKit raises a variety of sanitisation errors
        return None


def load_activity_data(
    path: PathLike,
    smiles_col: str = "smiles",
    target_col: str = "pIC50",
    value_col: Optional[str] = None,
    unit: str = "nM",
    sep: Optional[str] = None,
) -> pd.DataFrame:
    """Read a CSV/TSV file and return a frame with ``smiles`` and ``target_col`` columns.

    If ``target_col`` is missing but ``value_col`` (e.g. ChEMBL ``standard_value``
    in nM) is given, p-activities are computed from it.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Activity file not found: {path}")
    if sep is None:
        sep = "\t" if path.suffix.lower() in {".tsv", ".tab"} else ","
    # ChEMBL CSV exports are ';'-separated; sniff that case.
    df = pd.read_csv(path, sep=sep)
    if df.shape[1] == 1 and sep == "," and ";" in str(df.columns[0]):
        df = pd.read_csv(path, sep=";")

    if smiles_col not in df.columns:
        raise KeyError(f"SMILES column '{smiles_col}' not in {list(df.columns)}")

    if target_col not in df.columns:
        if value_col is None or value_col not in df.columns:
            raise KeyError(
                f"Target column '{target_col}' not found and no usable value_col given "
                f"(columns: {list(df.columns)})"
            )
        df[target_col] = to_pactivity(pd.to_numeric(df[value_col], errors="coerce"), unit)

    out = pd.DataFrame({
        "smiles": df[smiles_col],
        target_col: pd.to_numeric(df[target_col], errors="coerce"),
    })
    return out.dropna().reset_index(drop=True)


def clean_dataset(
    df: pd.DataFrame,
    target_col: str = "pIC50",
    standardize: bool = True,
    aggregate: str = "median",
) -> pd.DataFrame:
    """Drop invalid structures and merge duplicate compounds.

    Duplicates (same canonical SMILES) are aggregated with ``aggregate``
    ('mean', 'median', 'max', 'min' or 'first').
    """
    n_in = len(df)
    df = df.dropna(subset=["smiles", target_col]).copy()
    if standardize:
        df["smiles"] = [standardize_smiles(s) for s in df["smiles"]]
    else:
        df["smiles"] = [
            Chem.MolToSmiles(m) if m is not None else None for m in mols_from_smiles(df["smiles"])
        ]
    n_invalid = int(df["smiles"].isna().sum())
    df = df.dropna(subset=["smiles"])

    df = df.groupby("smiles", as_index=False, sort=False)[target_col].agg(aggregate)
    logger.info(
        "Cleaned dataset: %d rows in, %d invalid SMILES removed, %d unique compounds out",
        n_in, n_invalid, len(df),
    )
    return df.reset_index(drop=True)


def read_smiles_file(path: PathLike, smiles_col: str = "smiles") -> pd.DataFrame:
    """Read candidate molecules from .csv/.tsv (with a SMILES column) or .smi files."""
    path = Path(path)
    if path.suffix.lower() in {".smi", ".smiles", ".txt"}:
        rows = []
        for line in path.read_text().splitlines():
            parts = line.split()
            if not parts or parts[0].startswith("#"):
                continue
            rows.append({"smiles": parts[0], "name": parts[1] if len(parts) > 1 else None})
        return pd.DataFrame(rows, columns=["smiles", "name"])
    sep = "\t" if path.suffix.lower() in {".tsv", ".tab"} else ","
    df = pd.read_csv(path, sep=sep)
    if smiles_col not in df.columns:
        raise KeyError(f"SMILES column '{smiles_col}' not in {list(df.columns)}")
    return df.rename(columns={smiles_col: "smiles"})
