"""Drug-likeness rules (Lipinski, Veber), QED and PAINS substructure alerts."""

from __future__ import annotations

from functools import lru_cache
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import QED, Crippen, Descriptors, Lipinski, rdMolDescriptors
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

from .data import mol_from_smiles

PROFILE_COLUMNS = [
    "mw",
    "logp",
    "hbd",
    "hba",
    "tpsa",
    "rotatable_bonds",
    "qed",
    "lipinski_violations",
    "lipinski_pass",
    "veber_pass",
    "pains_alert",
]


@lru_cache(maxsize=None)
def _pains_catalog() -> FilterCatalog:
    params = FilterCatalogParams()
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
    return FilterCatalog(params)


def druglikeness_profile(mol: Optional[Chem.Mol]) -> Dict[str, object]:
    """Compute common drug-likeness properties for a single molecule."""
    if mol is None or mol.GetNumAtoms() == 0:
        return {}
    mw = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    hbd = Lipinski.NumHDonors(mol)
    hba = Lipinski.NumHAcceptors(mol)
    tpsa = rdMolDescriptors.CalcTPSA(mol)
    rotb = Lipinski.NumRotatableBonds(mol)
    lipinski_violations = int(mw > 500) + int(logp > 5) + int(hbd > 5) + int(hba > 10)
    pains = _pains_catalog().GetFirstMatch(mol)
    return {
        "mw": round(mw, 2),
        "logp": round(logp, 2),
        "hbd": hbd,
        "hba": hba,
        "tpsa": round(tpsa, 2),
        "rotatable_bonds": rotb,
        "qed": round(QED.qed(mol), 3),
        "lipinski_violations": lipinski_violations,
        "lipinski_pass": lipinski_violations <= 1,
        "veber_pass": rotb <= 10 and tpsa <= 140,
        "pains_alert": pains.GetDescription() if pains is not None else "",
    }


def druglikeness_table(smiles: Sequence[str]) -> pd.DataFrame:
    """Drug-likeness profile for each SMILES (rows for invalid SMILES are NaN)."""
    rows = [druglikeness_profile(mol_from_smiles(s)) for s in smiles]
    frame = pd.DataFrame(rows, index=range(len(rows)), columns=PROFILE_COLUMNS)
    frame.insert(0, "smiles", list(smiles))
    return frame


def passes_filters(
    profile: Dict[str, object],
    max_lipinski_violations: int = 1,
    require_veber: bool = True,
    reject_pains: bool = True,
    min_qed: float = 0.0,
) -> bool:
    if not {"lipinski_violations", "veber_pass", "pains_alert", "qed"} <= profile.keys():
        return False
    if any(
        pd.isna(profile[k]) for k in ("lipinski_violations", "veber_pass", "pains_alert", "qed")
    ):
        return False
    if profile["lipinski_violations"] > max_lipinski_violations:
        return False
    if require_veber and not profile["veber_pass"]:
        return False
    if reject_pains and profile["pains_alert"]:
        return False
    qed = profile.get("qed", np.nan)
    return bool(qed >= min_qed)
