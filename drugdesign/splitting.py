"""Structure-aware grouping shared by hold-out and cross-validation splits."""

import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


def molecule_groups(mols, strategy="random"):
    """Canonical structures (random) or Murcko scaffolds (scaffold).

    All acyclic molecules share the empty scaffold. This conservative choice
    prevents acyclic analogues leaking between scaffold partitions.
    """
    if strategy not in {"random", "scaffold"}:
        raise ValueError("Unknown split strategy")
    if strategy == "scaffold":
        return np.asarray(
            [MurckoScaffold.MurckoScaffoldSmiles(mol=m, includeChirality=False) for m in mols]
        )
    return np.asarray([Chem.MolToSmiles(m, isomericSmiles=True) for m in mols])
