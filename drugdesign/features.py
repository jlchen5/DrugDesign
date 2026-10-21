"""Molecular featurizers: fingerprints and physico-chemical descriptors.

Every featurizer is a callable ``featurizer(mols) -> np.ndarray`` of shape
``(n_mols, n_features)``. Rows for ``None`` molecules are filled with NaN so
callers can detect and drop them. New featurizers are added with
``@FEATURIZERS.register("name")``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, MACCSkeys, rdFingerprintGenerator

from .registry import Registry
from .validation import positive_int

FEATURIZERS: Registry["Featurizer"] = Registry("featurizer")

MolList = Sequence[Optional[Chem.Mol]]


class Featurizer:
    """Base class. Subclasses implement :meth:`featurize_one` and ``n_features``."""

    name = "base"

    @property
    def n_features(self) -> int:
        raise NotImplementedError

    @property
    def feature_names(self) -> List[str]:
        return [f"{self.name}_{i}" for i in range(self.n_features)]

    def featurize_one(self, mol: Chem.Mol) -> np.ndarray:
        raise NotImplementedError

    def __call__(self, mols: MolList) -> np.ndarray:
        out = np.full((len(mols), self.n_features), np.nan, dtype=np.float32)
        for i, mol in enumerate(mols):
            if mol is not None:
                out[i] = self.featurize_one(mol)
        return out

    def get_params(self) -> Dict[str, Any]:
        return {}

    def __repr__(self) -> str:
        params = ", ".join(f"{k}={v!r}" for k, v in self.get_params().items())
        return f"{type(self).__name__}({params})"


def _bitvect_to_array(fp, n_bits: int) -> np.ndarray:
    arr = np.zeros((n_bits,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


@FEATURIZERS.register("morgan")
class MorganFingerprint(Featurizer):
    """ECFP-like circular fingerprint (radius 2 ~ ECFP4)."""

    name = "morgan"

    def __init__(
        self,
        radius: int = 2,
        n_bits: int = 2048,
        use_counts: bool = False,
        use_chirality: bool = False,
    ) -> None:
        positive_int(radius, "radius", minimum=0)
        positive_int(n_bits, "n_bits")
        self.radius = radius
        self.n_bits = n_bits
        self.use_counts = use_counts
        self.use_chirality = use_chirality
        self._generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=radius, fpSize=n_bits, includeChirality=use_chirality
        )

    @property
    def n_features(self) -> int:
        return self.n_bits

    def featurize_one(self, mol: Chem.Mol) -> np.ndarray:
        if self.use_counts:
            return self._generator.GetCountFingerprintAsNumPy(mol).astype(np.float32)
        return self._generator.GetFingerprintAsNumPy(mol).astype(np.float32)

    def bit_vector(self, mol: Chem.Mol):
        """RDKit ExplicitBitVect, handy for Tanimoto similarity."""
        return self._generator.GetFingerprint(mol)

    def get_params(self) -> Dict[str, Any]:
        return {
            "radius": self.radius,
            "n_bits": self.n_bits,
            "use_counts": self.use_counts,
            "use_chirality": self.use_chirality,
        }

    # The RDKit generator is not picklable; rebuild it after unpickling.
    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("_generator", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=self.radius, fpSize=self.n_bits, includeChirality=self.use_chirality
        )


@FEATURIZERS.register("maccs")
class MACCSFingerprint(Featurizer):
    """166 MACCS structural keys (bit 0 is unused by RDKit and kept for indexing)."""

    name = "maccs"

    @property
    def n_features(self) -> int:
        return 167

    def featurize_one(self, mol: Chem.Mol) -> np.ndarray:
        return _bitvect_to_array(MACCSkeys.GenMACCSKeys(mol), 167)


@FEATURIZERS.register("rdkit_descriptors")
class RDKitDescriptors(Featurizer):
    """RDKit 2D descriptors. ``names=None`` uses every descriptor RDKit provides."""

    name = "desc"

    def __init__(self, names: Optional[Sequence[str]] = None) -> None:
        available = dict(Descriptors._descList)
        if names is None:
            names = list(available)
        unknown = [n for n in names if n not in available]
        if unknown:
            raise ValueError(f"Unknown RDKit descriptors: {unknown}")
        self.names = list(names)
        if not self.names:
            raise ValueError("Select at least one descriptor")
        self._funcs = [available[n] for n in self.names]

    @property
    def n_features(self) -> int:
        return len(self.names)

    @property
    def feature_names(self) -> List[str]:
        return list(self.names)

    def featurize_one(self, mol: Chem.Mol) -> np.ndarray:
        values = []
        for func in self._funcs:
            try:
                values.append(float(func(mol)))
            except Exception:
                values.append(np.nan)
        # Preserve failures: callers reject the row instead of inventing zero values.
        with np.errstate(over="ignore"):
            return np.asarray(values, dtype=np.float32)

    def get_params(self) -> Dict[str, Any]:
        return {"names": self.names}

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("_funcs", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        available = dict(Descriptors._descList)
        self._funcs = [available[n] for n in self.names]


# A compact, interpretable descriptor set (Lipinski/Veber properties).
BASIC_DESCRIPTORS = [
    "MolWt",
    "MolLogP",
    "TPSA",
    "NumHDonors",
    "NumHAcceptors",
    "NumRotatableBonds",
    "RingCount",
    "NumAromaticRings",
    "FractionCSP3",
    "HeavyAtomCount",
]


@FEATURIZERS.register("basic_descriptors")
def basic_descriptors() -> RDKitDescriptors:
    return RDKitDescriptors(BASIC_DESCRIPTORS)


class CombinedFeaturizer(Featurizer):
    """Concatenate the outputs of several featurizers column-wise."""

    name = "combined"

    def __init__(self, featurizers: Sequence[Featurizer]) -> None:
        if not featurizers:
            raise ValueError("CombinedFeaturizer needs at least one featurizer")
        self.featurizers = list(featurizers)

    @property
    def n_features(self) -> int:
        return sum(f.n_features for f in self.featurizers)

    @property
    def feature_names(self) -> List[str]:
        return [n for f in self.featurizers for n in f.feature_names]

    def __call__(self, mols: MolList) -> np.ndarray:
        return np.hstack([f(mols) for f in self.featurizers])

    def featurize_one(self, mol: Chem.Mol) -> np.ndarray:
        return np.concatenate([f.featurize_one(mol) for f in self.featurizers])

    def __repr__(self) -> str:
        return f"CombinedFeaturizer({self.featurizers!r})"


FeaturizerSpec = Union[str, Dict[str, Any], Featurizer, Sequence[Any]]


def build_featurizer(spec: FeaturizerSpec) -> Featurizer:
    """Build a featurizer from a flexible spec.

    Accepted forms: ``"morgan"``, ``{"name": "morgan", "radius": 3}``,
    an existing :class:`Featurizer`, or a list of any of these (concatenated).
    """
    if isinstance(spec, Featurizer):
        positive_int(spec.n_features, "n_features")
        return spec
    if isinstance(spec, str):
        return FEATURIZERS.create(spec)
    if isinstance(spec, dict):
        params = dict(spec)
        name = params.pop("name", None)
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"Featurizer spec needs a 'name' key: {spec}")
        return FEATURIZERS.create(name, **params)
    if isinstance(spec, (list, tuple)):
        parts = [build_featurizer(s) for s in spec]
        return parts[0] if len(parts) == 1 else CombinedFeaturizer(parts)
    raise TypeError(f"Cannot build a featurizer from {spec!r}")


def tanimoto_max_similarity(query_fps, reference_fps) -> np.ndarray:
    """For each query fingerprint, the max Tanimoto similarity to the reference set.

    Useful as a simple applicability-domain check for a trained model.
    """
    if not reference_fps:
        return np.full(len(query_fps), np.nan)
    out = np.full(len(query_fps), np.nan)
    for i, fp in enumerate(query_fps):
        if fp is not None:
            out[i] = max(DataStructs.BulkTanimotoSimilarity(fp, reference_fps))
    return out
