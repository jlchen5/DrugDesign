"""Example plugin: register a custom featurizer and model.

Use it with ``drugdesign --plugin examples/plugins/custom_featurizer.py train ...``
or list it under ``plugins = [...]`` in a pipeline config. Afterwards
``--featurizer atom_pair`` and ``--model hist_gb`` become available.
"""

import numpy as np
from rdkit.Chem import rdFingerprintGenerator
from sklearn.ensemble import HistGradientBoostingRegressor

from drugdesign.features import FEATURIZERS, Featurizer
from drugdesign.models import MODELS


@FEATURIZERS.register("atom_pair")
class AtomPairFingerprint(Featurizer):
    name = "atom_pair"

    def __init__(self, n_bits: int = 2048) -> None:
        self.n_bits = n_bits

    @property
    def n_features(self) -> int:
        return self.n_bits

    def featurize_one(self, mol) -> np.ndarray:
        # Generators are cheap to build and not picklable, so build per call.
        gen = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=self.n_bits)
        return gen.GetCountFingerprintAsNumPy(mol).astype(np.float32)

    def get_params(self):
        return {"n_bits": self.n_bits}


@MODELS.register("hist_gb")
def hist_gradient_boosting(random_state=None, **params):
    return HistGradientBoostingRegressor(random_state=random_state, **params)
