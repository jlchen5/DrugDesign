import pickle

import numpy as np
import pytest
from rdkit import Chem

from drugdesign.features import (FEATURIZERS, CombinedFeaturizer, MorganFingerprint,
                                 RDKitDescriptors, build_featurizer)

MOLS = [Chem.MolFromSmiles(s) for s in ["CCO", "c1ccccc1O"]] + [None]


def test_morgan_shapes_and_invalid_rows():
    X = MorganFingerprint(n_bits=1024)(MOLS)
    assert X.shape == (3, 1024)
    assert set(np.unique(X[:2])) <= {0.0, 1.0}
    assert np.isnan(X[2]).all()


def test_morgan_counts():
    X = MorganFingerprint(n_bits=512, use_counts=True)(MOLS[:2])
    assert X.max() >= 1


def test_descriptors_named():
    feat = RDKitDescriptors(["MolWt", "TPSA"])
    X = feat(MOLS[:1])
    assert feat.feature_names == ["MolWt", "TPSA"]
    assert X[0, 0] == pytest.approx(46.07, abs=0.01)
    with pytest.raises(ValueError):
        RDKitDescriptors(["NotADescriptor"])


def test_build_featurizer_specs():
    assert isinstance(build_featurizer("morgan"), MorganFingerprint)
    assert build_featurizer({"name": "morgan", "radius": 3}).radius == 3
    combo = build_featurizer(["maccs", "basic_descriptors"])
    assert isinstance(combo, CombinedFeaturizer)
    assert combo(MOLS).shape == (3, 167 + 10)
    assert len(combo.feature_names) == combo.n_features
    with pytest.raises(KeyError):
        build_featurizer("unknown")
    with pytest.raises(ValueError):
        build_featurizer({"radius": 2})


def test_featurizers_are_picklable():
    for name in FEATURIZERS.names():
        feat = FEATURIZERS.create(name)
        clone = pickle.loads(pickle.dumps(feat))
        np.testing.assert_array_equal(clone(MOLS[:2]), feat(MOLS[:2]))


def test_registry_rejects_duplicates():
    with pytest.raises(KeyError):
        FEATURIZERS.register("morgan")(MorganFingerprint)
