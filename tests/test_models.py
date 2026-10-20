import numpy as np
import pytest

from drugdesign.filters import druglikeness_profile, druglikeness_table, passes_filters
from drugdesign.models import MODELS, QSARModel, regression_metrics
from rdkit import Chem


def test_regression_metrics_perfect():
    m = regression_metrics([1, 2, 3], [1, 2, 3])
    assert m["rmse"] == 0 and m["r2"] == 1 and m["pearson_r"] == pytest.approx(1)


@pytest.mark.parametrize("estimator", ["random_forest", "ridge", "knn"])
def test_train_predict_roundtrip(demo_data, tmp_path, estimator):
    model = QSARModel("morgan", estimator, random_state=0)
    report = model.train_and_evaluate(demo_data["smiles"], demo_data["pIC50"],
                                      test_size=0.2, cv_folds=3)
    assert report.n_test > 0 and "rmse" in report.test_metrics and "r2" in report.cv_metrics

    preds = model.predict(["CCOc1ccccc1NC(=O)C", "invalid"])
    assert np.isfinite(preds[0]) and np.isnan(preds[1])

    path = model.save(tmp_path / "m.joblib")
    loaded = QSARModel.load(path)
    np.testing.assert_allclose(loaded.predict(["CCOc1ccccc1NC(=O)C"]), preds[:1])

    frame = loaded.predict_frame(["CCOc1ccccc1NC(=O)C", "invalid"])
    assert frame["valid"].tolist() == [True, False]
    assert 0 <= frame["max_train_similarity"][0] <= 1


def test_descriptor_model_is_deterministic(demo_data):
    preds = []
    for _ in range(2):
        model = QSARModel(["basic_descriptors"], "random_forest",
                          {"n_estimators": 50}, random_state=1)
        model.fit(demo_data["smiles"], demo_data["pIC50"])
        preds.append(model.predict(["c1ccccc1O"]))
    np.testing.assert_allclose(*preds)


def test_classifier(demo_data):
    y = (demo_data["pIC50"] > demo_data["pIC50"].median()).astype(int)
    model = QSARModel("morgan", "random_forest_classifier", {"n_estimators": 50})
    report = model.train_and_evaluate(demo_data["smiles"], y, cv_folds=3)
    assert "accuracy" in report.cv_metrics and "roc_auc" in report.test_metrics
    assert "probability_active" in model.predict_frame(["CCO"]).columns


def test_fit_needs_valid_molecules():
    with pytest.raises(ValueError):
        QSARModel().fit(["bad", "worse"], [1.0, 2.0])


def test_unknown_model():
    with pytest.raises(KeyError):
        QSARModel(estimator="nope")
    assert "random_forest" in MODELS


def test_druglikeness():
    prof = druglikeness_profile(Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O"))
    assert prof["lipinski_violations"] == 0 and prof["veber_pass"]
    assert passes_filters(prof)
    assert not passes_filters({})
    table = druglikeness_table(["CCO", "bad"])
    assert len(table) == 2 and np.isnan(table.loc[1, "qed"])
