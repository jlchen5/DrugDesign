"""Regression tests for scientific validity and public API edge cases."""

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from drugdesign import PipelineConfig, QSARModel
from drugdesign import models as model_module
from drugdesign.data import clean_dataset, mols_from_smiles, to_pactivity
from drugdesign.filters import druglikeness_table, passes_filters
from drugdesign.models import TrainingReport, classification_metrics, regression_metrics
from drugdesign.pipeline import predict_candidates
from drugdesign.splitting import molecule_groups


def small_model(**kwargs):
    return QSARModel("basic_descriptors", "ridge", **kwargs)


def test_preprocessing_is_identical_after_save(tmp_path):
    m = small_model().fit(["CC(=O)O", "CCN", "CCCO"], [1, 2, 3])
    query = ["CC(=O)O", "CC(=O)[O-].[Na+]", "bad"]
    frame = QSARModel.load(m.save(tmp_path / "m.joblib")).predict_frame(query)
    assert frame.loc[0, "model_smiles"] == frame.loc[1, "model_smiles"]
    assert frame.loc[0, "predicted_pIC50"] == frame.loc[1, "predicted_pIC50"]
    assert frame.loc[1, "max_train_similarity"] == 1
    assert frame["valid"].tolist() == [True, True, False]
    raw = small_model(standardize=False).fit(["CCO", "CCN"], [1, 2])
    assert "." in raw.predict_frame([query[1]]).loc[0, "model_smiles"]


def test_clean_before_split_and_features_computed_once(monkeypatch, demo_data):
    m = small_model()
    original = m.featurize
    calls = []

    def spy(smiles):
        calls.append(1)
        return original(smiles)

    monkeypatch.setattr(m, "featurize", spy)
    smiles = list(demo_data.smiles) + ["bad", "CCO", "CCN"]
    y = list(demo_data.pIC50) + [5, np.inf, np.nan]
    report = m.train_and_evaluate(smiles, y, cv_folds=3)
    assert len(calls) == 1
    assert report.n_rejected == 3
    assert report.n_train + report.n_test == report.n_valid == len(demo_data)
    assert report.n_final_fit == len(demo_data)
    assert report.metadata["cv_scope"] == "holdout_training_partition"


@pytest.mark.parametrize("strategy", ["random", "scaffold"])
def test_no_structure_overlap_and_cv_excludes_holdout(monkeypatch, demo_data, strategy):
    m = small_model()
    smiles = list(demo_data.smiles) * 2
    y = list(demo_data.pIC50) * 2
    X, _, mols = m.featurize(smiles)
    groups = molecule_groups(mols, strategy)
    heldout_indices = []
    cv_seen = []
    original_cv = model_module.cross_val_outputs
    original_shuffle = model_module.GroupShuffleSplit
    original_random = model_module.train_test_split

    class SpyShuffle(original_shuffle):
        def split(self, X, y, groups):
            for tr, te in super().split(X, y, groups):
                assert set(groups[tr]).isdisjoint(groups[te])
                heldout_indices.extend(te)
                yield tr, te

    def random_split(unique, **kwargs):
        tr, te = original_random(unique, **kwargs)
        assert set(tr).isdisjoint(te)
        heldout_indices.extend(np.flatnonzero(np.isin(groups, te)))
        return tr, te

    def cv_spy(estimator, Xtr, ytr, cv, **kwargs):
        expected = np.flatnonzero(~np.isin(np.arange(len(y)), heldout_indices))
        np.testing.assert_array_equal(Xtr, X[expected])
        gtr = groups[expected]
        for train, test in cv:
            assert set(gtr[train]).isdisjoint(gtr[test])
        cv_seen.append(True)
        return original_cv(estimator, Xtr, ytr, cv=cv, **kwargs)

    monkeypatch.setattr(model_module, "GroupShuffleSplit", SpyShuffle)
    monkeypatch.setattr(model_module, "train_test_split", random_split)
    monkeypatch.setattr(model_module, "cross_val_outputs", cv_spy)
    report = m.train_and_evaluate(smiles, y, cv_folds=3, split_strategy=strategy)
    assert cv_seen and heldout_indices
    assert report.n_test == len(heldout_indices)


def test_acyclic_scaffolds_stay_together():
    groups = molecule_groups(
        mols_from_smiles(["CCO", "CCCCN", "c1ccccc1", "Cc1ccccc1"]), "scaffold"
    )
    assert groups[0] == groups[1] == ""
    assert groups[2] == groups[3] != ""
    with pytest.raises(ValueError):
        small_model().train_and_evaluate(
            ["CCO", "CCN", "CCCO"], [1, 2, 3], split_strategy="scaffold", cv_folds=0
        )


@pytest.mark.parametrize(
    "options",
    [
        {"test_size": -1},
        {"test_size": 1},
        {"test_size": float("nan")},
        {"cv_folds": 1},
        {"cv_folds": -2},
        {"cv_folds": True},
        {"split_strategy": "bogus"},
    ],
)
def test_invalid_evaluation_options(options):
    with pytest.raises(ValueError):
        small_model().train_and_evaluate(["CCO", "CCN"], [1, 2], **options)


def test_requested_evaluation_never_silently_skipped():
    with pytest.raises(ValueError, match="groups"):
        small_model().train_and_evaluate(["CCO", "CCN", "CCCO"], [1, 2, 3], cv_folds=5)
    r = small_model().train_and_evaluate(["CCO", "CCN"], [1, 2], cv_folds=0, test_size=0)
    assert r.n_test == 0 and r.test_metrics == r.cv_metrics == {}


def test_fit_lengths_unfitted_and_empty_predictions():
    with pytest.raises(ValueError, match="equal lengths"):
        small_model().fit(["CCO", "CCN"], [1])
    with pytest.raises(NotFittedError):
        small_model().predict(["bad"])
    with pytest.raises(NotFittedError):
        small_model().save("never_written.joblib")
    m = small_model().fit(["CCO", "CCN"], [1, 2])
    assert m.predict([]).shape == (0,)
    assert m.predict_frame([]).empty
    np.testing.assert_array_equal(m.predict(iter(["CCO"])), m.predict("CCO"))
    with pytest.raises(ValueError, match="No valid"):
        m.evaluate(["bad"], [1])


def test_report_json_is_strict_and_fit_clears_old_report(tmp_path):
    r = TrainingReport(1, 1, test_metrics={"r2": np.nan})
    assert json.loads(r.save(tmp_path / "report.json").read_text())["test_metrics"]["r2"] is None
    m = small_model()
    m.train_and_evaluate(["CCO", "CCN"], [1, 2], test_size=0, cv_folds=0)
    m.fit(["CCO", "CCN"], [2, 3])
    assert m.report is None
    assert np.isnan(regression_metrics([1, 1], [1, 2])["pearson_r"])


def test_numeric_class_labels_and_probabilities(demo_data):
    y = (demo_data.pIC50 > demo_data.pIC50.median()).astype(int) + 1
    m = QSARModel("basic_descriptors", "logistic_regression")
    r = m.train_and_evaluate(demo_data.smiles, y, cv_folds=3)
    assert "roc_auc" in r.cv_metrics and "roc_auc" in r.test_metrics
    f = m.predict_frame(["CCO", "bad"])
    assert "probability_active" not in f
    assert f.loc[0, ["probability_class_1", "probability_class_2"]].sum() == pytest.approx(1)
    invalid = m.predict_frame(["bad"])
    assert "probability_class_2" in invalid and invalid["probability_class_2"].isna().all()
    assert classification_metrics([1, 2], [1, 2])["f1"] == 1


def test_multiclass_outputs_do_not_assume_active_class():
    m = QSARModel("basic_descriptors", "logistic_regression").fit(
        ["CCO", "CCN", "CCC", "CCCC", "CCCO", "CCCN"], [0, 1, 2, 0, 1, 2]
    )
    frame = m.predict_frame(["CCO", "bad"])
    assert "probability_active" not in frame
    assert frame.filter(like="probability_class_").iloc[0].sum() == pytest.approx(1)
    assert classification_metrics([0, 1, 2], [0, 1, 2])["f1"] == 1


@pytest.mark.parametrize("targets", [[0, 0], [0, 0.5], [1, 2]])
def test_reject_invalid_or_conflicting_classes(targets):
    with pytest.raises(ValueError):
        QSARModel(estimator="logistic_regression").fit(["CCO", "OCC"], targets)


def test_batch_predictions_keep_metadata_and_schema():
    m = small_model().fit(["CCO", "CCN"], [1, 2])
    candidates = pd.DataFrame(
        {
            "smiles": ["CCO", "bad", "CCN"],
            "valid": ["raw"] * 3,
            "qed": [999] * 3,
            "input_qed": [123] * 3,
            "predicted_other": [777] * 3,
        },
        index=[8, 2, 7],
    )
    a = predict_candidates(m, candidates, batch_size=1)
    b = predict_candidates(m, candidates, batch_size=100)
    pd.testing.assert_frame_equal(a, b)
    assert a.columns.is_unique and a.input_valid.tolist() == ["raw"] * 3
    assert "input_input_qed" in a and "input_predicted_other" in a
    assert np.isnan(a.loc[1, "qed"]) and a.loc[0, "qed"] < 1
    assert predict_candidates(m, pd.DataFrame(columns=["smiles"])).empty
    assert "qed" in druglikeness_table(["bad", "bad"])
    assert not passes_filters(druglikeness_table(["bad"]).iloc[0].to_dict())


def test_infinite_targets_and_class_aggregation():
    assert np.isnan(to_pactivity([np.inf, -np.inf, np.nan])).all()
    df = pd.DataFrame({"smiles": ["CCO", "CCN", "CCC"], "pIC50": [1, np.inf, "oops"]})
    assert clean_dataset(df).pIC50.tolist() == [1]
    with pytest.raises(ValueError, match="conflicting"):
        clean_dataset(
            pd.DataFrame({"smiles": ["CCO", "OCC"], "pIC50": [0, 1]}), classification=True
        )
    with pytest.raises(ValueError, match="integer"):
        clean_dataset(pd.DataFrame({"smiles": ["CCO"], "pIC50": [0.5]}), classification=True)


@pytest.mark.parametrize(
    "raw",
    [
        [],
        {"model": "ridge"},
        {"data": None},
        {"model": {"cv_folds": True}},
        {"model": {"params": []}},
        {"model": {"test_size": float("nan")}},
        {"predict": {"smiles": "CCO"}},
        {"predict": {"druglikeness": "false"}},
        {"predict": {"batch_size": 0}},
        {"docking": {"size": [20, 0, 20]}},
        {"docking": {"enabled": True}},
        {"docking": {"center": [0, float("inf"), 0]}},
        {"docking": {"top_n": -1}},
        {"model": {"save_path": "same"}, "output": "same"},
        {"data": {"path": "same.csv"}, "output": "same.csv"},
    ],
)
def test_strict_config_validation(raw):
    with pytest.raises(ValueError):
        PipelineConfig.from_dict(raw)
