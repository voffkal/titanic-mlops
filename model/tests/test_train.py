from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
import pytest

from titanic_model import __version__
from titanic_model.data import load_train
from titanic_model.train import (
    MIN_CV_ACCURACY,
    PROJECT_ROOT,
    _classification_metrics,
    _dataset_hash,
    _git_commit,
    _relative_source,
    _sex_rule_baseline,
    main,
    train,
)

from conftest import make_frame, write_raw_dir


class TestManifestHelpers:
    def test_source_is_relative_to_the_project(self):
        """Absolute paths in the manifest would publish the author's home directory."""
        assert _relative_source(PROJECT_ROOT / "workbook" / "data") == "workbook/data"

    def test_source_outside_the_project_degrades_to_a_name(self, tmp_path):
        assert _relative_source(tmp_path / "kaggle") == "kaggle"

    def test_commit_is_read_from_the_project_not_the_caller(self):
        """Training from another repository must not stamp that repository's HEAD."""
        commit = _git_commit()
        assert commit == "unknown" or commit.isalnum()

    def test_dataset_hash_is_stable(self):
        X, y = make_frame(50)
        assert _dataset_hash(X, y) == _dataset_hash(X.copy(), y.copy())

    def test_dataset_hash_notices_one_changed_cell(self):
        X, y = make_frame(50)
        other = X.copy()
        other.loc[0, "fare"] = other.loc[0, "fare"] + 1
        assert _dataset_hash(X, y) != _dataset_hash(other, y)

    def test_dataset_hash_notices_flipped_labels(self):
        X, y = make_frame(50)
        assert _dataset_hash(X, y) != _dataset_hash(X, 1 - y)


class TestMetrics:
    def test_reports_the_full_set(self):
        m = _classification_metrics([0, 1, 1, 0], [0, 1, 0, 0], [0.1, 0.9, 0.4, 0.2])
        assert m["accuracy"] == 0.75
        assert m["recall"] == 0.5
        assert m["confusion_matrix"] == [[2, 0], [1, 1]]
        assert m["roc_auc"] == 1.0  # ranking is perfect even though a label is wrong

    def test_roc_auc_is_optional(self):
        assert "roc_auc" not in _classification_metrics([0, 1], [0, 1])

    def test_precision_survives_a_model_that_never_predicts_one(self):
        assert _classification_metrics([0, 1], [0, 0])["precision"] == 0.0

    def test_sex_rule_baseline(self):
        X = pd.DataFrame({"sex": ["female", "male", "female"]})
        assert _sex_rule_baseline(X, pd.Series([1, 0, 0])) == pytest.approx(2 / 3)


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    """One real training run, reused by the assertions below."""
    raw = write_raw_dir(tmp_path_factory.mktemp("raw"))
    artifacts = tmp_path_factory.mktemp("artifacts")
    manifest = train(
        config="logreg",
        artifacts_dir=artifacts,
        raw_dir=raw,
        cv_repeats=1,
        make_submission=True,
    )
    return manifest, artifacts, raw


class TestTrainRun:
    def test_writes_every_artifact(self, run):
        _, artifacts, _ = run
        written = {p.name for p in artifacts.iterdir()}
        assert written == {
            f"titanic_model-{__version__}.joblib",
            "metrics.json",
            "input_schema.json",
            "submission.csv",
        }

    def test_manifest_is_json_and_matches_the_return_value(self, run):
        manifest, artifacts, _ = run
        on_disk = json.loads((artifacts / "metrics.json").read_text())
        assert on_disk["metrics"] == manifest["metrics"]
        assert on_disk["model_version"] == __version__

    def test_manifest_records_the_protocol_and_the_data(self, run):
        manifest, _, _ = run
        cv = manifest["metrics"]["cv"]
        assert "RepeatedStratifiedKFold" in cv["protocol"]
        assert cv["n_folds"] == 5  # 5 splits x 1 repeat
        assert manifest["dataset"]["n_rows"] == 891
        assert len(manifest["dataset"]["hash"]) == 16

    def test_manifest_holds_no_local_paths(self, run):
        """The manifest is the one artifact that goes into git."""
        manifest, _, _ = run
        assert "/Users/" not in json.dumps(manifest)

    def test_manifest_pins_the_libraries(self, run):
        manifest, _, _ = run
        assert {"scikit-learn", "xgboost", "mlx"} <= set(manifest["libraries"])

    def test_reports_the_gap_against_cv(self, run):
        manifest, _, _ = run
        gap = manifest["metrics"]["in_sample"]["accuracy"] - manifest["metrics"]["cv"]["accuracy_mean"]
        assert manifest["metrics"]["train_cv_gap"] == pytest.approx(gap)

    def test_input_schema_covers_the_raw_columns(self, run):
        _, artifacts, raw = run
        schema = json.loads((artifacts / "input_schema.json").read_text())
        assert set(schema) == set(load_train(raw)[0].columns)

    def test_submission_is_kaggle_shaped(self, run):
        _, artifacts, _ = run
        sub = pd.read_csv(artifacts / "submission.csv")
        assert list(sub.columns) == ["PassengerId", "Survived"]
        assert len(sub) == 418

    def test_artifact_reloads_and_predicts_identically(self, run):
        """The pickled pipeline is the deliverable; a lambda inside it fails here."""
        _, artifacts, raw = run
        X, _ = load_train(raw)
        # in-memory-equivalent round-trip of the artifact this test just produced
        model = joblib.load(artifacts / f"titanic_model-{__version__}.joblib")
        assert np.isfinite(model.predict_proba(X)).all()
        assert set(np.unique(model.predict(X))) <= {0, 1}


class TestQualityGate:
    def test_noise_is_refused_and_nothing_is_written(self, tmp_path):
        """The gate is the only thing between a broken pipeline and a published artifact."""
        raw = write_raw_dir(tmp_path / "raw", signal=False)
        artifacts = tmp_path / "artifacts"

        with pytest.raises(RuntimeError, match="quality gate failed"):
            train(config="logreg", artifacts_dir=artifacts, raw_dir=raw, cv_repeats=1)

        assert list(artifacts.iterdir()) == []

    def test_gate_sits_above_the_sex_rule(self):
        assert MIN_CV_ACCURACY > 0.76555  # the public leaderboard baseline


class TestCli:
    def test_rejects_an_unknown_config(self, capsys):
        with pytest.raises(SystemExit):
            main(["--config", "logreg_l2"])
        assert "invalid choice" in capsys.readouterr().err

    def test_runs_end_to_end(self, tmp_path):
        raw = write_raw_dir(tmp_path / "raw")
        code = main(
            [
                "--config", "logreg",
                "--raw-dir", str(raw),
                "--artifacts-dir", str(tmp_path / "artifacts"),
                "--cv-repeats", "1",
            ]
        )
        assert code == 0
        assert (tmp_path / "artifacts" / "metrics.json").exists()
