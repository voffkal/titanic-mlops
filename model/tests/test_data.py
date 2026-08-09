from __future__ import annotations

import pandas as pd
import pytest

from titanic_model import data as data_mod
from titanic_model.data import (
    DEFAULT_RAW_DIR,
    FEATURE_COLUMNS,
    ID_COLUMN,
    TARGET,
    load_test,
    load_train,
    normalize_columns,
    write_submission,
)

from conftest import requires_kaggle_data, write_raw_dir


def test_normalize_columns_lowercases_names_only():
    df = pd.DataFrame({"PassengerId": [1], "SibSp": [0], "Name": ["Smith, Mr. John"]})
    out = normalize_columns(df)
    assert list(out.columns) == ["passengerid", "sibsp", "name"]
    assert out.loc[0, "name"] == "Smith, Mr. John"


def test_load_train_returns_features_only(raw_dir):
    X, y = load_train(raw_dir)
    assert list(X.columns) == FEATURE_COLUMNS
    assert TARGET not in X.columns and ID_COLUMN not in X.columns  # no leakage into X
    assert len(X) == len(y) == 891
    assert y.dtype.kind == "i"


def test_load_test_returns_ids(raw_dir):
    X, ids = load_test(raw_dir)
    assert list(X.columns) == FEATURE_COLUMNS
    assert len(X) == len(ids) == 418
    assert ids.is_unique


def test_missing_file_names_the_download(tmp_path):
    with pytest.raises(FileNotFoundError, match="kaggle.com/competitions/titanic"):
        load_train(tmp_path)


def test_wrong_row_count_rejected(tmp_path, raw_dir):
    """A hand-made or re-split file is the failure this guard exists for."""
    df = pd.read_csv(raw_dir / "train.csv").head(500)
    (tmp_path / "train.csv").write_text(df.to_csv(index=False))
    with pytest.raises(ValueError, match="expected 891 rows, got 500"):
        load_train(tmp_path)


def test_missing_column_rejected(tmp_path, raw_dir):
    df = pd.read_csv(raw_dir / "train.csv").drop(columns=["Cabin"])
    (tmp_path / "train.csv").write_text(df.to_csv(index=False))
    with pytest.raises(ValueError, match=r"missing columns \['cabin'\]"):
        load_train(tmp_path)


def test_raw_dir_follows_the_env_var(tmp_path, monkeypatch):
    import importlib

    monkeypatch.setenv("TITANIC_RAW_DIR", str(tmp_path))
    reloaded = importlib.reload(data_mod)
    assert tmp_path == reloaded.DEFAULT_RAW_DIR
    monkeypatch.delenv("TITANIC_RAW_DIR")
    importlib.reload(data_mod)  # other tests import the module-level default


def test_submission_matches_the_kaggle_format(tmp_path):
    path = write_submission(pd.Series(range(892, 892 + 418)), [1.0, 0.0] * 209, tmp_path / "s.csv")
    out = pd.read_csv(path)
    assert list(out.columns) == ["PassengerId", "Survived"]  # Kaggle rejects other headers
    assert out["Survived"].dtype.kind == "i"  # floats would be rejected too
    assert len(out) == 418


def test_submission_rejects_wrong_length(tmp_path):
    with pytest.raises(ValueError, match="418 rows, got 3"):
        write_submission(pd.Series([1, 2, 3]), [0, 1, 0], tmp_path / "s.csv")


def test_submission_ignores_index_alignment(tmp_path):
    """ids and predictions come from different objects; only their order matters."""
    ids = pd.Series(range(892, 892 + 418), index=range(5000, 5418))
    preds = pd.Series([1] * 418)
    out = pd.read_csv(write_submission(ids, preds, tmp_path / "s.csv"))
    assert out["PassengerId"].tolist() == list(range(892, 892 + 418))
    assert out["Survived"].sum() == 418


@requires_kaggle_data
def test_official_files_load():
    X, y = load_train(DEFAULT_RAW_DIR)
    assert len(X) == 891
    assert y.mean() == pytest.approx(0.3838, abs=1e-4)  # 342 survivors
    assert X["age"].isna().sum() == 177
    assert X["embarked"].isna().sum() == 2


@requires_kaggle_data
def test_official_test_file_loads():
    X, ids = load_test(DEFAULT_RAW_DIR)
    assert len(X) == 418
    assert ids.iloc[0] == 892
    assert X["fare"].isna().sum() == 1


def test_synthetic_raw_dir_is_kaggle_shaped(tmp_path):
    """Guards the fixture itself: if it drifts, every other test lies."""
    path = write_raw_dir(tmp_path / "raw")
    assert "PassengerId" in pd.read_csv(path / "train.csv").columns
    assert len(load_train(path)[0]) == 891
    assert len(load_test(path)[0]) == 418
