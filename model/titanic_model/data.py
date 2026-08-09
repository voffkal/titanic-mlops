from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

__all__ = [
    "DEFAULT_RAW_DIR",
    "TARGET",
    "ID_COLUMN",
    "FEATURE_COLUMNS",
    "load_train",
    "load_test",
    "normalize_columns",
    "write_submission",
]

DEFAULT_RAW_DIR = Path(
    os.environ.get(
        "TITANIC_RAW_DIR",
        Path(__file__).resolve().parents[2] / "workbook" / "data" / "raw",
    )
)

TARGET = "survived"
ID_COLUMN = "passengerid"

FEATURE_COLUMNS = [
    "pclass",
    "name",
    "sex",
    "age",
    "sibsp",
    "parch",
    "ticket",
    "fare",
    "cabin",
    "embarked",
]


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={c: c.lower() for c in df.columns})


def _read(path: Path, expected_rows: int) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"missing {path}. Download the competition archive "
            f"(kaggle.com/competitions/titanic/data) and unpack it into {path.parent}"
        )

    df = normalize_columns(pd.read_csv(path))

    # row count catches a hand-made file swapped in for the official one
    if len(df) != expected_rows:
        raise ValueError(
            f"{path.name}: expected {expected_rows} rows, got {len(df)}. "
            f"This does not look like the official Kaggle file."
        )

    missing = set(FEATURE_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"{path.name}: missing columns {sorted(missing)}")

    return df


def load_train(raw_dir: Path | str = DEFAULT_RAW_DIR) -> tuple[pd.DataFrame, pd.Series]:
    df = _read(Path(raw_dir) / "train.csv", expected_rows=891)
    y = df[TARGET].astype(int)
    X = df[FEATURE_COLUMNS].copy()
    return X, y


def load_test(raw_dir: Path | str = DEFAULT_RAW_DIR) -> tuple[pd.DataFrame, pd.Series]:
    df = _read(Path(raw_dir) / "test.csv", expected_rows=418)
    return df[FEATURE_COLUMNS].copy(), df[ID_COLUMN]


def write_submission(
    passenger_ids: pd.Series,
    predictions,
    path: Path | str,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    submission = pd.DataFrame(
        {
            "PassengerId": pd.Series(passenger_ids).to_numpy(),
            "Survived": pd.Series(predictions).astype(int).to_numpy(),
        }
    )

    if len(submission) != 418:
        raise ValueError(f"submission must contain 418 rows, got {len(submission)}")

    submission.to_csv(path, index=False)
    return path
