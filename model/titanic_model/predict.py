"""Inference: load the artifact and predict against the frozen input schema."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import __version__
from .train import DEFAULT_ARTIFACTS_DIR

__all__ = ["REQUIRED_COLUMNS", "coerce_to_schema", "load_schema", "make_prediction"]

#: ``pclass``/``sex`` are encoded without an imputer and ``sibsp``/``parch`` add
#: up to ``family_size``, so a missing value there raises deeper in the pipeline.
#: ``name`` is a policy choice rather than a technical one: it survives, but the
#: title falls back to "Empty", a category absent from the training set, and the
#: probability moves materially. Refusing beats answering from out of distribution.
REQUIRED_COLUMNS = ("pclass", "name", "sex", "sibsp", "parch")


@lru_cache(maxsize=2)
def _load(artifacts_dir: str) -> tuple[object, dict[str, str]]:
    d = Path(artifacts_dir)
    # A joblib artifact is a pickle: load() executes code while deserialising.
    # Load ONLY our own artifacts — produced by this train.py or downloaded from
    # a release of this repository. Never a third-party .joblib.
    model = joblib.load(d / f"titanic_model-{__version__}.joblib")
    schema: dict[str, str] = json.loads((d / "input_schema.json").read_text())
    return model, schema


def load_schema(artifacts_dir: str | Path | None = None) -> dict[str, str]:
    return _load(str(artifacts_dir or DEFAULT_ARTIFACTS_DIR))[1]


def coerce_to_schema(df: pd.DataFrame, schema: dict[str, str]) -> pd.DataFrame:
    """Type frontier: the column set is mandatory, dtypes are forced to the train schema.

    A text column that is missing for every row in the batch would otherwise be
    inferred as float64 and break the ``.str`` accessors downstream.
    """
    missing = [c for c in schema if c not in df.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")

    out = pd.DataFrame(index=df.index)
    for col, dtype in schema.items():
        if dtype == "str":
            out[col] = df[col].astype("object").where(df[col].notna(), None)
        else:
            out[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")

    empty = [c for c in REQUIRED_COLUMNS if out[c].isna().any()]
    if empty:
        raise ValueError(f"columns that cannot be imputed are empty: {empty}")

    return out


def make_prediction(
    input_data: pd.DataFrame | list[dict], artifacts_dir: str | Path | None = None
) -> dict:
    """Survival predictions and P(survived) for a batch of passengers."""
    model, schema = _load(str(artifacts_dir or DEFAULT_ARTIFACTS_DIR))

    df = pd.DataFrame(input_data)
    if df.empty:
        raise ValueError("no passengers in the request")

    X = coerce_to_schema(df, schema)
    proba = model.predict_proba(X)[:, 1]

    return {
        "predictions": [int(p) for p in (proba >= 0.5).astype(int)],
        "probabilities": [float(p) for p in np.round(proba, 6)],
        "version": __version__,
    }
