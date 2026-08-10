"""Synthetic Titanic-shaped data.

The Kaggle files are not in the repository and cannot be redistributed, so the
suite generates its own frame with the same schema, the same missingness and a
sex-driven target. Everything the pipeline is picky about is present: titles
inside ``name``, enough distinct fares for ``qcut(q=4)``, shared tickets,
missing age/fare/cabin/embarked, both classes in every fold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from titanic_model.data import DEFAULT_RAW_DIR, FEATURE_COLUMNS, ID_COLUMN, TARGET

MALE_TITLES = ["Mr", "Master", "Dr", "Rev"]
FEMALE_TITLES = ["Mrs", "Miss", "the Countess"]


def make_frame(n: int = 120, seed: int = 0, signal: bool = True) -> tuple[pd.DataFrame, pd.Series]:
    """Titanic-shaped features and a target. ``signal=False`` gives coin-flip labels."""
    rng = np.random.default_rng(seed)

    sex = rng.choice(["male", "female"], size=n)
    title = np.where(
        sex == "male",
        rng.choice(MALE_TITLES, size=n, p=[0.7, 0.1, 0.1, 0.1]),
        rng.choice(FEMALE_TITLES, size=n, p=[0.45, 0.5, 0.05]),
    )

    age = rng.uniform(1, 70, size=n).round(1)
    age[rng.random(n) < 0.20] = np.nan

    fare = np.abs(rng.lognormal(3, 0.8, size=n)).round(2)
    fare[rng.random(n) < 0.03] = np.nan

    embarked = rng.choice(["S", "C", "Q"], size=n, p=[0.7, 0.2, 0.1]).astype(object)
    embarked[rng.random(n) < 0.02] = None

    pclass = rng.choice([1, 2, 3], size=n, p=[0.25, 0.25, 0.5])

    # Families share a surname and usually a ticket, and share a fate. Without
    # this the frame gives every passenger a unique surname, GroupSurvival is
    # degenerate, and nothing exercises it.
    family = rng.integers(0, max(2, n // 3), size=n)
    ticket = np.array([f"T{f}" for f in family])
    apart = rng.random(n) < 0.15  # some relatives booked separately
    ticket[apart] = np.array([f"T{f}" for f in rng.integers(0, max(2, n // 3), size=apart.sum())])

    X = pd.DataFrame(
        {
            "pclass": pclass,
            "name": [
                f"Fam{f}, {t}. Given{i}"
                for i, (f, t) in enumerate(zip(family, title, strict=True))
            ],
            "sex": sex,
            "age": age,
            "sibsp": rng.integers(0, 4, size=n),
            "parch": rng.integers(0, 3, size=n),
            "ticket": ticket,
            "fare": fare,
            "cabin": np.where(rng.random(n) < 0.25, "C" + rng.integers(1, 120, n).astype(str), None),
            "embarked": embarked,
        }
    )

    if signal:
        # 0.4 keeps the synthetic problem above the 0.78 quality gate; a stronger
        # family effect makes it unlearnable from the thin per-fold group statistics
        family_effect = rng.normal(0, 0.4, size=family.max() + 1)[family]
        logit = 1.8 * (sex == "female") - 0.5 * pclass + family_effect + rng.normal(0, 0.7, n)
        labels = (logit > np.median(logit)).astype(int)
    else:
        labels = rng.integers(0, 2, size=n)

    return X, pd.Series(labels, name=TARGET)


def write_raw_dir(path, seed: int = 0, signal: bool = True):
    """Kaggle-shaped train.csv/test.csv (891/418 rows, CamelCase) under ``path``."""
    path.mkdir(parents=True, exist_ok=True)

    X, y = make_frame(891, seed=seed, signal=signal)
    train = X.copy()
    train[TARGET] = y
    train[ID_COLUMN] = range(1, 892)

    X_test, _ = make_frame(418, seed=seed + 100, signal=signal)
    test = X_test.copy()
    test[ID_COLUMN] = range(892, 892 + 418)

    # the loader lowercases Kaggle's CamelCase; write it the way Kaggle does
    camel = {c: c.title().replace("Passengerid", "PassengerId") for c in train.columns}
    train.rename(columns=camel).to_csv(path / "train.csv", index=False)
    test.rename(columns=camel).to_csv(path / "test.csv", index=False)
    return path


@pytest.fixture(scope="session")
def frame() -> tuple[pd.DataFrame, pd.Series]:
    return make_frame(120)


@pytest.fixture(scope="session")
def raw_dir(tmp_path_factory):
    return write_raw_dir(tmp_path_factory.mktemp("raw"))


requires_kaggle_data = pytest.mark.skipif(
    not (DEFAULT_RAW_DIR / "train.csv").exists(),
    reason=f"no Kaggle train.csv under {DEFAULT_RAW_DIR} (set TITANIC_RAW_DIR)",
)

__all__ = ["FEATURE_COLUMNS", "make_frame", "requires_kaggle_data", "write_raw_dir"]
