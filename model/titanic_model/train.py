"""Train the release model and write artifacts.

CLI::

    python -m titanic_model.train
    python -m titanic_model.train --config logreg --submission
    python -m titanic_model.train --cv-repeats 1        # quick run

Writes into ``--artifacts-dir``: the pipeline as joblib, ``metrics.json``,
``input_schema.json`` and optionally ``submission.csv``. See README.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    cross_val_predict,
    cross_val_score,
)

from . import __version__
from .builders import MODELS, STACK_MEMBERS, build, build_stack
from .data import DEFAULT_RAW_DIR, load_test, load_train, write_submission

DEFAULT_ARTIFACTS_DIR = Path(__file__).resolve().parents[2] / "model_artifacts"

#: Quality gate: below this no artifacts are written. Reference point is the
#: sex rule (LB 0.76555).
MIN_CV_ACCURACY = 0.78


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _dataset_hash(X: pd.DataFrame, y: pd.Series) -> str:
    payload = pd.concat([X.reset_index(drop=True), y.reset_index(drop=True)], axis=1)
    return hashlib.sha256(
        pd.util.hash_pandas_object(payload, index=False).values.tobytes()
    ).hexdigest()[:16]


def _relative_source(raw_dir: Path) -> str:
    """Path relative to the project root, so the manifest carries no local paths."""
    root = Path(__file__).resolve().parents[2]
    try:
        return str(Path(raw_dir).resolve().relative_to(root))
    except ValueError:
        return Path(raw_dir).name


def _library_versions() -> dict[str, str]:
    import importlib.metadata as md

    names = [
        "scikit-learn",
        "pandas",
        "numpy",
        "scipy",
        "feature-engine",
        "category-encoders",
        "imbalanced-learn",
        "xgboost",
        "mlx",
        "joblib",
    ]
    out = {}
    for n in names:
        try:
            out[n] = md.version(n)
        except md.PackageNotFoundError:
            out[n] = "not installed"
    return out


def _classification_metrics(y_true, pred, proba=None) -> dict[str, Any]:
    m = {
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred)),
        "f1": float(f1_score(y_true, pred)),
        "confusion_matrix": confusion_matrix(y_true, pred).tolist(),
    }
    if proba is not None:
        m["roc_auc"] = float(roc_auc_score(y_true, proba))
    return m


def _sex_rule_baseline(X: pd.DataFrame, y: pd.Series) -> float:
    return float(accuracy_score(y, (X["sex"] == "female").astype(int)))


def train(
    config: str = "stack",
    artifacts_dir: Path = DEFAULT_ARTIFACTS_DIR,
    raw_dir: Path = DEFAULT_RAW_DIR,
    cv_repeats: int = 5,
    make_submission: bool = False,
) -> dict[str, Any]:
    started = time.perf_counter()
    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    X, y = load_train(raw_dir)
    estimator = build_stack() if config == "stack" else build(config)

    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=cv_repeats, random_state=0)
    n_jobs = 1 if config == "stack" else -1  # loky forks processes, which clashes with the MLX Metal context

    print(f"[1/4] CV 5x{cv_repeats} ({config})...", flush=True)
    cv_acc = cross_val_score(estimator, X, y, scoring="accuracy", cv=cv, n_jobs=n_jobs)
    cv_auc = cross_val_score(estimator, X, y, scoring="roc_auc", cv=cv, n_jobs=n_jobs)

    print("[2/4] out-of-fold predictions...", flush=True)
    oof_proba = cross_val_predict(
        estimator, X, y, cv=5, method="predict_proba", n_jobs=n_jobs
    )[:, 1]

    print("[3/4] fitting on all data...", flush=True)
    estimator.fit(X, y)
    in_sample_proba = estimator.predict_proba(X)[:, 1]

    baseline = _sex_rule_baseline(X, y)
    manifest: dict[str, Any] = {
        "model_version": __version__,
        "config": config,
        "stack_members": list(STACK_MEMBERS) if config == "stack" else None,
        "metrics": {
            "cv": {
                "protocol": f"RepeatedStratifiedKFold(n_splits=5, n_repeats={cv_repeats}, random_state=0)",
                "accuracy_mean": float(cv_acc.mean()),
                "accuracy_std": float(cv_acc.std()),
                "roc_auc_mean": float(cv_auc.mean()),
                "roc_auc_std": float(cv_auc.std()),
                "n_folds": int(len(cv_acc)),
            },
            "oof": _classification_metrics(y, (oof_proba >= 0.5).astype(int), oof_proba),
            # in-sample is only an overfitting check (its gap against CV)
            "in_sample": _classification_metrics(
                y, estimator.predict(X), in_sample_proba
            ),
        },
        "baseline": {
            "rule": "all women survive (public Kaggle baseline, LB 0.76555)",
            "train_accuracy": baseline,
            "cv_lift": float(cv_acc.mean() - baseline),
        },
        "dataset": {
            "hash": _dataset_hash(X, y),
            "n_rows": int(len(X)),
            "n_features_raw": int(X.shape[1]),
            "positive_rate": float(y.mean()),
            "source": _relative_source(raw_dir),
        },
        "git_commit": _git_commit(),
        "trained_at": datetime.now(UTC).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "libraries": _library_versions(),
    }

    gap = manifest["metrics"]["in_sample"]["accuracy"] - manifest["metrics"]["cv"]["accuracy_mean"]
    manifest["metrics"]["train_cv_gap"] = float(gap)

    if cv_acc.mean() < MIN_CV_ACCURACY:
        raise RuntimeError(
            f"quality gate failed: CV accuracy {cv_acc.mean():.4f} < {MIN_CV_ACCURACY}. "
            f"No artifacts written."
        )

    print("[4/4] writing artifacts...", flush=True)
    model_path = artifacts_dir / f"titanic_model-{__version__}.joblib"
    joblib.dump(estimator, model_path)
    (artifacts_dir / "metrics.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (artifacts_dir / "input_schema.json").write_text(
        json.dumps({c: str(t) for c, t in X.dtypes.items()}, indent=2), encoding="utf-8"
    )

    if make_submission:
        X_test, ids = load_test(raw_dir)
        path = write_submission(ids, estimator.predict(X_test), artifacts_dir / "submission.csv")
        print(f"      submission: {path}")

    manifest["elapsed_sec"] = round(time.perf_counter() - started, 1)
    print(
        f"\ndone in {manifest['elapsed_sec']}s\n"
        f"  CV accuracy {cv_acc.mean():.4f} ± {cv_acc.std():.4f}"
        f"   (sex-rule baseline {baseline:.4f}, lift {cv_acc.mean() - baseline:+.4f})\n"
        f"  CV roc_auc  {cv_auc.mean():.4f} ± {cv_auc.std():.4f}\n"
        f"  train-CV gap {gap:+.4f}\n"
        f"  artifact: {model_path} ({model_path.stat().st_size / 1024:.0f} KB)"
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--config",
        default="stack",
        choices=["stack", *sorted(MODELS)],
        help="what to train: stack (default) or a single model",
    )
    p.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR)
    p.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    p.add_argument(
        "--cv-repeats", type=int, default=5, help="cross-validation repeats; 1 for a quick run"
    )
    p.add_argument("--submission", action="store_true", help="write submission.csv for the leaderboard")
    a = p.parse_args(argv)

    train(
        config=a.config,
        artifacts_dir=a.artifacts_dir,
        raw_dir=a.raw_dir,
        cv_repeats=a.cv_repeats,
        make_submission=a.submission,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
