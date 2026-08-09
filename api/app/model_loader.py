"""Fail-fast bundle loading: version skew and a smoke prediction at startup."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import sklearn

from app.schemas import Passenger
from titanic_model import __version__ as model_version
from titanic_model.predict import make_prediction

RESOURCES = Path(__file__).resolve().parent / "resources"


@dataclass(frozen=True)
class ModelBundle:
    manifest: dict
    sample: dict


def load_model_bundle(artifacts_dir: Path) -> ModelBundle:
    manifest = json.loads((artifacts_dir / "metrics.json").read_text())

    trained_with = manifest["libraries"]["scikit-learn"]
    if trained_with != sklearn.__version__:
        raise RuntimeError(
            f"version skew: the artifact was trained on sklearn {trained_with}, "
            f"runtime has {sklearn.__version__} — refusing to start rather than "
            f"unpickling into silently different estimators"
        )
    if manifest["model_version"] != model_version:
        raise RuntimeError(
            f"version skew: artifact {manifest['model_version']}, package {model_version}"
        )

    # the sample is stored under Kaggle names, so the smoke test also exercises
    # alias parsing — the same path a request takes
    sample = json.loads((RESOURCES / "sample_payload.json").read_text())
    record = Passenger.model_validate(sample).model_dump()
    probability = make_prediction([record], artifacts_dir=artifacts_dir)["probabilities"][0]
    # the sample is a first-class woman, whom the model should call a survivor
    if not 0.5 < probability < 1.0:
        raise RuntimeError(f"implausible smoke prediction: {probability}")

    return ModelBundle(manifest=manifest, sample=sample)
