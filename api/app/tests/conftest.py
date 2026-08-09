"""A real artifact trained on synthetic data, so the suite needs no Kaggle files."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from titanic_model.train import train

MODEL_TESTS = Path(__file__).resolve().parents[3] / "model" / "tests"
sys.path.insert(0, str(MODEL_TESTS))  # reuse the synthetic Kaggle-shaped generator

from conftest import write_raw_dir  # noqa: E402

SAMPLE = json.loads(
    (Path(__file__).resolve().parents[1] / "resources" / "sample_payload.json").read_text()
)


@pytest.fixture(scope="session")
def trained_dir(tmp_path_factory) -> Path:
    artifacts = tmp_path_factory.mktemp("artifacts")
    raw = write_raw_dir(tmp_path_factory.mktemp("raw"))
    train(config="logreg", artifacts_dir=artifacts, raw_dir=raw, cv_repeats=1)
    return artifacts


@pytest.fixture
def client(trained_dir):
    from app.config import Settings
    from app.main import create_app

    app = create_app(Settings(artifacts_dir=trained_dir))
    with TestClient(app) as c:  # the with-block runs lifespan, i.e. loads the model
        yield c
