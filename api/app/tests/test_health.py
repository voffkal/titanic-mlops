from __future__ import annotations

import json
import shutil

import pytest
import sklearn
from fastapi.testclient import TestClient

from titanic_model import __version__ as model_version


def test_health_reports_versions_and_quality(client):
    body = client.get("/api/v1/health").json()
    assert body["status"] == "ok"
    assert body["model_version"] == model_version
    assert body["sklearn_version"] == sklearn.__version__
    assert 0.0 < body["cv_accuracy"] <= 1.0


def test_root_redirects_to_docs(client):
    r = client.get("/", follow_redirects=False)
    assert r.status_code in (302, 307)
    assert r.headers["location"] == "/docs"


def test_openapi_is_served_under_the_prefix(client):
    assert client.get("/api/v1/openapi.json").status_code == 200


def doctored(trained_dir, tmp_path, **changes):
    copy = tmp_path / "artifacts"
    shutil.copytree(trained_dir, copy)
    manifest = json.loads((copy / "metrics.json").read_text())
    manifest.update(changes)
    (copy / "metrics.json").write_text(json.dumps(manifest))
    return copy


def start(artifacts_dir):
    from app.config import Settings
    from app.main import create_app

    return TestClient(create_app(Settings(artifacts_dir=artifacts_dir)))


def test_sklearn_skew_refuses_to_start(trained_dir, tmp_path):
    """Unpickling into a different sklearn gives silently different estimators."""
    libraries = json.loads((trained_dir / "metrics.json").read_text())["libraries"]
    copy = doctored(trained_dir, tmp_path, libraries={**libraries, "scikit-learn": "0.0.0"})

    with pytest.raises(RuntimeError, match="sklearn"), start(copy):
        pass


def test_model_version_skew_refuses_to_start(trained_dir, tmp_path):
    copy = doctored(trained_dir, tmp_path, model_version="9.9.9")

    with pytest.raises(RuntimeError, match="artifact 9.9.9"), start(copy):
        pass


def test_missing_artifact_refuses_to_start(tmp_path):
    with pytest.raises(FileNotFoundError), start(tmp_path):
        pass
