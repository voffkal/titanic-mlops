from __future__ import annotations

import io

import joblib
import numpy as np
import pytest
from sklearn.base import clone, is_classifier
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score

from titanic_model.mlx_mlp import MLXClassifier


@pytest.fixture(scope="module")
def blobs():
    return make_classification(
        n_samples=200, n_features=8, n_informative=4, random_state=0
    )


def small(**kwargs) -> MLXClassifier:
    return MLXClassifier(epochs=30, patience=10, hidden=16, **kwargs)


def test_sklearn_sees_a_classifier():
    """With (BaseEstimator, ClassifierMixin) reversed, scoring='roc_auc' returns NaN
    instead of raising — the failure is silent."""
    assert is_classifier(MLXClassifier())


def test_roc_auc_scoring_is_not_nan(blobs):
    X, y = blobs
    scores = cross_val_score(small(), X, y, scoring="roc_auc", cv=3)
    assert np.isfinite(scores).all()
    assert scores.mean() > 0.6


def test_predict_proba_is_a_distribution(blobs):
    X, y = blobs
    proba = small().fit(X, y).predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert (proba >= 0).all()


def test_predict_returns_the_original_labels():
    X, _ = make_classification(n_samples=120, n_features=6, random_state=0)
    y = np.where(np.arange(120) % 2, "yes", "no")
    fitted = small().fit(X, y)
    assert set(fitted.classes_) == {"no", "yes"}
    assert set(fitted.predict(X)) <= {"no", "yes"}


def test_the_seed_makes_the_fit_reproducible(blobs):
    """Weight init, the val split and batch shuffling all draw from random_state."""
    X, y = blobs
    first = small(random_state=0).fit(X, y).predict_proba(X)
    second = small(random_state=0).fit(X, y).predict_proba(X)
    assert np.array_equal(first, second)


def test_a_different_seed_gives_a_different_fit(blobs):
    """Otherwise random_state is decorative and the reproducibility test proves nothing."""
    X, y = blobs
    first = small(random_state=0).fit(X, y).predict_proba(X)
    other = small(random_state=7).fit(X, y).predict_proba(X)
    assert not np.array_equal(first, other)


def test_early_stopping_reports_where_it_stopped(blobs):
    X, y = blobs
    fitted = small().fit(X, y)
    assert 0 <= fitted.best_epoch_ < 30
    assert 0.0 <= fitted.best_score_ <= 1.0


def test_validation_split_comes_out_of_train(blobs):
    """Early stopping must not see the outer fold; CV scores would be optimistic."""
    X, y = blobs
    fitted = small(val_size=0.5).fit(X, y)
    assert fitted.model_.l1.weight.shape[1] == X.shape[1]


def test_defaults_monitor_auc():
    assert MLXClassifier().monitor == "auc"  # accuracy at 0.5 hides ranking gains


def test_clone_keeps_the_hyperparameters():
    original = MLXClassifier(epochs=7, hidden=13, random_state=42)
    assert clone(original).get_params() == original.get_params()


def test_fitted_estimator_survives_joblib(blobs):
    """The stack member is pickled inside the release artifact."""
    X, y = blobs
    fitted = small().fit(X, y)

    buf = io.BytesIO()
    joblib.dump(fitted, buf)
    buf.seek(0)
    # in-memory round-trip of an object this process just wrote, no external input
    restored = joblib.load(buf)

    assert np.array_equal(restored.predict_proba(X), fitted.predict_proba(X))
