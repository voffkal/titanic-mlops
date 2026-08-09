from __future__ import annotations

import io

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from titanic_model.builders import build
from titanic_model.pipeline import POLY_COLUMNS, ZERO, make_ct, make_pipe


class TestZero:
    def test_zeroes_one_column_keeping_the_shape(self):
        out = ZERO.fit_transform(pd.DataFrame({"age": [10.0, 20.0, 30.0]}))
        assert out.shape == (3, 1)
        assert not out.any()

    def test_is_picklable(self):
        """Built from lambdas this passes fit and CV, then joblib.dump fails at the very end."""
        buf = io.BytesIO()
        joblib.dump(ZERO, buf)
        buf.seek(0)
        # in-memory round-trip of an object this process just wrote, no external input
        assert not joblib.load(buf).fit_transform(pd.DataFrame({"a": [1.0]})).any()

    def test_feature_name_marks_the_column(self):
        assert list(ZERO.get_feature_names_out(["age"])) == ["age_ZERO"]

    def test_survives_clone(self):
        assert clone(ZERO).fit_transform(np.zeros((2, 1))).shape == (2, 1)


class TestColumnTransformer:
    def test_every_slot_is_addressable(self):
        """The frozen configs are flat ct__<col>__<step> dicts; a renamed slot breaks them."""
        params = make_ct().get_params()
        for key in ("title__encode", "age__outlier", "fare_per_person__transform", "poly"):
            assert key in params

    def test_poly_slot_is_dropped_not_passed_through(self, frame):
        """passthrough here emits age/pclass/family_size/fare_per_person a second time."""
        X, y = frame
        ct = build("logreg").fit(X, y).named_steps["ct"]
        names = list(ct.get_feature_names_out())
        assert len(names) == len(set(names))
        for col in POLY_COLUMNS:
            assert names.count(col) <= 1

    def test_zero_columns_reach_the_matrix_but_not_the_model(self, frame):
        X, y = frame
        pipe = build("logreg").fit(X, y)

        ct_names = list(pipe.named_steps["ct"].get_feature_names_out())
        assert "pclass_ZERO" in ct_names  # logreg's config switches pclass off

        kept = pipe.named_steps["drop_constant"].get_feature_names_out()
        assert "pclass_ZERO" not in kept  # a constant column never reaches the estimator

    def test_remainder_drops_raw_text_columns(self, frame):
        X, y = frame
        names = build("logreg").fit(X, y).named_steps["ct"].get_feature_names_out()
        assert not {"name", "cabin", "ticket"} & set(names)


class TestPipelineOrder:
    def steps(self) -> list[str]:
        return [name for name, _ in make_pipe().steps]

    def test_expected_steps_present(self):
        assert self.steps() == [
            "missing_indicators",
            "fare_imputer",
            "embarked_imputer",
            "title",
            "age_feats",
            "fare_feats",
            "cabin_cat",
            "family_feats",
            "pclass_feats",
            "ct",
            "nan_after_ct",
            "drop_correlated",
            "drop_constant",
            "scaler",
            "sampler",
            "model",
        ]

    @pytest.mark.parametrize(
        ("earlier", "later", "why"),
        [
            ("missing_indicators", "fare_imputer", "the indicator must see the original NaN"),
            ("fare_imputer", "fare_feats", "quantile edges need a non-null fare"),
            ("title", "age_feats", "age is imputed by title"),
            ("fare_feats", "ct", "ct consumes fare_per_person"),
            ("ct", "scaler", "the scaler needs a numeric matrix"),
            ("scaler", "sampler", "SMOTE interpolates in the scaled space"),
        ],
    )
    def test_order_invariants(self, earlier, later, why):
        steps = self.steps()
        assert steps.index(earlier) < steps.index(later), why

    def test_sampler_is_the_last_step_before_the_model(self):
        """Resampling must not touch the validation fold; only imblearn guarantees that."""
        steps = self.steps()
        assert steps[-2:] == ["sampler", "model"]


def test_matrix_is_numeric_and_finite(frame):
    """nan_after_ct raises inside fit; this pins the contract at the model boundary."""
    X, y = frame
    pipe = build("logreg").fit(X, y)
    matrix = pipe[:-3].transform(X)  # up to and including drop_constant
    assert np.isfinite(np.asarray(matrix, dtype=float)).all()


def test_unseen_categories_do_not_produce_nan(frame):
    """Kaggle's test set holds decks and title/pclass pairs absent from train."""
    X, y = frame
    pipe = build("logreg").fit(X, y)

    novel = X.head(3).copy()
    novel["cabin"] = ["Z99", "Z100", None]
    novel["embarked"] = ["X", "S", None]
    novel["ticket"] = ["UNSEEN-1", "UNSEEN-2", "UNSEEN-3"]
    novel["name"] = ["Fam, Wizard. A", "Fam, Mr. B", "Fam, Dona. C"]

    proba = pipe.predict_proba(novel)
    assert np.isfinite(proba).all()
