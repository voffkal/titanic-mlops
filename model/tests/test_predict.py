from __future__ import annotations

import pandas as pd
import pytest

from titanic_model import __version__
from titanic_model.predict import (
    REQUIRED_COLUMNS,
    _load,
    coerce_to_schema,
    load_schema,
    make_prediction,
)
from titanic_model.train import train

from conftest import make_frame, write_raw_dir

SCHEMA = {
    "pclass": "int64",
    "name": "str",
    "sex": "str",
    "age": "float64",
    "sibsp": "int64",
    "parch": "int64",
    "ticket": "str",
    "fare": "float64",
    "cabin": "str",
    "embarked": "str",
}


@pytest.fixture(scope="module")
def artifacts(tmp_path_factory):
    _load.cache_clear()  # lru_cache is keyed by directory; other modules train too
    out = tmp_path_factory.mktemp("artifacts")
    raw = write_raw_dir(tmp_path_factory.mktemp("raw"))
    train(config="logreg", artifacts_dir=out, raw_dir=raw, cv_repeats=1)
    return out


@pytest.fixture
def passenger():
    X, _ = make_frame(5)
    return X.iloc[0].to_dict()


class TestCoercion:
    def test_forces_the_train_dtypes(self):
        raw = pd.DataFrame([{**{c: None for c in SCHEMA}, "pclass": "3", "sibsp": 0, "parch": 0,
                             "name": "A, Mr. B", "sex": "male"}])
        out = coerce_to_schema(raw, SCHEMA)
        assert out["pclass"].dtype == "float64"  # numeric strings are parsed
        assert out["pclass"].iloc[0] == 3.0
        assert out["cabin"].dtype == "object"

    def test_text_column_stays_text_when_wholly_missing(self):
        """A one-row batch without a cabin would otherwise be inferred as float64."""
        raw = pd.DataFrame([{**{c: None for c in SCHEMA}, "pclass": 1, "sibsp": 0, "parch": 0,
                             "name": "A, Mrs. B", "sex": "female"}])
        out = coerce_to_schema(raw, SCHEMA)
        assert out["cabin"].dtype == "object"
        assert out["cabin"].iloc[0] is None

    def test_column_order_follows_the_schema(self, passenger):
        shuffled = pd.DataFrame([dict(reversed(list(passenger.items())))])
        assert list(coerce_to_schema(shuffled, SCHEMA).columns) == list(SCHEMA)

    def test_missing_column_is_rejected(self, passenger):
        del passenger["fare"]
        with pytest.raises(ValueError, match=r"missing required columns: \['fare'\]"):
            coerce_to_schema(pd.DataFrame([passenger]), SCHEMA)

    @pytest.mark.parametrize("column", REQUIRED_COLUMNS)
    def test_unimputable_column_is_rejected(self, passenger, column):
        passenger[column] = None
        with pytest.raises(ValueError, match="cannot be imputed"):
            coerce_to_schema(pd.DataFrame([passenger]), SCHEMA)

    @pytest.mark.parametrize("column", ["age", "fare", "ticket", "cabin", "embarked"])
    def test_imputable_column_may_be_empty(self, passenger, column):
        passenger[column] = None
        assert coerce_to_schema(pd.DataFrame([passenger]), SCHEMA)[column].isna().all()

    def test_unparseable_number_becomes_nan_not_an_error(self, passenger):
        passenger["fare"] = "дорого"
        assert coerce_to_schema(pd.DataFrame([passenger]), SCHEMA)["fare"].isna().all()


class TestMakePrediction:
    def test_returns_labels_probabilities_and_version(self, artifacts, passenger):
        result = make_prediction([passenger], artifacts_dir=artifacts)
        assert result["version"] == __version__
        assert result["predictions"][0] in (0, 1)
        assert 0.0 <= result["probabilities"][0] <= 1.0

    def test_labels_are_the_probabilities_at_one_half(self, artifacts):
        X, _ = make_frame(20)
        result = make_prediction(X, artifacts_dir=artifacts)
        for label, p in zip(result["predictions"], result["probabilities"], strict=True):
            assert label == int(p >= 0.5)

    def test_batch_keeps_its_order(self, artifacts):
        X, _ = make_frame(10)
        batch = make_prediction(X, artifacts_dir=artifacts)["probabilities"]
        one_by_one = [
            make_prediction(X.iloc[[i]], artifacts_dir=artifacts)["probabilities"][0]
            for i in range(len(X))
        ]
        assert batch == pytest.approx(one_by_one, abs=1e-6)

    def test_accepts_a_dataframe_and_a_list_of_dicts(self, artifacts):
        X, _ = make_frame(3)
        assert (
            make_prediction(X, artifacts_dir=artifacts)["probabilities"]
            == make_prediction(X.to_dict("records"), artifacts_dir=artifacts)["probabilities"]
        )

    def test_empty_request_is_rejected(self, artifacts):
        with pytest.raises(ValueError, match="no passengers"):
            make_prediction([], artifacts_dir=artifacts)

    def test_json_serialisable_types(self, artifacts, passenger):
        result = make_prediction([passenger], artifacts_dir=artifacts)
        assert type(result["predictions"][0]) is int  # not np.int64
        assert type(result["probabilities"][0]) is float


def test_load_schema_matches_the_artifact(artifacts):
    assert load_schema(artifacts) == SCHEMA
