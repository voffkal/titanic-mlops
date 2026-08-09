from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.schemas import MultiplePassengers, Passenger
from app.tests.conftest import SAMPLE
from titanic_model.predict import REQUIRED_COLUMNS, load_schema


def test_fields_match_the_artifact_schema(trained_dir):
    """The wire contract is hand-written; this is what keeps it from drifting."""
    assert set(Passenger.model_fields) == set(load_schema(trained_dir))


def test_required_fields_match_the_predictor(trained_dir):
    required = {name for name, f in Passenger.model_fields.items() if f.is_required()}
    assert required == set(REQUIRED_COLUMNS)


def test_aliases_are_the_kaggle_names():
    aliases = {f.alias for f in Passenger.model_fields.values()}
    assert {"Pclass", "Name", "Sex", "SibSp", "Parch", "Age", "Ticket", "Fare"} <= aliases


def test_both_spellings_produce_the_same_record():
    by_alias = Passenger.model_validate(SAMPLE)
    by_name = Passenger.model_validate({k.lower(): v for k, v in SAMPLE.items()})
    assert by_alias.model_dump() == by_name.model_dump()


def test_dump_uses_the_column_names_the_model_expects(trained_dir):
    record = Passenger.model_validate(SAMPLE).model_dump()
    assert set(record) == set(load_schema(trained_dir))


def test_optional_fields_default_to_none():
    record = Passenger.model_validate(
        {"Pclass": 1, "Name": "A, Mrs. B", "Sex": "female", "SibSp": 0, "Parch": 0}
    ).model_dump()
    assert record["age"] is None and record["cabin"] is None


@pytest.mark.parametrize(
    "change",
    [{"Pclass": 0}, {"Sex": "f"}, {"SibSp": -1}, {"Age": 0}, {"Name": ""}],
)
def test_domain_violations_are_rejected(change):
    with pytest.raises(ValidationError):
        Passenger.model_validate({**SAMPLE, **change})


def test_batch_must_not_be_empty():
    with pytest.raises(ValidationError):
        MultiplePassengers(inputs=[])
