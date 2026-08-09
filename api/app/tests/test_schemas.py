from __future__ import annotations

import pandas as pd
import pytest
from pydantic import ValidationError

from app.schemas import MultiplePassengers, Passenger
from app.tests.conftest import SAMPLE, requires_kaggle_data
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


@pytest.mark.parametrize("name", ["string", "John Smith", "Smith, Mr Owen", "Mr. Smith"])
def test_name_without_a_title_is_rejected(name):
    """Requiring the field is pointless if a name with no title sails through."""
    with pytest.raises(ValidationError, match="Surname, Title"):
        Passenger.model_validate({**SAMPLE, "Name": name})


@pytest.mark.parametrize(
    "name",
    [
        "Braund, Mr. Owen Harris",
        "Rothes, the Countess. of (Lucy Noel Martha Dyer-Edwards)",
        "Peter, Mrs. Catherine (Catherine Rizk)",
    ],
)
def test_real_name_shapes_are_accepted(name):
    assert Passenger.model_validate({**SAMPLE, "Name": name}).name == name


@pytest.mark.parametrize("cabin", ["NaN", "nan", "None", "Z9", "1"])
def test_cabin_without_a_real_deck_is_rejected(cabin):
    """'NaN' as a string would otherwise become deck N, which does not exist."""
    with pytest.raises(ValidationError, match="deck letter"):
        Passenger.model_validate({**SAMPLE, "Cabin": cabin})


@pytest.mark.parametrize("cabin", ["C85", "F G73", "C23 C25 C27", "T"])
def test_real_cabin_shapes_are_accepted(cabin):
    assert Passenger.model_validate({**SAMPLE, "Cabin": cabin}).cabin == cabin


@pytest.mark.parametrize("field", ["Cabin", "Ticket", "Embarked"])
def test_blank_text_is_treated_as_absent(field):
    record = Passenger.model_validate({**SAMPLE, field: "   "}).model_dump()
    assert record[field.lower()] is None


def test_batch_must_not_be_empty():
    with pytest.raises(ValidationError):
        MultiplePassengers(inputs=[])


@requires_kaggle_data
def test_every_official_record_validates():
    """The validators must never reject legitimate competition data: 1309 rows."""
    from titanic_model.data import load_test, load_train

    frames = [load_train()[0], load_test()[0]]
    for frame in frames:
        for record in frame.to_dict("records"):
            Passenger.model_validate({k: (None if pd.isna(v) else v) for k, v in record.items()})
