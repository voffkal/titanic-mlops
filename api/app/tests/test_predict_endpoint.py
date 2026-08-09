from __future__ import annotations

from app.tests.conftest import SAMPLE
from titanic_model import __version__ as model_version


def post(client, *records):
    return client.post("/api/v1/predict", json={"inputs": list(records)})


def test_happy_path(client):
    r = post(client, SAMPLE, SAMPLE)
    assert r.status_code == 200

    body = r.json()
    assert body["model_version"] == model_version
    assert body["predictions"] == body["predictions"][:1] * 2  # same input, same answer
    assert all(0.0 <= p <= 1.0 for p in body["probabilities"])


def test_predictions_follow_the_probabilities(client):
    body = post(client, SAMPLE).json()
    assert body["predictions"][0] == int(body["probabilities"][0] >= 0.5)


def test_kaggle_column_names_are_accepted(client):
    """A client can post a row of test.csv without renaming anything."""
    assert "SibSp" in SAMPLE
    assert post(client, SAMPLE).status_code == 200


def test_lowercase_names_are_accepted_too(client):
    lowered = {k.lower(): v for k, v in SAMPLE.items()}
    assert post(client, lowered).status_code == 200


def test_optional_fields_may_be_missing(client):
    """age/fare/cabin/embarked/ticket have an imputer or a 'missing' category."""
    minimal = {"Pclass": 3, "Name": "Smith, Mr. John", "Sex": "male", "SibSp": 0, "Parch": 0}
    r = post(client, minimal)
    assert r.status_code == 200
    assert 0.0 <= r.json()["probabilities"][0] <= 1.0


def test_a_cabinless_single_passenger_works(client):
    """77% of passengers have no cabin; a one-row batch made the column float64."""
    r = post(client, {**SAMPLE, "Cabin": None})
    assert r.status_code == 200


def test_required_field_missing_is_422(client):
    incomplete = {k: v for k, v in SAMPLE.items() if k != "Pclass"}
    assert post(client, incomplete).status_code == 422


def test_unknown_class_is_422(client):
    assert post(client, {**SAMPLE, "Pclass": 4}).status_code == 422


def test_unknown_sex_is_422(client):
    assert post(client, {**SAMPLE, "Sex": "unknown"}).status_code == 422


def test_impossible_age_is_422(client):
    assert post(client, {**SAMPLE, "Age": 200}).status_code == 422


def test_negative_fare_is_422(client):
    assert post(client, {**SAMPLE, "Fare": -1}).status_code == 422


def test_wrong_type_is_422(client):
    assert post(client, {**SAMPLE, "Age": "тридцать"}).status_code == 422


def test_empty_batch_is_422(client):
    assert post(client).status_code == 422


def test_batch_is_capped(client):
    assert post(client, *([SAMPLE] * 1001)).status_code == 422


def test_batch_order_is_preserved(client):
    woman = SAMPLE
    man = {**SAMPLE, "Sex": "male", "Name": "Cumings, Mr. John", "Pclass": 3, "Cabin": None}
    probabilities = post(client, woman, man, woman).json()["probabilities"]

    assert probabilities[0] == probabilities[2]
    assert probabilities[0] != probabilities[1]
