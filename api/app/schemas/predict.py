"""Wire contract. Field names match ``input_schema.json``; the aliases are the
Kaggle column names, so a client can post a row of ``test.csv`` unchanged.

Ten fields are hand-written rather than generated: ``test_schemas.py`` asserts
they still match the artifact's schema, which catches drift without a codegen step.
"""

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

#: "Braund, Mr. Owen Harris" — the title between the comma and the period is the
#: feature; all 1309 names in the two official files match this.
TITLE_PATTERN = re.compile(r",\s*[^.]+\.")

#: Decks present in the data. There is no deck N, so "NaN" as a string must not
#: quietly become one.
DECKS = frozenset("ABCDEFGT")


class Passenger(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    # required: no imputer downstream, or (name) the title would fall out of distribution
    pclass: Literal[1, 2, 3] = Field(alias="Pclass")
    name: str = Field(alias="Name", min_length=1)
    sex: Literal["male", "female"] = Field(alias="Sex")
    sibsp: int = Field(alias="SibSp", ge=0, le=20)
    parch: int = Field(alias="Parch", ge=0, le=20)

    # imputed by the pipeline when absent
    age: float | None = Field(None, alias="Age", gt=0, le=120)
    ticket: str | None = Field(None, alias="Ticket")
    fare: float | None = Field(None, alias="Fare", ge=0)
    cabin: str | None = Field(None, alias="Cabin")
    embarked: str | None = Field(None, alias="Embarked")

    @field_validator("name")
    @classmethod
    def name_must_carry_a_title(cls, value: str) -> str:
        if not TITLE_PATTERN.search(value):
            raise ValueError(
                "expected the Kaggle format 'Surname, Title. Given names' — the title "
                "is a feature, and without one the model predicts out of distribution"
            )
        return value

    @field_validator("cabin", "ticket", "embarked")
    @classmethod
    def blank_means_absent(cls, value: str | None) -> str | None:
        return value.strip() or None if value else None

    @field_validator("cabin")
    @classmethod
    def cabin_must_name_a_deck(cls, value: str | None) -> str | None:
        if value is not None and value[0].upper() not in DECKS:
            raise ValueError(
                f"cabin must start with a deck letter ({''.join(sorted(DECKS))}); "
                f"omit the field or send null when it is unknown"
            )
        return value


class MultiplePassengers(BaseModel):
    inputs: list[Passenger] = Field(min_length=1, max_length=1000)


class PredictionResults(BaseModel):
    predictions: list[int]
    probabilities: list[float]  # P(survived); predictions are these at a 0.5 threshold
    model_version: str
