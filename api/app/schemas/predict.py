"""Wire contract. Field names match ``input_schema.json``; the aliases are the
Kaggle column names, so a client can post a row of ``test.csv`` unchanged.

Ten fields are hand-written rather than generated: ``test_schemas.py`` asserts
they still match the artifact's schema, which catches drift without a codegen step.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


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


class MultiplePassengers(BaseModel):
    inputs: list[Passenger] = Field(min_length=1, max_length=1000)


class PredictionResults(BaseModel):
    predictions: list[int]
    probabilities: list[float]  # P(survived); predictions are these at a 0.5 threshold
    model_version: str
