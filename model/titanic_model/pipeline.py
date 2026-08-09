from __future__ import annotations

import numpy as np
from feature_engine import encoding as fe
from feature_engine import imputation as mdi
from feature_engine.selection import DropConstantFeatures, DropCorrelatedFeatures
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from .features import (
    AgeFeatures,
    CabinCatFeature,
    FamilyFeatures,
    FareFeatures,
    InteractionFeatures,
    NaNReporter,
    TitleFeature,
)

__all__ = ["ZERO", "POLY_COLUMNS", "make_ct", "make_pipe"]


#: Zeroes a column without changing matrix shape: this is how a config
#: "switches off" a feature. Named functions, not lambdas: with lambdas
#: joblib.dump of the artifact fails.
def _zeros_column(X: np.ndarray) -> np.ndarray:
    return np.zeros((len(X), 1))


def _zero_feature_names(self, input_features) -> np.ndarray:
    return np.array([f"{n}_ZERO" for n in input_features])


ZERO = FunctionTransformer(_zeros_column, feature_names_out=_zero_feature_names)


#: Input of the ``ct__poly`` slot: the only meaningfully continuous columns.
#: ct hands RAW columns to each slot, so a transform for the skewed
#: fare_per_person belongs inside the slot — see builders.LOGREG_POLY_PREP.
POLY_COLUMNS = ["age", "fare_per_person", "family_size", "pclass"]


def make_ct() -> ColumnTransformer:
    return ColumnTransformer(
        [
            (
                "title",
                Pipeline(
                    [
                        ("rare", "passthrough"),
                        ("encode", fe.OneHotEncoder(drop_last=True, ignore_format=True)),
                        ("drop", "passthrough"),
                    ]
                ),
                ["title"],
            ),
            (
                "age",
                Pipeline(
                    [
                        ("outlier", "passthrough"),
                        ("drop", "passthrough"),
                    ]
                ),
                ["age"],
            ),
            (
                "pclass",
                Pipeline(
                    [
                        (
                            "encode",
                            fe.OrdinalEncoder(
                                encoding_method="ordered",
                                unseen="encode",
                                ignore_format=True,
                            ),
                        ),
                        ("drop", "passthrough"),
                    ]
                ),
                ["pclass"],
            ),
            (
                "sex",
                Pipeline(
                    [
                        (
                            "encode",
                            fe.OneHotEncoder(drop_last_binary=True, ignore_format=True),
                        ),
                        ("drop", "passthrough"),
                    ]
                ),
                ["sex"],
            ),
            ("sibsp", Pipeline([("drop", "passthrough")]), ["sibsp"]),
            ("parch", Pipeline([("drop", "passthrough")]), ["parch"]),
            (
                "family_size",
                Pipeline(
                    [
                        ("outlier", "passthrough"),
                        ("drop", "passthrough"),
                    ]
                ),
                ["family_size"],
            ),
            ("is_alone", "passthrough", ["is_alone"]),
            (
                "fare_per_person",
                Pipeline(
                    [
                        ("outlier", "passthrough"),
                        ("transform", "passthrough"),
                        ("drop", "passthrough"),
                    ]
                ),
                ["fare_per_person"],
            ),
            (
                "fare_bin",
                Pipeline(
                    [
                        (
                            "encode",
                            fe.OrdinalEncoder(
                                encoding_method="ordered",
                                unseen="encode",
                                ignore_format=True,
                            ),
                        ),
                        ("drop", "passthrough"),
                    ]
                ),
                ["fare_bin"],
            ),
            (
                "embarked",
                Pipeline(
                    [
                        ("encode", fe.OneHotEncoder(drop_last=True, ignore_format=True)),
                        ("drop", "passthrough"),
                    ]
                ),
                ["embarked"],
            ),
            (
                "cabin_cat",
                Pipeline(
                    [
                        ("rare", "passthrough"),
                        ("encode", fe.OneHotEncoder(drop_last=True, ignore_format=True)),
                        ("drop", "passthrough"),
                    ]
                ),
                ["cabin_cat"],
            ),
            ("cabin_na", "passthrough", ["cabin_na"]),
            (
                "sex_pclass",
                Pipeline(
                    [
                        ("encode", fe.OneHotEncoder(drop_last=True, ignore_format=True)),
                        ("drop", "passthrough"),
                    ]
                ),
                ["sex_pclass"],
            ),
            (
                "embarked_pclass",
                Pipeline(
                    [
                        ("encode", fe.OneHotEncoder(drop_last=True, ignore_format=True)),
                        ("drop", "passthrough"),
                    ]
                ),
                ["embarked_pclass"],
            ),
            (
                "age_bin",
                Pipeline(
                    [
                        ("encode", "passthrough"),  # already int 0..4, order is meaningful
                        ("drop", "passthrough"),
                    ]
                ),
                ["age_bin"],
            ),
            (
                "age_pclass",
                Pipeline(
                    [
                        ("rare", "passthrough"),
                        (
                            "encode",
                            fe.OrdinalEncoder(
                                encoding_method="ordered",
                                unseen="encode",
                                ignore_format=True,
                            ),
                        ),
                        ("drop", "passthrough"),
                    ]
                ),
                ["age_pclass"],
            ),
            (
                "sex_fare",
                Pipeline(
                    [
                        ("rare", "passthrough"),
                        ("encode", fe.OneHotEncoder(drop_last=True, ignore_format=True)),
                        ("drop", "passthrough"),
                    ]
                ),
                ["sex_fare"],
            ),
            (
                "pclass_fare",
                Pipeline(
                    [
                        ("rare", "passthrough"),
                        ("encode", fe.OneHotEncoder(drop_last=True, ignore_format=True)),
                        ("drop", "passthrough"),
                    ]
                ),
                ["pclass_fare"],
            ),
            # 'drop', not 'passthrough': otherwise the columns are emitted twice
            ("poly", "drop", POLY_COLUMNS),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def make_pipe() -> ImbPipeline:
    return ImbPipeline(
        [
            ("missing_indicators", mdi.AddMissingIndicator(variables=["cabin"])),  # before imputers
            ("fare_imputer", None),  # before fare_feats: quantile edges need non-null fare
            ("embarked_imputer", None),
            ("title", TitleFeature()),  # before age_feats: age is imputed by title
            ("age_feats", AgeFeatures(group="title")),
            ("fare_feats", FareFeatures(q=4, drop_ticket=True)),
            ("cabin_cat", CabinCatFeature(fill_value="Missing")),
            ("family_feats", FamilyFeatures()),
            ("pclass_feats", InteractionFeatures()),
            ("ct", make_ct()),
            ("nan_after_ct", NaNReporter("after_ct", raise_on_nan=True)),
            (
                "drop_correlated",
                DropCorrelatedFeatures(threshold=0.99, missing_values="raise", method="pearson"),
            ),
            ("drop_constant", DropConstantFeatures(tol=0.99, missing_values="raise")),
            ("scaler", None),
            ("sampler", None),
            ("model", None),
        ]
    )
