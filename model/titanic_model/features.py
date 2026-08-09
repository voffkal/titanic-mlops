from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PolynomialFeatures

__all__ = [
    "DEFAULT_TITLE_MAP",
    "Interactions",
    "TitleFeature",
    "AgeFeatures",
    "FareFeatures",
    "CabinCatFeature",
    "FamilyFeatures",
    "InteractionFeatures",
    "NaNReporter",
]


DEFAULT_TITLE_MAP: dict[str, str] = {
    "Mr": "Mr",
    "Miss": "Miss",
    "Mrs": "Mrs",
    "Master": "Master",
    "Mlle": "Miss",
    "Ms": "Miss",
    "Mme": "Mrs",
    "Lady": "Aristocratic",
    "Countess": "Aristocratic",
    "Sir": "Aristocratic",
    "Don": "Aristocratic",
    "Dona": "Aristocratic",
    "Jonkheer": "Aristocratic",
    "Dr": "Officer",
    "Rev": "Officer",
    "Col": "Officer",
    "Major": "Officer",
    "Capt": "Officer",
}


class Interactions(BaseEstimator, TransformerMixin):

    def __init__(
        self,
        degree: int = 2,
        interaction_only: bool = True,
        include_bias: bool = False,
    ) -> None:
        self.degree = degree
        self.interaction_only = interaction_only
        self.include_bias = include_bias

    def fit(self, X, y=None) -> Interactions:
        self.poly_ = PolynomialFeatures(
            degree=self.degree,
            interaction_only=self.interaction_only,
            include_bias=self.include_bias,
        ).fit(X)
        self.keep_ = self.poly_.powers_.sum(axis=1) >= 2  # 0 = bias, 1 = originals -> out
        if not self.keep_.any():
            raise ValueError(
                f"Interactions: nothing to keep (degree={self.degree}, "
                f"columns={self.poly_.n_features_in_}). Needs >= 2 columns and degree >= 2."
            )
        self.n_features_in_ = self.poly_.n_features_in_
        return self

    def transform(self, X) -> np.ndarray:
        # asarray: guards against a global set_config(transform_output='pandas')
        return np.asarray(self.poly_.transform(X))[:, self.keep_]

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return self.poly_.get_feature_names_out(input_features)[self.keep_]


class TitleFeature(BaseEstimator, TransformerMixin):
    def __init__(self, title_map: dict[str, str] | None = None, fallback: str = "Empty") -> None:
        self.title_map = title_map  # no default resolution here: clone must return what was passed in
        self.fallback = fallback

    def _resolved_map(self) -> dict[str, str]:
        return DEFAULT_TITLE_MAP if self.title_map is None else self.title_map

    def fit(self, X: pd.DataFrame, y=None) -> TitleFeature:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        raw_title = X["name"].str.extract(r",\s*([^\.]+)\.")[0]
        raw_title = raw_title.str.replace(r"^the\s+", "", regex=True).str.strip()  # "the Countess"
        X["title"] = raw_title.map(self._resolved_map()).fillna(self.fallback)
        return X


class AgeFeatures(BaseEstimator, TransformerMixin):

    def __init__(
        self,
        group: str = "title",
        bins: tuple[float, ...] = (0, 12, 18, 35, 60, np.inf),
    ) -> None:
        self.group = group
        self.bins = bins

    def fit(self, X: pd.DataFrame, y=None) -> AgeFeatures:
        if self.group not in X.columns:
            raise ValueError(
                f"AgeFeatures(group={self.group!r}): column not present in input. "
                f"With group='title' this step must come after TitleFeature. "
                f"Available: {sorted(X.columns)}"
            )
        self.age_median_by_group_ = X.groupby(self.group)["age"].median()
        self.age_median_ = X["age"].median()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        # second fillna covers a group unseen in train
        by_group = X[self.group].map(self.age_median_by_group_)
        X["age"] = X["age"].fillna(by_group).fillna(self.age_median_)

        X["age_bin"] = pd.cut(X["age"], bins=self.bins, labels=False)
        X["age_pclass"] = X["age_bin"].astype(str) + "_" + X["pclass"].astype(str)
        return X


class FareFeatures(BaseEstimator, TransformerMixin):

    def __init__(self, q: int = 4, drop_ticket: bool = True) -> None:
        self.q = q
        self.drop_ticket = drop_ticket

    def fit(self, X: pd.DataFrame, y=None) -> FareFeatures:
        _, self.fare_bins_ = pd.qcut(X["fare"], q=self.q, retbins=True, duplicates="drop")
        # duplicates='drop' may yield fewer bins -> fit labels to the actual count
        n_bins = len(self.fare_bins_) - 1
        base_labels = ["low", "mid", "high", "vip"]
        self.labels_ = base_labels[:n_bins] if n_bins <= len(base_labels) else list(range(n_bins))

        self.ticket_sizes_ = X.groupby("ticket")["ticket"].count()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        group_size = X["ticket"].map(self.ticket_sizes_).fillna(1)  # unseen ticket = single passenger
        X["fare_per_person"] = (X["fare"] / group_size).astype(float)

        X["fare_bin"] = pd.cut(  # cut, not qcut: train boundaries
            X["fare"],
            bins=self.fare_bins_,
            labels=self.labels_,
            include_lowest=True,
        )
        # outside train range -> own category, not the nearest edge
        X["fare_bin"] = X["fare_bin"].cat.add_categories(["_oob"]).fillna("_oob")

        X["sex_fare"] = X["sex"].astype(str) + "_" + X["fare_bin"].astype(str)
        X["pclass_fare"] = X["pclass"].astype(str) + "_" + X["fare_bin"].astype(str)

        if self.drop_ticket:
            X = X.drop(columns=["ticket"])

        return X


class CabinCatFeature(BaseEstimator, TransformerMixin):

    def __init__(self, fill_value: str = "Missing", add_side: bool = True) -> None:
        self.fill_value = fill_value
        self.add_side = add_side

    def fit(self, X: pd.DataFrame, y=None) -> CabinCatFeature:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        deck = X["cabin"].str[0]

        if self.add_side:
            num = X["cabin"].str.extract(r"(\d+)", expand=False).astype("float")
            # cabin number parity = side of the ship
            side = np.where(num.isna(), "na", np.where(num % 2 == 0, "even", "odd"))
            X["cabin_cat"] = (deck + "_" + side).fillna(self.fill_value)
        else:
            X["cabin_cat"] = deck.fillna(self.fill_value)

        return X


class FamilyFeatures(BaseEstimator, TransformerMixin):

    def fit(self, X: pd.DataFrame, y=None) -> FamilyFeatures:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X["family_size"] = X["sibsp"] + X["parch"] + 1
        X["is_alone"] = (X["family_size"] == 1).astype(int)
        return X


class InteractionFeatures(BaseEstimator, TransformerMixin):

    def fit(self, X: pd.DataFrame, y=None) -> InteractionFeatures:
        self.embarked_mode_ = X["embarked"].mode()[0]  # from train: Kaggle train has 2 missing values
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X["sex_pclass"] = X["sex"] + "_" + X["pclass"].astype(str)

        embarked_filled = X["embarked"].fillna(self.embarked_mode_)
        X["embarked_pclass"] = embarked_filled.astype(str) + "_" + X["pclass"].astype(str)
        return X


class NaNReporter(BaseEstimator, TransformerMixin):

    def __init__(
        self,
        name: str = "check",
        raise_on_nan: bool = False,
        verbose: bool = False,
    ) -> None:
        self.name = name
        self.raise_on_nan = raise_on_nan
        self.verbose = verbose

    def fit(self, X, y=None) -> NaNReporter:
        self._report(X, stage="fit")
        return self

    def transform(self, X):
        self._report(X, stage="transform")
        return X

    def _report(self, X, stage: str) -> None:
        if not isinstance(X, pd.DataFrame):  # after ct the input is numpy; names are needed for the message
            X = pd.DataFrame(X)

        na_counts = X.isna().sum()
        na_cols = na_counts[na_counts > 0]

        if len(na_cols) == 0:
            if self.verbose:
                print(f"[{self.name}/{stage}] OK: no NaN, shape={X.shape}")
            return

        report = "\n".join(
            f"    {col}: {cnt} NaN ({cnt / len(X):.1%}), dtype={X[col].dtype}"
            for col, cnt in na_cols.items()
        )
        msg = f"[{self.name}/{stage}] NaN in {len(na_cols)} columns (shape={X.shape}):\n{report}"

        non_numeric = X.select_dtypes(exclude="number").columns.tolist()
        if non_numeric and self.raise_on_nan:
            raise ValueError(
                f"[{self.name}/{stage}] non-numeric columns before the model: {non_numeric}"
            )

        if self.raise_on_nan:
            raise ValueError(msg)
        print(msg)
