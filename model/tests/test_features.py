from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from titanic_model.features import (
    DEFAULT_TITLE_MAP,
    AgeFeatures,
    CabinCatFeature,
    FamilyFeatures,
    FareFeatures,
    GroupSurvival,
    InteractionFeatures,
    Interactions,
    NaNReporter,
    TitleFeature,
)


def names(*titles: str) -> pd.DataFrame:
    return pd.DataFrame({"name": [f"Fam{i}, {t}. Given{i}" for i, t in enumerate(titles)]})


class TestTitleFeature:
    def test_maps_the_documented_titles(self):
        out = TitleFeature().fit_transform(names("Mr", "Mlle", "Mme", "Capt", "Dona"))
        assert out["title"].tolist() == ["Mr", "Miss", "Mrs", "Officer", "Aristocratic"]

    def test_strips_the_article(self):
        """'the Countess' is one row of the Kaggle train set and maps to nothing without this."""
        assert TitleFeature().fit_transform(names("the Countess"))["title"][0] == "Aristocratic"

    def test_unknown_title_falls_back(self):
        out = TitleFeature(fallback="Empty").fit_transform(names("Wizard"))
        assert out["title"][0] == "Empty"

    def test_default_map_is_not_baked_into_params(self):
        """clone must return what was passed in, so __init__ may not resolve the default."""
        assert TitleFeature().get_params()["title_map"] is None

        custom = {"Mr": "X"}
        assert clone(TitleFeature(custom)).get_params()["title_map"] == custom
        assert clone(TitleFeature(custom)).fit_transform(names("Mr"))["title"][0] == "X"

    def test_keeps_the_original_columns(self):
        df = names("Mr")
        out = TitleFeature().fit_transform(df)
        assert "name" in out.columns
        assert "title" not in df.columns  # transform must not mutate its input

    @pytest.mark.parametrize("dtype", ["object", "float64"])
    def test_all_missing_column_falls_back(self, dtype):
        """An empty text column arrives as float64 and .str refuses it."""
        df = pd.DataFrame({"name": pd.Series([None], dtype=dtype)})
        assert TitleFeature().fit_transform(df)["title"].tolist() == ["Empty"]


class TestAgeFeatures:
    def frame(self):
        return pd.DataFrame(
            {
                "title": ["Master", "Master", "Mr", "Mr", "Mr"],
                "age": [4.0, np.nan, 40.0, 50.0, np.nan],
                "pclass": [3, 3, 1, 1, 2],
            }
        )

    def test_imputes_by_group_not_by_global_median(self):
        """The whole point of grouping: a child gets a child's age, not 40-ish."""
        out = AgeFeatures(group="title").fit_transform(self.frame())
        assert out["age"][1] == 4.0  # Master median
        assert out["age"][4] == 45.0  # Mr median
        assert out["age"].isna().sum() == 0

    def test_unseen_group_falls_back_to_the_global_median(self):
        fitted = AgeFeatures(group="title").fit(self.frame())
        unseen = pd.DataFrame({"title": ["Officer"], "age": [np.nan], "pclass": [1]})
        assert fitted.transform(unseen)["age"][0] == fitted.age_median_

    def test_requires_the_title_column(self):
        """Pipeline order guard: AgeFeatures must run after TitleFeature."""
        with pytest.raises(ValueError, match="must come after TitleFeature"):
            AgeFeatures(group="title").fit(pd.DataFrame({"age": [1.0], "pclass": [1]}))

    def test_bins_and_interaction(self):
        out = AgeFeatures().fit_transform(self.frame())
        assert out["age_bin"].tolist() == [0, 0, 3, 3, 3]
        assert out["age_pclass"][0] == "0_3"


class TestFareFeatures:
    def frame(self, fares, tickets=None):
        n = len(fares)
        return pd.DataFrame(
            {
                "fare": fares,
                "ticket": tickets or [f"T{i}" for i in range(n)],
                "sex": ["male"] * n,
                "pclass": [3] * n,
            }
        )

    def test_fare_is_split_across_the_ticket_group(self):
        out = FareFeatures().fit_transform(self.frame([100.0] * 4, tickets=["A", "A", "B", "C"]))
        assert out["fare_per_person"].tolist() == [50.0, 50.0, 100.0, 100.0]

    def test_unseen_ticket_counts_as_one_passenger(self):
        fitted = FareFeatures().fit(self.frame([10.0, 20.0, 30.0, 40.0]))
        out = fitted.transform(self.frame([60.0], tickets=["NEW"]))
        assert out["fare_per_person"][0] == 60.0

    def test_transform_uses_train_boundaries(self):
        """cut, not qcut: quartiles recomputed on the test set would leak and shift."""
        fitted = FareFeatures().fit(self.frame([10.0, 20.0, 30.0, 40.0]))
        out = fitted.transform(self.frame([10.0, 40.0], tickets=["A", "B"]))
        assert out["fare_bin"].tolist() == ["low", "vip"]

    def test_out_of_range_fare_gets_its_own_category(self):
        fitted = FareFeatures().fit(self.frame([10.0, 20.0, 30.0, 40.0]))
        out = fitted.transform(self.frame([999.0], tickets=["A"]))
        assert out["fare_bin"][0] == "_oob"  # not silently clipped to 'vip'

    def test_drops_ticket_when_asked(self):
        out = FareFeatures(drop_ticket=True).fit_transform(self.frame([1.0, 2.0, 3.0, 4.0]))
        assert "ticket" not in out.columns
        assert "sex_fare" in out.columns and "pclass_fare" in out.columns


class TestCabinCatFeature:
    def test_deck_and_side(self):
        df = pd.DataFrame({"cabin": ["C85", "C86", "B", None]})
        out = CabinCatFeature().fit_transform(df)
        assert out["cabin_cat"].tolist() == ["C_odd", "C_even", "B_na", "Missing"]

    def test_deck_only(self):
        df = pd.DataFrame({"cabin": ["C85", None]})
        out = CabinCatFeature(add_side=False).fit_transform(df)
        assert out["cabin_cat"].tolist() == ["C", "Missing"]

    @pytest.mark.parametrize("dtype", ["object", "float64"])
    def test_all_missing_column(self, dtype):
        """A single-passenger request without a cabin: feature_engine's imputers
        downcast the empty column to float64 and .str refuses a float column."""
        df = pd.DataFrame({"cabin": pd.Series([None], dtype=dtype)})
        assert CabinCatFeature().fit_transform(df)["cabin_cat"].tolist() == ["Missing"]

    def test_output_dtype_is_object(self):
        """The encoders downstream were fitted against object, not StringDtype."""
        out = CabinCatFeature().fit_transform(pd.DataFrame({"cabin": ["C85", None]}))
        assert out["cabin_cat"].dtype == object


class TestGroupSurvival:
    def frame(self):
        """Two families: the Browns all died, the Greens all survived."""
        rows = [
            ("Brown, Mrs. A", "female", "Mrs", "T1", 0),
            ("Brown, Miss. B", "female", "Miss", "T1", 0),
            ("Brown, Master. C", "male", "Master", "T1", 0),
            ("Green, Mrs. D", "female", "Mrs", "T2", 1),
            ("Green, Miss. E", "female", "Miss", "T2", 1),
            ("Smith, Mr. F", "male", "Mr", "T3", 0),
        ]
        X = pd.DataFrame(
            [{"name": n, "sex": s, "title": t, "ticket": k} for n, s, t, k, _ in rows]
        )
        return X, pd.Series([r[-1] for r in rows])

    def test_needs_labels(self):
        X, _ = self.frame()
        with pytest.raises(ValueError, match="needs y"):
            GroupSurvival().fit(X)

    def test_separates_the_two_families(self):
        X, y = self.frame()
        out = GroupSurvival().fit(X, y).transform(X)
        assert out.loc[0, "group_rate"] < 0.3  # Browns
        assert out.loc[3, "group_rate"] > 0.7  # Greens
        assert out["group_known"].tolist() == [1, 1, 1, 1, 1, 0]

    def test_leave_one_out_hides_the_own_label(self):
        """Without it a row reads its own outcome back and the model memorises."""
        X, y = self.frame()
        gs = GroupSurvival()
        loo = gs.fit_transform(X, y)["group_rate"]
        plain = gs.transform(X)["group_rate"]
        assert (loo != plain).any()
        # the lone man contributes nothing to any group, so he sees only the prior
        assert loo.iloc[5] == pytest.approx(gs.prior_)

    def test_adult_men_do_not_count_towards_a_group(self):
        """They died almost regardless; their labels only dilute the statistic."""
        X, y = self.frame()
        extra = pd.concat([X, pd.DataFrame([{"name": "Green, Mr. G", "sex": "male",
                                             "title": "Mr", "ticket": "T2"}])], ignore_index=True)
        y_extra = pd.concat([y, pd.Series([0])], ignore_index=True)
        rate = GroupSurvival().fit(extra, y_extra).transform(extra).loc[3, "group_rate"]
        assert rate > 0.7  # still a surviving family despite the dead man

    def test_unknown_group_falls_back_to_the_prior(self):
        X, y = self.frame()
        fitted = GroupSurvival().fit(X, y)
        stranger = pd.DataFrame([{"name": "Nobody, Mrs. Z", "sex": "female",
                                  "title": "Mrs", "ticket": "T99"}])
        out = fitted.transform(stranger)
        assert out["group_rate"].iloc[0] == pytest.approx(fitted.prior_)
        assert out["group_known"].iloc[0] == 0

    def test_missing_ticket_is_not_a_group(self):
        X, y = self.frame()
        X.loc[0, "ticket"] = None
        out = GroupSurvival().fit(X, y).transform(X)
        assert np.isfinite(out["group_rate"]).all()  # surname still carries her


def test_family_features():
    df = pd.DataFrame({"sibsp": [0, 1, 3], "parch": [0, 2, 0]})
    out = FamilyFeatures().fit_transform(df)
    assert out["family_size"].tolist() == [1, 4, 4]
    assert out["is_alone"].tolist() == [1, 0, 0]


def test_interaction_features_fill_embarked_from_train_mode():
    train = pd.DataFrame({"sex": ["male"] * 3, "pclass": [1, 2, 3], "embarked": ["S", "S", "C"]})
    fitted = InteractionFeatures().fit(train)
    out = fitted.transform(pd.DataFrame({"sex": ["female"], "pclass": [2], "embarked": [None]}))
    assert out["sex_pclass"][0] == "female_2"
    assert out["embarked_pclass"][0] == "S_2"  # train mode, not the test mode


class TestInteractions:
    def test_keeps_only_the_products(self):
        X = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        out = Interactions().fit_transform(X)
        assert out.shape == (2, 1)  # a*b, without bias and without a, b
        assert out[:, 0].tolist() == [3.0, 8.0]

    def test_feature_names(self):
        X = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0], "c": [5.0, 6.0]})
        assert list(Interactions().fit(X).get_feature_names_out()) == ["a b", "a c", "b c"]

    def test_single_column_is_an_error_not_an_empty_matrix(self):
        with pytest.raises(ValueError, match="nothing to keep"):
            Interactions().fit(pd.DataFrame({"a": [1.0, 2.0]}))

    def test_survives_global_pandas_output(self):
        """Without the asarray guard the inner mask becomes a label lookup and raises."""
        from sklearn import set_config

        X = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        set_config(transform_output="pandas")
        try:
            out = Interactions().fit_transform(X)
        finally:
            set_config(transform_output="default")
        assert np.asarray(out).ravel().tolist() == [3.0, 8.0]


class TestNaNReporter:
    def test_raises_on_nan_when_configured(self):
        with pytest.raises(ValueError, match="NaN in 1 columns"):
            NaNReporter("check", raise_on_nan=True).fit(pd.DataFrame({"a": [1.0, np.nan]}))

    def test_raises_on_non_numeric_columns(self):
        with pytest.raises(ValueError, match="non-numeric columns before the model"):
            NaNReporter("check", raise_on_nan=True).fit(
                pd.DataFrame({"a": [1.0, np.nan], "b": ["x", None]})
            )

    def test_passes_data_through_unchanged(self):
        df = pd.DataFrame({"a": [1.0, 2.0]})
        pd.testing.assert_frame_equal(NaNReporter().fit_transform(df), df)


def test_title_map_covers_every_kaggle_title():
    """Titles in the two official files; anything else would silently become 'Empty'."""
    kaggle = {
        "Mr", "Mrs", "Miss", "Master", "Don", "Rev", "Dr", "Mme", "Ms", "Major",
        "Lady", "Sir", "Mlle", "Col", "Capt", "Countess", "Jonkheer", "Dona",
    }
    assert kaggle - set(DEFAULT_TITLE_MAP) == set()
