from __future__ import annotations

import numpy as np
import pytest
from sklearn.base import is_classifier
from sklearn.ensemble import StackingClassifier

from titanic_model.builders import (
    MODELS,
    PREPROCESSING,
    SEARCH_PROVENANCE,
    STACK_MEMBERS,
    build,
    build_stack,
)

ALL_CONFIGS = sorted(MODELS)


class TestRegistry:
    def test_the_three_registries_agree(self):
        assert set(PREPROCESSING) == set(MODELS)
        assert set(SEARCH_PROVENANCE) == set(MODELS)

    def test_stack_members_exist(self):
        assert set(STACK_MEMBERS) <= set(MODELS)
        assert len(set(STACK_MEMBERS)) == len(STACK_MEMBERS)

    def test_unknown_config_lists_the_alternatives(self):
        with pytest.raises(KeyError, match="unknown config"):
            build("logreg_l2")

    @pytest.mark.parametrize("name", ALL_CONFIGS)
    def test_config_has_a_provenance_note(self, name):
        """Where a winner came from is the only thing that makes a frozen dict auditable."""
        assert len(SEARCH_PROVENANCE[name]) > 20

    @pytest.mark.parametrize("name", ALL_CONFIGS)
    def test_every_config_addresses_the_group_slot(self, name):
        """A slot missing from a config keeps its default and the config lies about it."""
        keys = {"ct__group_known", "ct__group_rate__drop", "ct__group_rate__outlier"}
        assert keys <= set(PREPROCESSING[name])

    @pytest.mark.parametrize("name", ALL_CONFIGS)
    def test_every_config_key_is_a_real_pipeline_slot(self, name):
        """set_params raises on a typo, so building is the check."""
        assert build(name) is not None


class TestIsolation:
    @pytest.mark.parametrize("name", ALL_CONFIGS)
    def test_two_pipelines_share_no_estimator(self, name):
        """Configs hold live objects: without clone() both pipelines fit one scaler."""
        a, b = build(name), build(name)
        for key, value in PREPROCESSING[name].items():
            if hasattr(value, "get_params"):
                assert a.get_params()[key] is not b.get_params()[key], key

    def test_the_registry_itself_is_not_fitted(self, frame):
        """A shared estimator would come back from build() carrying train state."""
        X, y = frame
        build("logreg").fit(X, y)
        assert not hasattr(PREPROCESSING["logreg"]["scaler"], "data_min_")

    def test_model_factories_return_new_objects(self):
        assert MODELS["logreg"]() is not MODELS["logreg"]()


class TestFit:
    @pytest.mark.parametrize("name", ALL_CONFIGS)
    def test_config_fits_and_predicts(self, name, frame):
        X, y = frame
        pipe = build(name).fit(X, y)

        pred = pipe.predict(X)
        proba = pipe.predict_proba(X)

        assert set(np.unique(pred)) <= {0, 1}
        assert proba.shape == (len(X), 2)
        assert np.allclose(proba.sum(axis=1), 1.0)
        assert is_classifier(pipe)

    @pytest.mark.parametrize("name", ALL_CONFIGS)
    def test_config_is_deterministic(self, name, frame):
        """Every estimator is seeded; two runs must agree bit for bit."""
        X, y = frame
        first = build(name).fit(X, y).predict_proba(X)
        second = build(name).fit(X, y).predict_proba(X)
        assert np.array_equal(first, second)


class TestStack:
    def test_default_membership(self):
        stack = build_stack()
        assert isinstance(stack, StackingClassifier)
        assert [name for name, _ in stack.estimators] == list(STACK_MEMBERS)

    def test_runs_single_threaded_by_default(self):
        """loky forks processes and the MLX Metal context does not survive a fork."""
        assert build_stack().n_jobs == 1

    def test_meta_learner_sees_probabilities_only(self):
        stack = build_stack()
        assert stack.stack_method == "predict_proba"
        assert stack.passthrough is False  # raw features would swamp 12 meta-columns

    def test_out_of_fold_meta_features(self):
        """cv=5 shuffled: fitting the meta-learner on in-sample probabilities leaks."""
        assert build_stack().cv.get_n_splits() == 5

    def test_fits_and_beats_a_coin_flip(self, frame):
        X, y = frame
        stack = build_stack(members=("logreg", "gb")).fit(X, y)
        assert (stack.predict(X) == y).mean() > 0.6

    def test_custom_membership(self, frame):
        X, y = frame
        stack = build_stack(members=("logreg", "rf")).fit(X, y)
        assert len(stack.estimators_) == 2
