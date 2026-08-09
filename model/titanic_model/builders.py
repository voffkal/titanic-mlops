"""Frozen model configs — the winners of the notebook searches.

``ct__*__drop`` slots set to ``ZERO`` are switched-off features. Do not read
them as feature importance: the sets barely overlap across models while the
scores stay the same. See README.
"""

from __future__ import annotations

from collections.abc import Callable

from category_encoders import CatBoostEncoder, GLMMEncoder, HelmertEncoder
from feature_engine.encoding import (
    CountFrequencyEncoder,
    MeanEncoder,
    OneHotEncoder,
    OrdinalEncoder,
    RareLabelEncoder,
)
from feature_engine.imputation import CategoricalImputer, MeanMedianImputer
from feature_engine.outliers import Winsorizer
from feature_engine.transformation import LogCpTransformer, YeoJohnsonTransformer
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import EditedNearestNeighbours, TomekLinks
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, PowerTransformer, StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from .features import Interactions
from .mlx_mlp import MLXClassifier
from .pipeline import ZERO, make_pipe

__all__ = ["MODELS", "PREPROCESSING", "SEARCH_PROVENANCE", "STACK_MEMBERS", "build", "build_stack"]

LOGREG_PREP = {
    'ct__age__drop': 'passthrough',
    'ct__age__outlier': Winsorizer(fold=3, tail='both'),
    'ct__age_bin__drop': 'passthrough',
    'ct__age_bin__encode': 'passthrough',
    'ct__age_pclass__drop': ZERO,
    'ct__age_pclass__encode': CatBoostEncoder(random_state=0),
    'ct__age_pclass__rare': RareLabelEncoder(ignore_format=True, n_categories=2),
    'ct__cabin_cat__drop': 'passthrough',
    'ct__cabin_cat__encode': CountFrequencyEncoder(encoding_method='frequency', ignore_format=True, unseen='encode'),
    'ct__cabin_cat__rare': RareLabelEncoder(ignore_format=True, n_categories=2, tol=0.03),
    'ct__cabin_na': 'passthrough',
    'ct__embarked__drop': 'passthrough',
    'ct__embarked__encode': OrdinalEncoder(encoding_method='arbitrary', ignore_format=True, unseen='encode'),
    'ct__embarked_pclass__drop': 'passthrough',
    'ct__embarked_pclass__encode': MeanEncoder(ignore_format=True, unseen='encode'),
    'ct__family_size__drop': 'passthrough',
    'ct__family_size__outlier': 'passthrough',
    'ct__fare_bin__drop': ZERO,
    'ct__fare_bin__encode': OrdinalEncoder(ignore_format=True, unseen='encode'),
    'ct__fare_per_person__drop': ZERO,
    'ct__fare_per_person__outlier': 'passthrough',
    'ct__fare_per_person__transform': 'passthrough',
    'ct__is_alone': 'passthrough',
    'ct__parch__drop': 'passthrough',
    'ct__pclass__drop': ZERO,
    'ct__pclass__encode': 'passthrough',
    'ct__pclass_fare__drop': ZERO,
    'ct__pclass_fare__encode': CatBoostEncoder(random_state=0),
    'ct__pclass_fare__rare': RareLabelEncoder(ignore_format=True, n_categories=2),
    'ct__sex__drop': 'passthrough',
    'ct__sex__encode': OrdinalEncoder(encoding_method='arbitrary', ignore_format=True, unseen='encode'),
    'ct__sex_fare__drop': 'passthrough',
    'ct__sex_fare__encode': MeanEncoder(ignore_format=True, unseen='encode'),
    'ct__sex_fare__rare': RareLabelEncoder(ignore_format=True, n_categories=2, tol=0.03),
    'ct__sex_pclass__drop': 'passthrough',
    'ct__sex_pclass__encode': OneHotEncoder(drop_last=True, ignore_format=True),
    'ct__sibsp__drop': 'passthrough',
    'ct__title__drop': 'passthrough',
    'ct__title__encode': HelmertEncoder(),
    'ct__title__rare': RareLabelEncoder(ignore_format=True, n_categories=2),
    'embarked_imputer': CategoricalImputer(ignore_format=True, imputation_method='frequent', variables=['embarked']),
    'fare_imputer': MeanMedianImputer(variables=['fare']),
    'sampler': None,
    'scaler': MinMaxScaler(),
}

LOGREG_L1_PREP = {
    'ct__age__drop': ZERO,
    'ct__age__outlier': Winsorizer(fold=3, tail='both'),
    'ct__age_bin__drop': 'passthrough',
    'ct__age_bin__encode': MeanEncoder(ignore_format=True, unseen='encode'),
    'ct__age_pclass__drop': 'passthrough',
    'ct__age_pclass__encode': MeanEncoder(ignore_format=True, smoothing=10, unseen='encode'),
    'ct__age_pclass__rare': RareLabelEncoder(ignore_format=True, n_categories=2),
    'ct__cabin_cat__drop': ZERO,
    'ct__cabin_cat__encode': MeanEncoder(ignore_format=True, unseen='encode'),
    'ct__cabin_cat__rare': RareLabelEncoder(ignore_format=True, n_categories=2, tol=0.03),
    'ct__cabin_na': 'passthrough',
    'ct__embarked__drop': 'passthrough',
    'ct__embarked__encode': OrdinalEncoder(encoding_method='arbitrary', ignore_format=True, unseen='encode'),
    'ct__embarked_pclass__drop': ZERO,
    'ct__embarked_pclass__encode': MeanEncoder(ignore_format=True, unseen='encode'),
    'ct__family_size__drop': 'passthrough',
    'ct__family_size__outlier': 'passthrough',
    'ct__fare_bin__drop': ZERO,
    'ct__fare_bin__encode': OrdinalEncoder(ignore_format=True, unseen='encode'),
    'ct__fare_per_person__drop': ZERO,
    'ct__fare_per_person__outlier': Winsorizer(capping_method='quantiles', fold=0.01, tail='both'),
    'ct__fare_per_person__transform': LogCpTransformer(),
    'ct__is_alone': 'drop',
    'ct__parch__drop': ZERO,
    'ct__pclass__drop': 'passthrough',
    'ct__pclass__encode': OrdinalEncoder(ignore_format=True, unseen='encode'),
    'ct__pclass_fare__drop': ZERO,
    'ct__pclass_fare__encode': MeanEncoder(ignore_format=True, unseen='encode'),
    'ct__pclass_fare__rare': RareLabelEncoder(ignore_format=True, n_categories=2),
    'ct__sex__drop': 'passthrough',
    'ct__sex__encode': OrdinalEncoder(encoding_method='arbitrary', ignore_format=True, unseen='encode'),
    'ct__sex_fare__drop': ZERO,
    'ct__sex_fare__encode': MeanEncoder(ignore_format=True, unseen='encode'),
    'ct__sex_fare__rare': RareLabelEncoder(ignore_format=True, n_categories=2, tol=0.03),
    'ct__sex_pclass__drop': ZERO,
    'ct__sex_pclass__encode': CountFrequencyEncoder(encoding_method='frequency', ignore_format=True, unseen='encode'),
    'ct__sibsp__drop': ZERO,
    'ct__title__drop': 'passthrough',
    'ct__title__encode': MeanEncoder(ignore_format=True, smoothing=10, unseen='encode'),
    'ct__title__rare': 'passthrough',
    'embarked_imputer': CategoricalImputer(ignore_format=True, imputation_method='frequent', variables=['embarked']),
    'fare_imputer': MeanMedianImputer(variables=['fare']),
    'sampler': SMOTEENN(enn=EditedNearestNeighbours(n_jobs=-1), random_state=0, smote=SMOTE(random_state=0)),
    'scaler': MinMaxScaler(),
}

RF_PREP = {
    'ct__age__drop': ZERO,
    'ct__age__outlier': Winsorizer(fold=3, tail='both'),
    'ct__age_bin__drop': 'passthrough',
    'ct__age_bin__encode': MeanEncoder(ignore_format=True, unseen='encode'),
    'ct__age_pclass__drop': 'passthrough',
    'ct__age_pclass__encode': CountFrequencyEncoder(encoding_method='frequency', ignore_format=True, unseen='encode'),
    'ct__age_pclass__rare': RareLabelEncoder(ignore_format=True, n_categories=2),
    'ct__cabin_cat__drop': 'passthrough',
    'ct__cabin_cat__encode': CountFrequencyEncoder(encoding_method='frequency', ignore_format=True, unseen='encode'),
    'ct__cabin_cat__rare': RareLabelEncoder(ignore_format=True, n_categories=2, tol=0.03),
    'ct__cabin_na': 'passthrough',
    'ct__embarked__drop': 'passthrough',
    'ct__embarked__encode': OrdinalEncoder(encoding_method='arbitrary', ignore_format=True, unseen='encode'),
    'ct__embarked_pclass__drop': 'passthrough',
    'ct__embarked_pclass__encode': CountFrequencyEncoder(encoding_method='frequency', ignore_format=True, unseen='encode'),
    'ct__family_size__drop': 'passthrough',
    'ct__family_size__outlier': Winsorizer(capping_method='iqr', fold=1.5),
    'ct__fare_bin__drop': ZERO,
    'ct__fare_bin__encode': OrdinalEncoder(ignore_format=True, unseen='encode'),
    'ct__fare_per_person__drop': 'passthrough',
    'ct__fare_per_person__outlier': Winsorizer(capping_method='quantiles', fold=0.01, tail='both'),
    'ct__fare_per_person__transform': YeoJohnsonTransformer(),
    'ct__is_alone': 'drop',
    'ct__parch__drop': 'passthrough',
    'ct__pclass__drop': ZERO,
    'ct__pclass__encode': OrdinalEncoder(ignore_format=True, unseen='encode'),
    'ct__pclass_fare__drop': 'passthrough',
    'ct__pclass_fare__encode': CatBoostEncoder(random_state=0),
    'ct__pclass_fare__rare': RareLabelEncoder(ignore_format=True, n_categories=2),
    'ct__sex__drop': 'passthrough',
    'ct__sex__encode': OneHotEncoder(drop_last_binary=True, ignore_format=True),
    'ct__sex_fare__drop': 'passthrough',
    'ct__sex_fare__encode': CountFrequencyEncoder(encoding_method='frequency', ignore_format=True, unseen='encode'),
    'ct__sex_fare__rare': RareLabelEncoder(ignore_format=True, n_categories=2, tol=0.03),
    'ct__sex_pclass__drop': 'passthrough',
    'ct__sex_pclass__encode': MeanEncoder(ignore_format=True, unseen='encode'),
    'ct__sibsp__drop': 'passthrough',
    'ct__title__drop': 'passthrough',
    'ct__title__encode': GLMMEncoder(binomial_target=True),
    'ct__title__rare': 'passthrough',
    'embarked_imputer': CategoricalImputer(ignore_format=True, imputation_method='frequent', variables=['embarked']),
    'fare_imputer': MeanMedianImputer(variables=['fare']),
    'sampler': SMOTEENN(enn=EditedNearestNeighbours(n_jobs=-1), random_state=0, smote=SMOTE(random_state=0)),
    'scaler': MinMaxScaler(),
}

XGB_PREP = {
    'ct__age__drop': 'passthrough',
    'ct__age__outlier': 'passthrough',
    'ct__age_bin__drop': 'passthrough',
    'ct__age_bin__encode': 'passthrough',
    'ct__age_pclass__drop': 'passthrough',
    'ct__age_pclass__encode': CountFrequencyEncoder(encoding_method='frequency', ignore_format=True, unseen='encode'),
    'ct__age_pclass__rare': RareLabelEncoder(ignore_format=True, n_categories=2),
    'ct__cabin_cat__drop': 'passthrough',
    'ct__cabin_cat__encode': CountFrequencyEncoder(encoding_method='frequency', ignore_format=True, unseen='encode'),
    'ct__cabin_cat__rare': RareLabelEncoder(ignore_format=True, n_categories=2),
    'ct__cabin_na': 'passthrough',
    'ct__embarked__drop': 'passthrough',
    'ct__embarked__encode': OrdinalEncoder(encoding_method='arbitrary', ignore_format=True, unseen='encode'),
    'ct__embarked_pclass__drop': 'passthrough',
    'ct__embarked_pclass__encode': CountFrequencyEncoder(encoding_method='frequency', ignore_format=True, unseen='encode'),
    'ct__family_size__drop': 'passthrough',
    'ct__family_size__outlier': Winsorizer(capping_method='iqr', fold=1.5),
    'ct__fare_bin__drop': 'passthrough',
    'ct__fare_bin__encode': OrdinalEncoder(ignore_format=True, unseen='encode'),
    'ct__fare_per_person__drop': 'passthrough',
    'ct__fare_per_person__outlier': 'passthrough',
    'ct__fare_per_person__transform': 'passthrough',
    'ct__is_alone': 'passthrough',
    'ct__parch__drop': 'passthrough',
    'ct__pclass__drop': 'passthrough',
    'ct__pclass__encode': 'passthrough',
    'ct__pclass_fare__drop': 'passthrough',
    'ct__pclass_fare__encode': CountFrequencyEncoder(encoding_method='frequency', ignore_format=True, unseen='encode'),
    'ct__pclass_fare__rare': RareLabelEncoder(ignore_format=True, n_categories=2),
    'ct__sex__drop': 'passthrough',
    'ct__sex__encode': OneHotEncoder(drop_last_binary=True, ignore_format=True),
    'ct__sex_fare__drop': 'passthrough',
    'ct__sex_fare__encode': CountFrequencyEncoder(encoding_method='frequency', ignore_format=True, unseen='encode'),
    'ct__sex_fare__rare': RareLabelEncoder(ignore_format=True, n_categories=2),
    'ct__sex_pclass__drop': 'passthrough',
    'ct__sex_pclass__encode': OneHotEncoder(drop_last=True, ignore_format=True),
    'ct__sibsp__drop': 'passthrough',
    'ct__title__drop': 'passthrough',
    'ct__title__encode': CountFrequencyEncoder(encoding_method='frequency', ignore_format=True, unseen='encode'),
    'ct__title__rare': 'passthrough',
    'embarked_imputer': CategoricalImputer(ignore_format=True, imputation_method='frequent', variables=['embarked']),
    'fare_imputer': MeanMedianImputer(variables=['fare']),
    'sampler': TomekLinks(n_jobs=-1),
    'scaler': StandardScaler(),
}

GB_PREP = {
    'ct__age__drop': 'passthrough',
    'ct__age__outlier': Winsorizer(fold=3, tail='both'),
    'ct__age_bin__drop': 'passthrough',
    'ct__age_bin__encode': OneHotEncoder(drop_last=True, ignore_format=True),
    'ct__age_pclass__drop': ZERO,
    'ct__age_pclass__encode': MeanEncoder(ignore_format=True, unseen='encode'),
    'ct__age_pclass__rare': RareLabelEncoder(ignore_format=True, n_categories=2),
    'ct__cabin_cat__drop': 'passthrough',
    'ct__cabin_cat__encode': CountFrequencyEncoder(encoding_method='frequency', ignore_format=True, unseen='encode'),
    'ct__cabin_cat__rare': RareLabelEncoder(ignore_format=True, n_categories=2, tol=0.03),
    'ct__cabin_na': 'passthrough',
    'ct__embarked__drop': ZERO,
    'ct__embarked__encode': OneHotEncoder(drop_last=True, ignore_format=True),
    'ct__embarked_pclass__drop': 'passthrough',
    'ct__embarked_pclass__encode': CountFrequencyEncoder(encoding_method='frequency', ignore_format=True, unseen='encode'),
    'ct__family_size__drop': ZERO,
    'ct__family_size__outlier': 'passthrough',
    'ct__fare_bin__drop': ZERO,
    'ct__fare_bin__encode': OrdinalEncoder(ignore_format=True, unseen='encode'),
    'ct__fare_per_person__drop': 'passthrough',
    'ct__fare_per_person__outlier': 'passthrough',
    'ct__fare_per_person__transform': YeoJohnsonTransformer(),
    'ct__is_alone': 'passthrough',
    'ct__parch__drop': 'passthrough',
    'ct__pclass__drop': ZERO,
    'ct__pclass__encode': OrdinalEncoder(ignore_format=True, unseen='encode'),
    'ct__pclass_fare__drop': 'passthrough',
    'ct__pclass_fare__encode': CatBoostEncoder(random_state=0),
    'ct__pclass_fare__rare': RareLabelEncoder(ignore_format=True, n_categories=2),
    'ct__sex__drop': ZERO,
    'ct__sex__encode': OrdinalEncoder(encoding_method='arbitrary', ignore_format=True, unseen='encode'),
    'ct__sex_fare__drop': ZERO,
    'ct__sex_fare__encode': OrdinalEncoder(ignore_format=True, unseen='encode'),
    'ct__sex_fare__rare': RareLabelEncoder(ignore_format=True, n_categories=2, tol=0.03),
    'ct__sex_pclass__drop': ZERO,
    'ct__sex_pclass__encode': MeanEncoder(ignore_format=True, unseen='encode'),
    'ct__sibsp__drop': 'passthrough',
    'ct__title__drop': 'passthrough',
    'ct__title__encode': OrdinalEncoder(encoding_method='arbitrary', ignore_format=True, unseen='encode'),
    'ct__title__rare': 'passthrough',
    'embarked_imputer': CategoricalImputer(ignore_format=True, imputation_method='frequent', variables=['embarked']),
    'fare_imputer': MeanMedianImputer(variables=['fare']),
    'sampler': None,
    'scaler': None,
}


#: Network preprocessing from the notebook (`params_ann`); ZERO is banned there: 22 features.
MLX_PREP = {
    'ct__age__drop': 'passthrough',
    'ct__age__outlier': 'passthrough',
    'ct__age_bin__drop': 'passthrough',
    'ct__age_bin__encode': 'passthrough',
    'ct__age_pclass__drop': 'passthrough',
    'ct__age_pclass__encode': CountFrequencyEncoder(encoding_method='frequency', ignore_format=True, unseen='encode'),
    'ct__age_pclass__rare': RareLabelEncoder(ignore_format=True, n_categories=2),
    'ct__cabin_cat__drop': 'passthrough',
    'ct__cabin_cat__encode': CountFrequencyEncoder(encoding_method='frequency', ignore_format=True, unseen='encode'),
    'ct__cabin_cat__rare': RareLabelEncoder(ignore_format=True, n_categories=2),
    'ct__cabin_na': 'passthrough',
    'ct__embarked__drop': 'passthrough',
    'ct__embarked__encode': OrdinalEncoder(encoding_method='arbitrary', ignore_format=True, unseen='encode'),
    'ct__embarked_pclass__drop': 'passthrough',
    'ct__embarked_pclass__encode': CountFrequencyEncoder(encoding_method='frequency', ignore_format=True, unseen='encode'),
    'ct__family_size__drop': 'passthrough',
    'ct__family_size__outlier': Winsorizer(capping_method='iqr', fold=1.5),
    'ct__fare_bin__drop': 'passthrough',
    'ct__fare_bin__encode': OrdinalEncoder(ignore_format=True, unseen='encode'),
    'ct__fare_per_person__drop': 'passthrough',
    'ct__fare_per_person__outlier': 'passthrough',
    'ct__fare_per_person__transform': 'passthrough',
    'ct__is_alone': 'passthrough',
    'ct__parch__drop': 'passthrough',
    'ct__pclass__drop': 'passthrough',
    'ct__pclass__encode': 'passthrough',
    'ct__pclass_fare__drop': 'passthrough',
    'ct__pclass_fare__encode': CountFrequencyEncoder(encoding_method='frequency', ignore_format=True, unseen='encode'),
    'ct__pclass_fare__rare': RareLabelEncoder(ignore_format=True, n_categories=2),
    'ct__sex__drop': 'passthrough',
    'ct__sex__encode': OneHotEncoder(drop_last_binary=True, ignore_format=True),
    'ct__sex_fare__drop': 'passthrough',
    'ct__sex_fare__encode': CountFrequencyEncoder(encoding_method='frequency', ignore_format=True, unseen='encode'),
    'ct__sex_fare__rare': RareLabelEncoder(ignore_format=True, n_categories=2),
    'ct__sex_pclass__drop': 'passthrough',
    'ct__sex_pclass__encode': OneHotEncoder(drop_last=True, ignore_format=True),
    'ct__sibsp__drop': 'passthrough',
    'ct__title__drop': 'passthrough',
    'ct__title__encode': CountFrequencyEncoder(encoding_method='frequency', ignore_format=True, unseen='encode'),
    'ct__title__rare': 'passthrough',
    'embarked_imputer': CategoricalImputer(ignore_format=True, imputation_method='frequent', variables=['embarked']),
    'fare_imputer': MeanMedianImputer(variables=['fare']),
    'sampler': None,
    'scaler': MinMaxScaler(),
}


#: logreg plus the ``ct__poly`` slot (Yeo-Johnson, then products and squares):
#: 19 features -> 29. Loses to plain logreg on CV, see README.
LOGREG_POLY_PREP = dict(
    LOGREG_PREP,
    ct__poly=Pipeline(
        [
            ("transform", PowerTransformer(method="yeo-johnson")),
            ("expand", Interactions(interaction_only=False)),
        ]
    ),
)


#: SVM was not searched in the notebook; this config was tuned separately. Its
#: CV is optimistic: params were picked on the same 891 rows they are scored on.
SVM_PREP = dict(LOGREG_PREP, scaler=StandardScaler())


MODELS: dict[str, Callable[[], object]] = {
    'logreg': lambda: LogisticRegression(random_state=0, C=3.242062083573112, class_weight=None, max_iter=1000, solver='saga'),
    'logreg_l1': lambda: LogisticRegression(penalty='l1', random_state=0, C=6.693862626303041, class_weight='balanced', max_iter=1000, solver='saga'),
    'rf': lambda: RandomForestClassifier(random_state=49, class_weight='balanced', max_depth=3, max_features='log2', min_samples_leaf=1, min_samples_split=12, n_estimators=410),
    'xgb': lambda: XGBClassifier(eval_metric='logloss', n_jobs=1, colsample_bytree=0.7028873755281689, gamma=0.4085225866670681, learning_rate=0.1338705927332418, max_depth=2, min_child_weight=5, n_estimators=80, random_state=10, reg_alpha=0.28801785226090976, reg_lambda=1.0958495498382252, subsample=0.8628529400684379),
    'gb': lambda: GradientBoostingClassifier(random_state=39, criterion='squared_error', learning_rate=0.25642089647899863, max_depth=2, max_features='log2', min_samples_leaf=9, min_samples_split=5, n_estimators=101, subsample=0.9636033092676406),
    # probability=True for the stack; Platt calibration makes fit ~5x slower
    'svm': lambda: SVC(kernel='poly', degree=2, C=0.0918, coef0=1.7931, gamma='scale', probability=True, random_state=0),
    'mlx': lambda: MLXClassifier(monitor='auc', random_state=0),
    # C=1.0 rather than 3.24: the optimum shifts on the expanded feature set
    'logreg_poly': lambda: LogisticRegression(C=1.0, max_iter=2000, solver='saga', random_state=0),
}

PREPROCESSING: dict[str, dict] = {
    'logreg': LOGREG_PREP,
    'logreg_l1': LOGREG_L1_PREP,
    'rf': RF_PREP,
    'xgb': XGB_PREP,
    'gb': GB_PREP,
    'svm': SVM_PREP,
    'mlx': MLX_PREP,
    'logreg_poly': LOGREG_POLY_PREP,
}

SEARCH_PROVENANCE: dict[str, str] = {
    'logreg': "notebook cell 36, n_iter=300, StratifiedKFold(6, rs=0), refit=accuracy",
    'logreg_l1': "notebook cell 41, n_iter=1000, StratifiedKFold(4, rs=0), refit=accuracy",
    'rf': "notebook cell 46, n_iter=900, StratifiedKFold(5, rs=49), refit=accuracy",
    'xgb': "notebook cells 50-51, n_iter=90, cv=5 without shuffle, scoring=accuracy",
    'gb': "notebook cell 55, n_iter=100, StratifiedKFold(5, rs=39), refit=accuracy",
    'svm': "not in the notebook; per-kernel search 2026-08-10, StratifiedKFold(5, rs=0)",
    'mlx': "notebook cells 59-62, hyperparameters not searched",
    'logreg_poly': "not in the notebook; 2026-08-10, C tuned by CV on the expanded feature set",
}


def build(name: str) -> ImbPipeline:
    if name not in MODELS:
        raise KeyError(f"unknown config {name!r}; available: {sorted(MODELS)}")

    # clone is required: configs hold live objects, otherwise two pipelines share one
    prep = {k: (clone(v) if hasattr(v, "get_params") else v) for k, v in PREPROCESSING[name].items()}

    pipe = make_pipe()
    pipe.set_params(**prep, model=MODELS[name]())
    return pipe


#: Six configs out of eight; ``logreg`` (a near-duplicate of ``logreg_poly``)
#: and ``rf`` (weakest ranker) are excluded. Membership was picked partly on the
#: test set, so this stack's test score is not an independent estimate. See README.
STACK_MEMBERS: tuple[str, ...] = ("logreg_l1", "xgb", "gb", "svm", "mlx", "logreg_poly")


def build_stack(
    members: tuple[str, ...] = STACK_MEMBERS,
    final_estimator=None,
    cv=None,
    n_jobs: int = 1,
) -> StackingClassifier:

    if final_estimator is None:
        final_estimator = LogisticRegression(C=0.1, max_iter=1000, random_state=0)
    if cv is None:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

    return StackingClassifier(
        estimators=[(name, build(name)) for name in members],
        final_estimator=final_estimator,
        cv=cv,
        stack_method="predict_proba",
        passthrough=False,   # True would also feed raw features to the meta-learner
        n_jobs=n_jobs,
    )
