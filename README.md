# titanic-mlops

Survival classification on the Kaggle Titanic dataset: a slot-based
preprocessing pipeline, eight frozen model configs, a six-model stack, and a
reproducible artifact with a manifest.

The project grew out of the research notebook `workbook/Titanic_code.ipynb`
(~70,000 hyperparameter-search fits). Those searches are not repeated here —
their winners are frozen in `builders.py`.

---

## Quick start

```bash
uv sync

# data (Kaggle rules require you to download it yourself)
kaggle competitions download -c titanic -p workbook/data/
unzip -o workbook/data/titanic.zip -d workbook/data/raw/

# train the release model and write a submission
uv run titanic-train --submission

# single model, quick run
uv run titanic-train --config logreg --cv-repeats 1

# tests (no Kaggle data needed)
uv run pytest

# serve the model
uv run uvicorn --app-dir api app.main:app --port 8002   # docs at /docs
```

The data path can be overridden with `TITANIC_RAW_DIR`.

A uv workspace with two members: `model/` builds `titanic-model` (the pipeline
and the training run), `api/` builds `titanic-api` (a FastAPI service over it).
One lock file covers both, so they cannot drift apart on a shared dependency.
The model version has a single source of truth in `titanic_model/__init__.py` —
hatch reads it, and the manifest and artifact filename use the same value.

---

## Evaluation protocol

Everything below rests on one rule, so it comes first.

**The only metric is cross-validation over the 891 labelled training rows**:
`RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=0)`, 25 folds.
Model comparison, feature choices and hyperparameters are settled on it alone.

**The 418 test rows carry no labels and are never scored locally.** Kaggle does
not publish them, so any locally computed number on that split would rest on
labels obtained outside the competition. The pipeline only ever predicts on
them, to produce `submission.csv`. The leaderboard is the sole source of truth
there, one submission per release.

No permanent holdout is carved out of the training rows either: over 891 rows
it would cost a quarter of the data, and its confidence interval would be
wider than any difference being measured.

A note on reading CV numbers: σ across folds is ≈0.023, so differences below
about 0.02 are noise. Several models below sit inside that band and should be
treated as tied, not ranked.

### What the leaderboard said

| release | leaderboard | of 418 | over the sex rule |
|---|---|---|---|
| sex rule | 0.76555 | 320 | — |
| v0.1.0 | 0.77990 | 326 | +6 |
| **v0.2.0** | **0.79665** | **333** | **+13** |

**The companions feature doubled what the project is worth**, from six
passengers to thirteen. More usefully, it is the first change here that
survived the whole route — paired t-test on CV, then release, then an external
score:

```
CV predicted    +0.0128
leaderboard gave +0.01675
```

The gain came in slightly above the CV estimate, in the direction predicted:
inside a fold the group statistics are built from 713 rows, at test time from
all 891. The CV→LB gap itself barely moved, −0.0538 to −0.0499, so the offset is
systematic rather than something the feature broke.

**v0.1.0 scored 0.77990** — 326 of 418 passengers. The "all women survive"
baseline scores 0.76555, or 320. Everything in this repository was worth **six
passengers** before v0.2.

### What the two releases actually bet on

Both releases only ever differ from the sex rule on a handful of passengers, and
the whole score is decided there.

| | overrides of the sex rule | right | wrong | hit rate |
|---|---|---|---|---|
| v0.1.0 | 42 | 24 | 18 | 57% |
| v0.2.0 | 33 | 23 | 10 | 70% |

v0.2 bets **less** and wins **more**: nine fewer calls against the baseline, one
fewer correct, eight fewer wrong.

The companions feature itself is measurable in isolation. Switching it off
through its `ZERO` slot and refitting the stack reproduces v0.1.0 exactly — CV
0.8337 ±0.0239, the same figure to four places — and changes 17 of the 418
predictions: 13 passengers it calls saved, 4 it calls lost. On a flipped label
exactly one of the two releases is right, so 333 − 326 = 2g − 17 pins the
feature at **g = 12 right of 17, a 71% hit rate**, against 57% for the model's
own overrides in v0.1.

CV said 0.8337 and the leaderboard says 0.7799, a gap of −0.054. The train-CV
gap is only +0.0137, so this is not the model memorising its training set. On
418 rows σ is 0.0203, which puts the 95% interval at [0.740, 0.820]: the
interval is twice as wide as the entire spread between the eight configs. A
leaderboard position in this competition is mostly a draw from that interval.

---

## Data

Official competition files: `train.csv` (891 labelled rows), `test.csv` (418
unlabelled), `gender_submission.csv` (format sample).

`data.py` lowercases Kaggle's CamelCase (`Pclass`, `SibSp`) once at the entry
point, so the rest of the code never deals with casing and config keys
(`ct__age_bin__encode`) match the notebook's.

The loader validates row counts and columns, which catches a hand-made file
swapped in for an official one.

---

## Architecture

```
DataFrame -> missing_indicators -> imputers -> feature generators
          -> ColumnTransformer (slots) -> drop_correlated -> drop_constant
          -> scaler -> sampler -> model
```

### Slots

Each column gets a small `Pipeline` with named steps (`rare` / `encode` /
`transform` / `outlier` / `drop`), defaulting to `'passthrough'`. A config
flips an individual slot through a key like `ct__title__encode`, so
preprocessing is specified together with the model.

The `ct__poly` slot defaults to `'drop'` rather than `'passthrough'` —
otherwise its columns would be emitted a second time and every frozen config
would shift.

The mechanism is exact, not approximate: setting `ct__group_rate__drop` to
`ZERO` and dropping `ct__group_known` reproduces the v0.1.0 stack at 0.8337
±0.0239, the released figure to four decimal places. An ablation costs two
dictionary keys and no code.

### Step order matters

* `missing_indicators` before the imputers, or the missing flag is always zero;
* `title` before `age_feats`: age is imputed with the median of its title;
* `group_feats` after `title` and before `fare_feats`: it needs `Master` to spot
  the boys, and `fare_feats` drops `ticket`, which is one of its two group keys;
* `fare_imputer` before `fare_feats`: quantile edges are computed on non-null
  `fare`, otherwise a missing value lands in the fallback category;
* `ct` after all generators: it addresses columns that do not exist in the raw
  file (`title`, `age_bin`, `cabin_cat`).

### Age imputation

Owned by `AgeFeatures`, using the median per title. There is no separate
`age_imputer` step: two strategies in sequence would conflict and the earlier
one would silently win.

Train medians: `Master` 3.5, `Miss` 21, `Mr` 30, `Mrs` 35, `Officer` 50 —
against a global median of 28. This is semantically sounder than grouping by
class, but it does not move the metric: of 40 `Master` rows, 36 already have an
age, so the dramatic correction touches four rows out of 891.

### Companions (v0.2)

`GroupSurvival` emits two columns: the share of surviving women and boys who
share a passenger's surname or ticket, and a flag for whether any companion is
known at all.

This is **not** family size. Size is already in the pipeline and is a count of
companions; this is their outcome. Among third-class women — the pocket where
everything else gives a coin flip — size barely moves the rate (52–62% for
sizes 1 to 4), while the companions' fate splits the same group 16% against
100%, and keeps splitting it at a fixed family size:

| family size 3, third-class women | survived |
|---|---|
| companions died (n=11) | 9% |
| companions survived (n=6) | 100% |

Adult men are excluded from the statistic: they died almost regardless, so
their labels only dilute the group. `Master` marks a boy, so no age is needed.

Two consequences, both deliberate:

* **The model is transductive.** The values come from other rows' labels. Not
  test leakage — the labels are in `train.csv` — but it works because families
  were split across train and test, and it is a lookup of relatives' fates
  rather than a prediction from a passenger's own attributes.
* **The artifact carries a table.** `fit` stores surname and ticket statistics,
  so the service answers a single request partly from that table. Predicting on
  a passenger who was in the training set returns a confident answer for the
  same reason, which is why `in_sample` in the manifest is further from CV than
  it used to be.

`fit_transform` leaves each row's own label out, per key, since the ticket may
be missing where the surname is not. Without it the label would flow into the
coefficients. Verified by permutation: with shuffled labels the pipeline scores
0.58–0.60 against a majority-class rate of 0.6162.

### Column names are lost after `ct`

ColumnTransformer returns an ndarray, so `drop_correlated` sees `x0…xN`
instead of `title_Mr`, `age`. Any logic that needs names must live **inside**
`ct` as a slot over a named column list — that is how `ct__poly` is wired.

---

## Models

Eight configs in `builders.MODELS`, all winners of the notebook searches except
`svm` and `logreg_poly`, which were tuned later.

Where each frozen config came from — note that every search used its own
cross-validation, which is the reason they could not be compared until they were
re-run under one protocol:

| Config | Search |
|---|---|
| `logreg` | notebook cell 36, `n_iter=300`, `StratifiedKFold(6, rs=0)`, `refit=accuracy` |
| `logreg_l1` | cell 41, `n_iter=1000`, `StratifiedKFold(4, rs=0)` |
| `rf` | cell 46, `n_iter=900`, `StratifiedKFold(5, rs=49)` |
| `xgb` | cells 50–51, `n_iter=90`, `cv=5` without shuffle |
| `gb` | cell 55, `n_iter=100`, `StratifiedKFold(5, rs=39)` |
| `mlx` | cells 59–62, hyperparameters not searched |
| `svm` | not in the notebook; per-kernel search, `StratifiedKFold(5, rs=0)` |
| `logreg_poly` | not in the notebook; `C` tuned by CV on the expanded feature set |

Numbers below come from the pinned environment (`uv sync`, see `uv.lock`).
Seven of the eight are bit-reproducible across environments; a different
xgboost major shifts `xgb` and the stack by about 0.0005.

| Config | What it is | CV accuracy | CV AUC |
|---|---|---|---|
| `svm` | SVC poly degree 2, `C=0.092` | 0.8498 ±0.0236 | 0.8677 |
| `logreg_l1` | LogisticRegression L1, `C=6.69`, balanced | 0.8480 ±0.0228 | 0.8763 |
| **stack of 6** | logreg_l1 + xgb + gb + svm + mlx + logreg_poly | **0.8465** ±0.0184 | **0.8863** |
| `logreg_poly` | `logreg` plus `ct__poly` (Yeo-Johnson → products, squares) | 0.8465 ±0.0230 | 0.8848 |
| `logreg` | LogisticRegression L2, `C=3.24`, saga | 0.8449 ±0.0235 | 0.8888 |
| `xgb` | XGBoost, depth 2, 80 trees | 0.8393 ±0.0219 | 0.8833 |
| `rf` | RandomForest, depth 3, 410 trees | 0.8388 ±0.0239 | 0.8712 |
| `gb` | GradientBoosting, depth 2, 101 trees | 0.8375 ±0.0243 | 0.8756 |
| `mlx` | MLP 26→64→32→2 on MLX, early stopping on AUC | 0.8359 ±0.0246 | 0.8847 |
| — | "all women survive" baseline (train) | 0.7868 | — |

The corridor spans 0.836–0.850 at σ ≈ 0.023, and the ranking inside it is
still noise: eight configs across four algorithm families, plus a neural net,
plus an ensemble, land on the same function with pairwise prediction
correlations of 0.91–0.98. What moved every one of them by the same +0.013 to
+0.023 was the companions feature, not a change of algorithm.

Most of the residual error is Bayes noise. Passengers with identical `sex`,
`pclass`, `title` and `age_bin` met different fates, because the outcome hinged
on which deck someone was on and whether they reached a lifeboat, and neither is
in the data. The v0.1 version of this paragraph claimed all of it was noise;
`GroupSurvival` proved that wrong for the part that families explain.

The release model is the six-member stack. `logreg` and `rf` are excluded —
`logreg` is a near-duplicate of `logreg_poly` (same frozen preprocessing, one
extra slot) and `rf` is the weakest ranker of the set.

Since v0.2 `svm` is nominally ahead of the stack on accuracy, by 0.0033 against
σ 0.023 — a tie by the rule stated above. The stack keeps the release for two
reasons that are not accuracy: the best AUC of the set (0.8863, against 0.8677
for `svm`), and the tightest fold-to-fold spread (±0.0184 against ±0.0236),
which is what one expects from averaging and is worth more than a third of a
noise unit.

### How the frozen configs were obtained

Parsed out of the `repr()` of `search_*.best_params_` in the notebook's stored
outputs — 45 keys per model, which cannot be transcribed by hand without
errors. Two fixes were needed:

* in `repr()` output, `ZERO` shows up as `FunctionTransformer(func=<function
  <lambda> at 0x3825dab60>)` — a memory address, not valid Python;
* the `age_imputer` key was dropped, since that step no longer exists.

---

## Tried and rejected

### The `ZERO` mechanism is not feature selection, it is a lottery

`ct__*__drop` slots set to `ZERO` zero out a feature. The sets barely overlap
across models: GradientBoosting switched off `sex`, LogReg switched off
`pclass`, RandomForest switched off `age` and `pclass`, XGBoost switched off
nothing — and all of them land inside the same CV band.

The cause is feature redundancy. `title` recovers sex on **888 of 891 rows**
(99.66%), the three misses being `Officer` and `Aristocratic`. `pclass` lives
on inside `sex_pclass`, `embarked_pclass` and `pclass_fare`.

Restoring key features to the configs changes nothing: adding `pclass` back to
`logreg` grows the matrix from 19 columns to 20 and moves CV on **none of the
25 folds** — for a linear model the raw `pclass` already lies in the span of
the existing columns.

Scale of the problem: the preprocessing search space is 9.3·10¹⁴ combinations,
5.2·10⁵ of them from `ZERO` alone (19 slots). At `n_iter=300` that is 3·10⁻¹³
coverage. `best_params_` records one arbitrary member of a large equivalence
class.

### Hyperparameter search is exhausted

The frozen `logreg` (`C=3.24`, selected over 300 iterations) scores 0.8301 on
CV; a plain `LogisticRegression(C=1.0)` on the default pipeline scores 0.8316.
Roughly 70,000 fits bought nothing over the default constructor.

### Polynomial features — no

The `ct__poly` slot over `[age, fare_per_person, family_size, pclass]`. CV does
not improve anywhere: between −0.0043 and +0.0022 depending on the variant, and
AUC is unchanged at 0.8718. Whatever the products add, it is not ranking power.

Why it worked in houseprice and not here: there the features were genuinely
continuous and products were the only route to interactions. Here the
interactions are already hand-built **as categories** (`sex_pclass`,
`age_pclass`, `pclass_fare`), and a category is more flexible than a product —
it learns a value per cell of the grid instead of imposing a multiplicative
form.

The config is kept as `logreg_poly` and used inside the stack, but it is not a
release candidate on its own.

### Stacking — almost nothing

A stack of seven gained +0.0036 on CV over the best single model at σ 0.024.
Dropping `logreg` and `rf` brought the six-member stack to 0.8337, the best CV
figure of v0.1 — still only +0.0007 over `svm` alone, and it costs 310 seconds
against 4. In v0.2 `svm` overtook it on accuracy outright.

Meta-learner weights land in 0.63–0.92 with no clear leader — the signature of
correlated base models and an ensemble degenerating into averaging.

### Resampling — no

The 38/62 imbalance is mild. Every winning search config used `sampler=None`;
the one run with forced SMOTETomek was 0.018 worse.

---

## Pitfalls

The ones that cost time. Matching guards are left in the code as one-line
comments — **do not delete them during refactoring**.

| Symptom | Cause | Where |
|---|---|---|
| `joblib.dump` fails with `Can't pickle <function <lambda>>`, while training and CV run fine | `ZERO` was built from lambdas; pickle stores a function by module and name | `pipeline.py`, named `_zeros_column` / `_zero_feature_names` |
| `scoring='roc_auc'` silently returns `NaN` | base order `(BaseEstimator, ClassifierMixin)`: since sklearn 1.6 the estimator type comes from `__sklearn_tags__`, and the MRO never reaches the mixin | `mlx_mlp.py`, `(ClassifierMixin, BaseEstimator)` |
| Network is non-deterministic, validation score swings by 10 pp | three sources of randomness; `mx.random.seed` must precede the constructor (that is where `Linear` init draws from), and shuffling must use a local `default_rng` | `mlx_mlp.py` |
| `ValueError: feature names should match` after two `.fit()` calls in a row | `PREPROCESSING[name]` holds live transformer instances; without `clone` two pipelines share one object. `SVM_PREP` is built from `LOGREG_PREP`, so even different models shared state | `builders.py`, `clone` inside `build()` |
| Stack with MLX hangs or crashes at `n_jobs=-1` | loky forks processes, which clashes with the Metal context | `builders.py`, `n_jobs=1` |
| A function argument changes nothing | `make_ct(kaggle_version=...)` and `make_pipe(...)` read the global `KAGGLE`; the parameter was dead. Fixed by removing the flag | — |
| SVM search hangs | `poly` and `sigmoid` with large `C` do not converge on 891 rows; needs `max_iter` and per-kernel grids | — |
| Up to 80% of search candidates come back `NaN` | `error_score='nan'` together with a global `filterwarnings('ignore')` swallows failures | use `error_score='raise'` |
| `saga` stops converging after a new feature is added, but only in tests | the synthetic fixture gave every passenger a unique surname, so `GroupSurvival` was degenerate there: 1941 iterations against 74 on the real data. Fixed the fixture, not `max_iter` | `tests/conftest.py`, families share a surname, a ticket and a fate |
| A one-passenger request crashes on `.str` | feature-engine's imputers downcast an all-missing text column to `float64` while imputing another column | `features.py`, `astype("string")` in `TitleFeature` / `CabinCatFeature` |
| The service answers differently from the batch path — 23 labels of 418 | `coerce_to_schema` left `pclass` as `float64`, and the string interactions render it `"3.0"`, so `sex_pclass`, `age_pclass` and `pclass_fare` were unseen categories on every request | `predict.py`, integers restored after the required-column check |

One more, methodological rather than technical: in the notebook each search had
its own cross-validation — 6, 4 and 5 folds, `random_state` 0, 49, 39, and
XGBoost without shuffle at all. What was being compared was splits, not models.
A single shared CV object is the first thing to fix when porting research code.

---

## Artifacts

`train.py` writes into `model_artifacts/`:

* `titanic_model-<version>.joblib` — the fitted pipeline;
* `metrics.json` — manifest: CV, out-of-fold and in-sample metrics with
  confusion matrices, the baseline and lift, a data hash, positive rate,
  versions of ten libraries, Python, platform, commit and timestamp;
* `input_schema.json` — columns and dtypes for validation at the API boundary;
* `submission.csv` — with `--submission`.

Quality gate: `MIN_CV_ACCURACY = 0.78`. Below it, nothing is written.

### Reaching an older release

What changed between versions and why — [CHANGELOG.md](CHANGELOG.md). Every
release is an annotated tag carrying its leaderboard score:

```bash
git tag -n1                                          # what shipped, and what it scored
git show model-v0.1.0:model_artifacts/metrics.json   # that release's manifest
git diff model-v0.1.0 model-v0.2.0 -- model/         # what changed between them
```

The `.joblib` itself is not in git — binaries bloat history irreversibly, so it
ships as a release asset. For a tag that predates the remote, rebuild it:
`git checkout model-v0.1.0 && uv run titanic-train`. Training is deterministic,
so the result is bit-identical to the artifact that scored 0.77990, apart from
the MLX member if the backend differs.

The `in_sample` block exists only as an overfitting check: the gap against CV
is +0.0525 for the v0.2 stack, up from +0.0137 in v0.1. That jump is expected
rather than alarming — at predict time `GroupSurvival` looks a training
passenger up in its own table and sees that passenger's own outcome, which
inflates in-sample and nothing else. CV and OOF are unaffected, since there the
statistics come from the training fold only.

---

## Serving

Inference lives in `titanic_model/predict.py`, not in the service: `api/app/` is
transport only. Anything that runs the model — a batch job, a notebook, a second
service — goes through `make_prediction` and gets the same type frontier.

```
GET  /api/v1/health    versions and the deployed CV accuracy
POST /api/v1/predict   {"inputs": [passenger, ...]} -> labels + P(survived)
```

```bash
curl -s localhost:8002/api/v1/predict -H 'content-type: application/json' -d '{"inputs":[
  {"Pclass":1,"Name":"Cumings, Mrs. John Bradley","Sex":"female","Age":38,
   "SibSp":1,"Parch":0,"Ticket":"PC 17599","Fare":71.2833,"Cabin":"C85","Embarked":"C"},
  {"Pclass":3,"Name":"Braund, Mr. Owen Harris","Sex":"male","Age":22,
   "SibSp":1,"Parch":0,"Ticket":"A/5 21171","Fare":7.25}
]}'
# {"predictions":[1,0],"probabilities":[0.898311,0.124421],"model_version":"0.1.0"}
```

**Field names are the Kaggle ones**, so a row of `test.csv` posts unchanged;
the lowercase names the model uses are accepted too. Five fields are required —
`Pclass`, `Name`, `Sex`, `SibSp`, `Parch`. The first four of those have no
imputer downstream. `Name` is a policy choice: it survives a `null`, but the
title degrades to a category absent from training and the probability moves by
about 0.17, so the service refuses rather than answering out of distribution.
`Age`, `Fare`, `Ticket`, `Cabin` and `Embarked` may be omitted.

**Startup is fail-fast.** The service loads the artifact once in the lifespan
hook and refuses to come up if the manifest's sklearn version differs from the
runtime, if the artifact version differs from the package, or if a smoke
prediction on a known passenger comes out implausible. The alternative —
unpickling into a different sklearn and serving quietly wrong numbers — is
worse than not starting.

Client mistakes return 422, never 500: pydantic rejects an unknown `Pclass`, a
negative fare or an age of 200, and `make_prediction`'s `ValueError` is mapped
to the same code.

---

## Container

```bash
make serve    # trains if needed, builds, runs on http://localhost:8002/docs
make stop
```

Two stages: `uv sync --locked --no-dev --no-editable` builds a `.venv` of wheels,
the runtime image copies only that plus `model_artifacts/`, so no sources ship.
`WORKDIR` is identical in both stages because uv bakes an absolute path into the
console-script shebangs. Image 1.3 GB, of which the artifact is 762 KB — the rest
is the scientific stack.

Two native dependencies have to be present or the artifact will not unpickle:

* **libgomp1** — the OpenMP runtime xgboost links at import;
* **mlx-cpu** — on Linux the `mlx` wheel is bindings only, and `libmlx.so` lives
  in the `cpu` extra. Missing it fails with `libmlx.so: cannot open shared object
  file` at import, long before any request. Hence `mlx[cpu]` in the dependencies:
  the extra is a no-op on macOS, where `mlx-metal` arrives anyway.

**The backend changes the numbers slightly.** The released artifact is trained on
Metal, and the container runs MLX on CPU. Across all 418 test passengers:

| | |
|---|---|
| largest probability shift | 9.16e-5 |
| median shift | 2.57e-5 |
| labels that differ | 0 of 418 |
| closest probability to the 0.5 threshold | 0.0030, i.e. 33× the largest shift |

Float32 addition order differs between the two backends. It has never flipped a
label here, but the margin is 33×, not 1000×, so a passenger sitting exactly on
the threshold could in principle land differently. Only the MLX member is
affected; the other five are float64 sklearn estimators.

**Training on the other backend is a different matter.** The figures above are
one artifact evaluated twice. Train the stack with MLX pinned to CPU and the
same divergence accumulates over 300 epochs of SGD:

| | inference, one artifact | training, two artifacts |
|---|---|---|
| largest probability shift | 9.16e-5 | 2.29e-2 |
| median shift | 2.57e-5 | 2.75e-3 |
| labels that differ | 0 of 418 | **2 of 418** |
| CV accuracy | — | 0.8339 vs 0.8337 |

Equal quality, different model. Two passengers out of 418 is 0.005 on the
leaderboard, which is visible. That is why `release-model.yml` produces
`submission.csv` as a release asset: the leaderboard number has to describe the
artifact that was published, not a local rebuild of it.

Those figures are v0.1. Measured again on the real releases, v0.2 turned out
insensitive: the artifact CI trained on CPU and the one trained here on Metal
agree on **all 418 labels**, CV 0.8456 ±0.0215 against 0.8465 ±0.0184. The
companions feature is decisive for the passengers it covers, so the MLX member's
drift no longer reaches the stack's output — the same robustness that narrowed
the fold-to-fold spread. The release still ships its own submission, because
that property was measured after the fact and is not guaranteed to survive the
next change to the model.

---

## CI

| Workflow | Trigger | What it does |
|---|---|---|
| `ci.yml` | push to main, PR | ruff, the full suite, then builds the image and probes `/health` and `/predict` in a running container |
| `release-model.yml` | tag `model-v*` | fetches the competition data, runs the gate, trains, publishes a release with the artifact, schema and manifest |
| `docker.yml` | tag `api-v*` | takes the artifact from the newest `model-v*` release, builds linux/amd64 + linux/arm64, pushes to ghcr |

**`ci.yml` needs no secret and downloads no data** — that is the payoff of the
synthetic fixtures. The image job trains a throwaway model on generated data
purely so the container has something to load: it proves the native stack
imports on Linux, which is the failure mode that actually happens, and says
nothing about model quality.

Only `release-model.yml` needs `KAGGLE_ACCESS_TOKEN`, because only it trains the
real thing. The quality gate inside `train()` runs there too, so a pipeline that
regressed below 0.78 CV accuracy fails the release instead of shipping.

Both architectures are buildable because mlx publishes `manylinux_2_35` wheels
for x86_64 and aarch64, and the base image is bookworm (glibc 2.36).

---

## Tests

`uv run pytest` — 245 tests, about 14 seconds, no Kaggle data required.

The competition files cannot be redistributed, so `conftest.py` generates a
frame with the same schema, the same missingness and a sex-driven target.
`write_raw_dir` produces a Kaggle-shaped 891/418 pair, which lets `train()` run
end to end in a temporary directory — artifacts, manifest, submission and the
quality gate included. A handful of tests do use the real files and skip
themselves when those are absent.

What the suite is actually pinning down:

| Area | What would otherwise break silently |
|---|---|
| `test_data` | wrong row count or a missing column accepted as the official file; a submission Kaggle rejects |
| `test_features` | age imputed from the global median; test-set quantiles recomputed on transform; `clone` losing a custom title map |
| `test_pipeline` | the `poly` slot emitting four columns twice; `ZERO` built from lambdas, which fails only at `joblib.dump`; steps reordered so age is imputed before titles exist |
| `test_builders` | two pipelines sharing one scaler; a config key that no longer names a slot; a stack member forked across processes with an MLX context |
| `test_mlx` | base-class order making `roc_auc` return NaN instead of raising; an unseeded fit |
| `test_train` | a local path in the manifest; the quality gate writing artifacts anyway |
| `test_predict` | an all-missing text column inferred as float64; a required field silently imputed |
| `api/…` | a schema that drifts from the artifact; version skew starting anyway; a 500 where 422 belongs |

Warnings are errors (`filterwarnings = ["error", ...]`) with two documented
exceptions from pinned dependencies.

---

## Layout

```
pyproject.toml       uv workspace root, ruff and pytest config
uv.lock              pinned resolution
CHANGELOG.md         what changed between releases, and why
Makefile             train, metrics, image, serve
Dockerfile           two-stage build of the api image
compose.yaml         local run with a healthcheck
.github/workflows/   ci, release-model, docker
LICENSE              MIT
model/
  pyproject.toml     the titanic-model package
  titanic_model/
    data.py          loading, name normalisation, submission
    features.py      7 feature transformers + Interactions + NaNReporter
    pipeline.py      slot-based ColumnTransformer + ImbPipeline
    mlx_mlp.py       MLX network + sklearn wrapper
    builders.py      frozen configs, build(), build_stack()
    train.py         training, metrics, artifacts
    predict.py       artifact loading, type frontier, make_prediction
  tests/             tests on synthetic Titanic-shaped data
api/
  pyproject.toml     the titanic-api package
  app/
    main.py          create_app, lifespan
    api.py           /health and /predict
    schemas/         the wire contract
    model_loader.py  fail-fast startup checks
    tests/
workbook/            out of git: notebooks, data, experiment notes, search runs
```

Hardware: MacBook Pro M5 Max, 48 GB. MLX runs on Metal; `mlx` wheels exist for
linux/amd64 (glibc ≥ 2.35, i.e. bookworm or ubuntu 22.04+), but Metal is not
available there — inference falls back to CPU, which is immaterial for a
network of three matrix multiplications.
