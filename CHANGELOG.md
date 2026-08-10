# Changelog

Model and service history, loosely following
[Keep a Changelog](https://keepachangelog.com/en/1.0.0/). Every notable decision
carries its motivation: what was wrong, and why this route was taken. Numbers are
CV under the one protocol the project uses —
`RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=0)` — and, where
stated, the Kaggle public leaderboard.

The two packages version independently: `model-v*` tags a trained artifact,
`api-v*` tags the service that serves it. Headings below are model versions;
API changes appear under the release they shipped with. `api-v0.1.0` covers
everything the service has done so far, since it had not been released before.
Push `model-v*` first — `docker.yml` builds against the newest model release.

## [0.2.0] — companions

**Leaderboard 0.79665**, 333 of 418. CV accuracy 0.8465 ±0.0184, roc_auc 0.8863.

### Model

- **`GroupSurvival`**: the share of surviving women and boys who share a
  passenger's surname or ticket, plus a flag for whether any companion is known.
  Two `ct` slots, addressed by all eight frozen configs.
  - *Why.* v0.1 beat the "all women survive" baseline by six passengers out of
    418, and its overrides of that baseline were right 57% of the time — barely
    a coin flip. The errors were not spread out: third-class women (72 of them,
    surviving at exactly 50%) and first-class men (45) held most of them. In the
    first pocket the companions' fate splits the group 16% against 100%.
  - *Why not family size.* Size was already in the pipeline and is a different
    quantity — a count of companions, not their outcome. Fixing family size at 3
    still leaves the companions' fate splitting third-class women 9% against
    100%. The information is orthogonal, which is why a model that had the count
    still scored 0.353 on the hard cases.
  - *How it was settled.* Paired over the 25 shared folds: +0.0153, 24 folds won
    of 25, p ≈ 0. That is the only change in this project to clear a paired test
    — roughly 70,000 search fits, eight configs and the stack all sat inside σ.
    A hard woman-child-group override was measured against a smoothed rate and
    won (+0.0153 against +0.0103), so the statistic excludes adult men, who died
    almost regardless and only dilute the group.
  - *Result.* All eight configs gained, +0.0128 to +0.0226. On the leaderboard
    the feature moved 17 of the 418 predictions and was right on 12 of them.

- The release stays the six-model stack although `svm` now leads it on accuracy,
  0.8498 against 0.8465.
  - *Why.* The margin is 0.0033 against σ 0.023, a tie by the project's own rule.
    The stack keeps the best AUC of the set (0.8863 against 0.8677) and the
    tightest fold-to-fold spread (±0.0184 against ±0.0236).

### Fixed

- **`coerce_to_schema` cast every numeric column to `float64`.** The interaction
  features are built by string concatenation, so `pclass` rendered as `"3.0"` and
  `sex_pclass`, `age_pclass` and `pclass_fare` carried categories no encoder had
  been fitted on. Every request served by the API took that path: 23 of the 418
  test labels differed from the batch path, which was right all along. Integers
  are now restored after the required-column check.
  - *How it surfaced.* Comparing `submission.csv` against the same rows posted
    through `/predict`. Neither number looked wrong on its own.

### Notes

- **The model is now transductive.** The feature reads other rows' labels. Not
  test leakage — the labels are in `train.csv` — but it works because families
  were split across train and test, and it is a lookup of relatives' fates rather
  than a prediction from a passenger's own attributes. Deliberate, and recorded
  in the README.
- **The artifact carries a table.** `fit` stores surname and ticket statistics,
  so a single request is answered partly from that table rather than purely from
  its own contents.
- **The train-CV gap widened from +0.0137 to +0.0525, which is benign.** At
  predict time the feature looks a training passenger up in its own table and
  sees that passenger's own outcome. CV and OOF are unaffected — OOF 0.8462
  against CV 0.8465 — because there the statistics come from the training fold.
- Leak check by permutation: on shuffled labels the pipeline scores 0.58–0.60
  against a majority-class rate of 0.6162.

### Housekeeping

- **Both published releases carry an earlier wording of the MLX caveat.** A tag
  reproduces the workflow as it stood, and the correction landed on `main` after
  both were cut. The note as published describes inference divergence rather
  than the training divergence that actually applies to a CI-built artifact.
- **The corrected wording no longer quotes a label count.** The v0.1 measurement
  (2 of 418) does not hold for v0.2: the CPU-trained release artifact and the
  Metal-trained local one agree on all 418 labels. Baking a model-dependent
  constant into a generic template is how notes go stale; it now states the
  operational point instead — the artifact and its submission belong together.
- **`docker.yml` assumed `model_artifacts/` was absent after checkout.** It is
  not: `metrics.json` and `input_schema.json` are tracked, so `mkdir` failed and
  the release download would have refused to overwrite them. `mkdir -p` plus
  `--clobber`. Inherited from houseprice, where that directory is fully ignored.
- Prose moved out of the code into the README: `SEARCH_PROVENANCE` (a dictionary
  of English sentences read only by a test asserting they were long enough) and
  `GroupSurvival`'s fifteen-line docstring. What stays is the one-line guards.
- Releases are annotated tags carrying their leaderboard score. `git tag -n1`
  lists them; `git show model-v0.1.0:model_artifacts/metrics.json` reaches an
  older manifest.

## [0.1.0] — first working pipeline

**Leaderboard 0.77990**, 326 of 418. CV accuracy 0.8337 ±0.0239, roc_auc 0.8734.

### Model

- The research notebook ported into a package: slot-based `ColumnTransformer`
  where each column carries a small pipeline (`rare` / `encode` / `transform` /
  `outlier` / `drop`), so a frozen search winner is a flat dict of 44 keys rather
  than a nested estimator.
- Eight configs frozen from the notebook's stored `best_params_`, parsed rather
  than transcribed.
  - *Why not re-run the searches.* They cost roughly 70,000 fits and bought
    nothing: the frozen `logreg` and a plain `LogisticRegression(C=1.0)` land on
    the same score. Re-running would spend hours to reproduce a number already
    known.
- Release model is a six-member `StackingClassifier`; `logreg` and `rf` are
  excluded as a near-duplicate and the weakest ranker.
- Age imputed by title (`Master` 3.5, `Mr` 30, `Mrs` 35) rather than globally
  (28), owned by `AgeFeatures` so that two imputation strategies cannot compete.

### Infrastructure

- Training CLI writing an artifact, a manifest with CV / OOF / in-sample metrics,
  an input schema, and optionally a submission. Quality gate at 0.78 CV accuracy.
- FastAPI service over `titanic_model.predict`, with inference living in the
  model package rather than the service, so a batch job takes the same path.
  Fail-fast startup on sklearn or model version skew, and on an implausible smoke
  prediction.
- Two-stage Docker image, compose file, make targets, and three workflows: CI,
  model release on `model-v*`, multi-arch image push on `api-v*`.
- 142 tests, none of which need the Kaggle files: the suite generates a frame
  with the same schema and missingness, and a Kaggle-shaped 891/418 pair that
  lets `train()` run end to end in a temporary directory.
  - *Why.* The data cannot be redistributed. Without synthetic fixtures, CI needs
    a Kaggle token and a download on every run; with them it needs neither.

### Fixed

- **`mlx` on Linux ships only the bindings.** `libmlx.so` lives in `mlx-cpu`,
  pulled in solely by an extra, so the container died at import. `mlx[cpu]`
  covers both platforms — the extra is a no-op on macOS.
- **An all-missing text column arrives as `float64`.** feature-engine's imputers
  downcast it while imputing another column, and `.str` refuses a float column.
  77% of passengers have no cabin, so this broke most single-passenger requests
  while training never saw it.
- **`MLXClassifier(BaseEstimator, ClassifierMixin)` made `roc_auc` return NaN.**
  Since sklearn 1.6 the estimator type comes from `__sklearn_tags__` and the MRO
  never reached the mixin, so scoring failed silently rather than raising.
- **A name with no title and a cabin with no deck were accepted.** `"string"`
  produced the out-of-distribution title the required-field rule existed to
  prevent, and the string `"NaN"` became deck N, which the ship did not have.
  Both are 422 now, checked against all 1309 official records.
- **`ZERO` built from lambdas** passed training and CV, then failed at
  `joblib.dump`. Named module-level functions instead.
- **`build()` returned pipelines sharing live estimators.** The frozen configs
  hold instances, so without `clone` two pipelines fit one scaler.
