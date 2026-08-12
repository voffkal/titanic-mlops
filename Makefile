ARTIFACTS = model_artifacts

SPACE_DIR ?= ../titanic-hf-space

.PHONY: help train test metrics submit image serve space stop clean

## list the available commands
help:
	@echo "titanic-mlops:"
	@echo ""
	@grep -E '^## ' -A1 $(MAKEFILE_LIST) | \
	  grep -v '^--' | \
	  awk '/^## / { desc=substr($$0,4); next } \
	       /^[a-zA-Z_-]+:/ { sub(/:.*/,"",$$1); printf "  \033[36mmake %-8s\033[0m %s\n", $$1, desc }'

## train the release stack (artifacts -> model_artifacts/)
train:
	uv run titanic-train

## run the test suite
test:
	uv run pytest

## report on the trained model
metrics:
	@test -f $(ARTIFACTS)/metrics.json || \
	  { echo "no $(ARTIFACTS)/metrics.json — train first: make train"; exit 1; }
	@jq -r '\
	  def pct: .*1000 | round / 10;\
	  def bar(frac): (frac*20 | round) as $$n | ("█"*$$n) + ("░"*(20-$$n));\
	  "=== Titanic — \(.config) v\(.model_version) ===",\
	  "",\
	  "trained  \(.trained_at[0:10]) · commit \(.git_commit) · \(.dataset.n_rows) rows (hash \(.dataset.hash))",\
	  "env      py \(.python) · sklearn \(.libraries."scikit-learn") · xgboost \(.libraries.xgboost) · mlx \(.libraries.mlx)",\
	  (if .stack_members then "members  " + (.stack_members | join(", ")) else empty end),\
	  "",\
	  "── accuracy, higher is better ──",\
	  "CV        \(.metrics.cv.accuracy_mean | pct)%  \(.metrics.cv.accuracy_mean | bar(.))  ± \(.metrics.cv.accuracy_std | pct) over \(.metrics.cv.n_folds) folds",\
	  "OOF       \(.metrics.oof.accuracy | pct)%  \(.metrics.oof.accuracy | bar(.))",\
	  "in-sample \(.metrics.in_sample.accuracy | pct)%  \(.metrics.in_sample.accuracy | bar(.))  (optimistic)",\
	  "baseline  \(.baseline.train_accuracy | pct)%  \(.baseline.train_accuracy | bar(.))  \(.baseline.rule)",\
	  "",\
	  "── ranking ──",\
	  "CV roc_auc \(.metrics.cv.roc_auc_mean | pct)% ± \(.metrics.cv.roc_auc_std | pct)",\
	  "",\
	  "lift over baseline +\(.baseline.cv_lift | pct)pp · train-CV gap \(.metrics.train_cv_gap | pct)pp"\
	' $(ARTIFACTS)/metrics.json

## predict Kaggle test.csv -> submission.csv
submit:
	uv run titanic-train --submission

## build the API image (trains first if there is no artifact)
image:
	@test -f $(ARTIFACTS)/metrics.json || $(MAKE) train
	docker compose build api

## run the API locally (http://localhost:8002/docs)
serve: image
	docker compose up -d api
	@echo "API: http://localhost:8002/docs"

## copy deploy/hf-space into the Space clone, commit and push
space:
	@test -d $(SPACE_DIR)/.git || { echo "no Space clone at $(SPACE_DIR)"; exit 1; }
	cp deploy/hf-space/app.py deploy/hf-space/requirements.txt deploy/hf-space/README.md $(SPACE_DIR)/
	@cd $(SPACE_DIR) && git add -A && \
	  if git diff --staged --quiet; then \
	    echo "the Space already matches deploy/hf-space"; \
	  else \
	    git commit -m "sync from $$(git -C $(CURDIR) rev-parse --short HEAD)" && git push; \
	  fi

## stop the containers
stop:
	docker compose down

## stop, drop the image and the artifacts
clean: stop
	rm -rf $(ARTIFACTS) dist
	-docker image rm titanic-api:local 2>/dev/null || true
	@echo "removed: $(ARTIFACTS)/, dist/, the local image."
