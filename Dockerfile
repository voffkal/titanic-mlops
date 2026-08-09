FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder
# WORKDIR must match the runtime: uv sync bakes an absolute path into the
# console-script shebangs inside .venv
WORKDIR /opt/titanic
COPY . .
# --no-editable installs the workspace members as wheels, so the runtime image
# needs no sources at all
RUN uv sync --locked --no-dev --no-editable --package titanic-api

FROM python:3.12-slim AS runtime
# libgomp1: the OpenMP runtime xgboost links against at import time. The stack
# holds an XGBClassifier, so without it the artifact fails to unpickle with
# "libgomp.so.1: cannot open shared object file".
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
RUN adduser --disabled-password --gecos '' api-user
WORKDIR /opt/titanic
COPY --from=builder /opt/titanic/.venv .venv
# Artifacts enter the build context from `make image` locally, or from the
# model release in docker.yml on CI
COPY model_artifacts/ model_artifacts/
ENV PATH="/opt/titanic/.venv/bin:$PATH" \
    ARTIFACTS_DIR=/opt/titanic/model_artifacts
USER api-user
EXPOSE 8002
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s \
    CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8002/api/v1/health')"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8002"]
