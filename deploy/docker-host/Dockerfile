# The Space runs the exact image docker.yml published, so what answers here is the
# artifact from the model release — not a rebuild that could differ from it.
# The image already runs as uid 1000 and writes nothing at runtime, which is what
# Spaces requires; app_port in README.md points at the port its CMD listens on.
FROM ghcr.io/voffkal/titanic-api:api-v0.1.0
