# Running the published image

A Hugging Face Docker Space needs a PRO subscription, so the Space in
`../hf-space` uses the Gradio SDK instead. This Dockerfile stays because it
works unchanged on any host that takes a container — Cloud Run, Render, Fly, or
a Docker Space once the account is on PRO.

```bash
docker run --rm -p 8002:8002 ghcr.io/voffkal/titanic-api:api-v0.1.0
```

It runs as uid 1000 and writes nothing at runtime, which is what most managed
platforms require.
