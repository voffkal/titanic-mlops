from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from app.api import router
from app.config import Settings
from app.model_loader import load_model_bundle


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # loading here, not per request: a broken artifact kills startup
        app.state.bundle = load_model_bundle(settings.artifacts_dir)
        yield

    app = FastAPI(
        title=settings.project_name,
        lifespan=lifespan,
        openapi_url=f"{settings.api_v1_prefix}/openapi.json",
    )
    app.state.settings = settings
    app.include_router(router, prefix=settings.api_v1_prefix)

    @app.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        return RedirectResponse(url="/docs")

    return app


app = create_app()
