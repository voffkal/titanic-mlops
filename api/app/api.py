import sklearn
from fastapi import APIRouter, HTTPException, Request

from app import __version__ as api_version
from app.schemas import Health, MultiplePassengers, PredictionResults
from titanic_model import __version__ as model_version
from titanic_model.predict import make_prediction

router = APIRouter()


@router.get("/health", response_model=Health)
def health(request: Request) -> Health:
    manifest = request.app.state.bundle.manifest
    return Health(
        status="ok",
        api_version=api_version,
        model_version=manifest["model_version"],
        sklearn_version=sklearn.__version__,
        cv_accuracy=manifest["metrics"]["cv"]["accuracy_mean"],
    )


@router.post("/predict", response_model=PredictionResults)
def predict(payload: MultiplePassengers, request: Request) -> PredictionResults:
    records = [p.model_dump() for p in payload.inputs]  # field names, not aliases
    try:
        result = make_prediction(
            records, artifacts_dir=request.app.state.settings.artifacts_dir
        )
    except ValueError as exc:
        # input and pipeline problems are the client's fault: 422, not 500
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return PredictionResults(
        predictions=result["predictions"],
        probabilities=result["probabilities"],
        model_version=model_version,
    )
