from pydantic import BaseModel


class Health(BaseModel):
    status: str
    api_version: str
    model_version: str
    sklearn_version: str
    cv_accuracy: float  # what quality is actually deployed, straight from the manifest
