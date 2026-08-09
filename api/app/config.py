from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

from titanic_model.train import DEFAULT_ARTIFACTS_DIR


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="", protected_namespaces=())

    api_v1_prefix: str = "/api/v1"
    project_name: str = "Titanic Survival API"
    artifacts_dir: Path = DEFAULT_ARTIFACTS_DIR  # env: ARTIFACTS_DIR
