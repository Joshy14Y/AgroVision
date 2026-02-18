from pathlib import Path
from typing import Literal

import torch
from pydantic import DirectoryPath, FilePath, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_ignore_missing=True,
        extra="ignore",
    )

    APP_ENV: str = "development"

    HOST: str = "0.0.0.0"
    PORT: int = 8000
    RELOAD: bool = False

    MLFLOW_SERVER_URI: str
    MLFLOW_MODEL_URI: str

    ALLOWED_ORIGINS: list[str]

    PROJECT_ROOT_DIR_PATH: DirectoryPath = Path(__file__).parent.parent.resolve()
    ARTIFACTS_DIR_NAME: str = "artifacts"
    CLASSES_FILENAME: str = "classes.json"
    STATS_FILENAME: str = "normalization_stats.json"
    INPUT_SIZE: tuple = (224, 224)

    @computed_field
    def ARTIFACTS_DIR_PATH(self) -> DirectoryPath:
        return self.PROJECT_ROOT_DIR_PATH / self.ARTIFACTS_DIR_NAME

    @computed_field
    def CLASSES_FILE_PATH(self) -> FilePath:
        return self.ARTIFACTS_DIR_PATH / self.CLASSES_FILENAME

    @computed_field
    def NORMALIZATION_STATS_FILE_PATH(self) -> FilePath:
        return self.ARTIFACTS_DIR_PATH / self.STATS_FILENAME

    @computed_field
    def DEVICE(self) -> Literal["cuda", "cpu"]:
        return "cuda" if torch.cuda.is_available() else "cpu"


settings = Settings()
