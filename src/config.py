from pathlib import Path
from typing import Literal

import torch
from pydantic import DirectoryPath, FilePath, computed_field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    PROJECT_ROOT_DIR_PATH: DirectoryPath = Path(__file__).parent.parent.resolve()
    models_checkpoints_dir_name: str = "models/checkpoints"
    artifacts_dir_name: str = "artifacts"
    weights_filename: str = "best_model.pt"
    classes_filename: str = "classes.json"
    stats_filename: str = "normalization_stats.json"
    INPUT_SIZE: tuple = (224, 224)

    @computed_field
    def MODELS_DIR_PATH(self) -> DirectoryPath:
        return self.PROJECT_ROOT_DIR_PATH / self.models_checkpoints_dir_name

    @computed_field
    def ARTIFACTS_DIR_PATH(self) -> DirectoryPath:
        return self.PROJECT_ROOT_DIR_PATH / self.artifacts_dir_name

    @computed_field
    def MODEL_WEIGHTS_FILE_PATH(self) -> FilePath:
        return self.MODELS_DIR_PATH / self.weights_filename

    @computed_field
    def CLASSES_FILE_PATH(self) -> FilePath:
        return self.ARTIFACTS_DIR_PATH / self.classes_filename

    @computed_field
    def NORMALIZATION_STATS_FILE_PATH(self) -> FilePath:
        return self.ARTIFACTS_DIR_PATH / self.stats_filename

    @computed_field
    def DEVICE(self) -> Literal["cuda", "cpu"]:
        return "cuda" if torch.cuda.is_available() else "cpu"


settings = Settings()
