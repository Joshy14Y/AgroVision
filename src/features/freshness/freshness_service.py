import functools
import json

import mlflow
import torch
from PIL import Image
from torchvision import transforms

from src.config import settings

from .dtos.freshness_res_dto import FreshnessResDto


class FreshnessService:
    def __init__(self):
        self.device = settings.DEVICE
        self.mlflow_server_uri = settings.MLFLOW_SERVER_URI
        self.mlflow_model_uri = settings.MLFLOW_MODEL_URI
        self._load_artifacts()
        self._load_model()
        self._build_pipeline()

    def _load_artifacts(self):
        with open(settings.CLASSES_FILE_PATH) as f:
            self.classes: list[str] = json.load(f)

        with open(settings.NORMALIZATION_STATS_FILE_PATH) as f:
            stats = json.load(f)
            self.mean = stats["mean"]
            self.std = stats["std"]

    def _load_model(self):
        mlflow.set_tracking_uri(self.mlflow_server_uri)
        self.model = mlflow.pytorch.load_model(
            self.mlflow_model_uri, map_location=self.device
        )
        self.model.eval()

    def _build_pipeline(self):
        self.transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(settings.INPUT_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std),
            ]
        )

    def predict(self, image: Image.Image) -> FreshnessResDto:
        image = image.convert("RGB")

        img_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)

        idx = predicted_idx.item()

        return FreshnessResDto(
            label=self.classes[idx],
            confidence=round(confidence.item(), 4),
            class_id=idx,
        )



@functools.lru_cache
def get_freshness_service() -> FreshnessService:
    return FreshnessService()
