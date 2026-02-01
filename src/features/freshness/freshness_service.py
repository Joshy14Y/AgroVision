import json

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

from src.config import settings

from .dtos.freshness_res_dto import FreshnessResDto


class FreshnessService:
    def __init__(self):
        self.device = settings.DEVICE
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
        self.model = models.resnet18(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, len(self.classes))

        state_dict = torch.load(
            settings.MODEL_WEIGHTS_FILE_PATH,
            map_location=self.device,
            weights_only=True,
        )

        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
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


service = FreshnessService()
