from fastapi import APIRouter, Depends
from PIL import Image

from .dtos.freshness_res_dto import FreshnessResDto
from .freshness_service import FreshnessService, get_freshness_service
from .pipes.decode_image_pipe import decode_image_pipe

router = APIRouter(prefix="/freshness")


@router.post("/predict", response_model=FreshnessResDto)
def predict(
    image: Image.Image = Depends(decode_image_pipe),
    service: FreshnessService = Depends(get_freshness_service),
):
    return service.predict(image)
