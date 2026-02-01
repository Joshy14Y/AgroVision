from fastapi import APIRouter, Depends

from .dtos.freshness_res_dto import FreshnessResDto
from .freshness_service import service
from .pipes.decode_image_pipe import decode_image_pipe

router = APIRouter(prefix="/freshness")


@router.post("/predict", response_model=FreshnessResDto)
async def predict(image=Depends(decode_image_pipe)):
    return service.predict(image)
