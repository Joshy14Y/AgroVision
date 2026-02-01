from pydantic import BaseModel


class FreshnessResDto(BaseModel):
    label: str
    confidence: float
    class_id: int
