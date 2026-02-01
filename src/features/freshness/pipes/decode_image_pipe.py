import io

from fastapi import HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError


async def decode_image_pipe(file: UploadFile) -> Image.Image:
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400, detail="File must be an image."
        )

    content = await file.read()

    try:
        return Image.open(io.BytesIO(content))
    except UnidentifiedImageError:
        raise HTTPException(
            status_code=400,
            detail="Could not decode image. Corrupted or unsupported format.",
        ) from None
