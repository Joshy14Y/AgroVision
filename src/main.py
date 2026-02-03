from fastapi import FastAPI

from src.config import settings
from src.features.freshness.freshness_controller import router as freshness_router

app = FastAPI(
    title="AgroVision API",
    description="Fruit Freshness Classification API",
    version="1.0.0",
)

app.include_router(freshness_router)


@app.get("/")
def health_check():
    return {
        "app": "AgroVision API",
        "status": "online",
        "device": settings.DEVICE,
        "docs": "/docs",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.RELOAD,
    )
