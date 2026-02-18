from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.config import settings
from src.features.freshness.freshness_controller import router as freshness_router
from src.features.freshness.freshness_service import get_freshness_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"🚀 Starting AgroVision API on device: {settings.DEVICE}")
    try:
        get_freshness_service()
        print("✅ Model and artifacts loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading the model: {e}")

    yield
    print("👋 Shutting down AgroVision API")


app = FastAPI(
    title="AgroVision API",
    description="Fruit Freshness Classification API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
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
