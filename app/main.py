"""FastAPI entrypoint for the Decision Intelligence Copilot."""

from fastapi import FastAPI

from app.api.routes import router
from app.config import get_settings
from app.utils.logging import configure_logging

configure_logging()
settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    description="AI-powered decision intelligence API for business data analysis.",
    version="1.0.0",
)

app.include_router(router)


@app.get("/health")
def health() -> dict[str, str]:
    """Simple service health check."""

    return {"status": "ok"}

