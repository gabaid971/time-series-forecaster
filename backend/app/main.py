"""
Time Series Forecaster API.

Run locally:   uv run python -m app          (auto-reload)
Production:    uvicorn app.main:app          (see Dockerfile)
"""

import logging

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import router
from app.config import settings

logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)

app = FastAPI(title="Time Series Forecaster API", version="2.0.0")

# CORS: only browsers on these origins may call the API (does not stop non-browser clients)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_origin_regex=settings.allowed_origin_regex,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.middleware("http")
async def limit_body_size(request: Request, call_next):
    """Reject oversized requests before reading/parsing their body."""
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > settings.max_body_mb * 1024 * 1024:
        return JSONResponse(
            status_code=413,
            content={"detail": f"Request too large (max {settings.max_body_mb:g} MB)"}
        )
    return await call_next(request)


app.include_router(router)
