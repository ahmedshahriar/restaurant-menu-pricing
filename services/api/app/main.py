from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1 import router as v1_router
from app.config import RequestLogMiddleware, setup_json_logging

ALLOWED_ORIGINS = ["*"]


# Lifespan: shared HTTP client
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.http_client = httpx.AsyncClient(timeout=httpx.Timeout(10.0))
    try:
        yield
    finally:
        await app.state.http_client.aclose()


# App factory
app = FastAPI(title="Menu Price API", version="1.0.0", lifespan=lifespan)

# logging + CORS
setup_json_logging()
app.add_middleware(RequestLogMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)

# routes
app.include_router(v1_router, prefix="/api/v1")
