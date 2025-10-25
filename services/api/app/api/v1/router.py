from fastapi import APIRouter

from .score import router as score_router

router = APIRouter()


@router.get("/healthz")
def healthz():
    return {"status": "ok"}


@router.get("/readyz")
def readyz():
    return {"ready": True}


router.include_router(score_router, tags=["score"])
