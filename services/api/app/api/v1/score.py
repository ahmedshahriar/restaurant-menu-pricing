import httpx
from fastapi import APIRouter, Depends, HTTPException, Request

from app.domain.inference import AzureMLInference
from app.domain.schemas import ScoreRequest, ScoreResponse

router = APIRouter()


def get_inference(request: Request) -> AzureMLInference:
    client = request.app.state.http_client  # set in lifespan
    return AzureMLInference(client)


# module-level dependency marker to avoid calling Depends in argument defaults (Bandit B008)
get_inference_dep = Depends(get_inference)


@router.post("/score", response_model=ScoreResponse)
async def score(req: ScoreRequest, svc: AzureMLInference = get_inference_dep):
    try:
        raw = await svc.score(req.model_dump())
        if isinstance(raw, dict) and "result" in raw:
            return ScoreResponse(result=raw["result"])
        return ScoreResponse(result=raw)
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text) from e
