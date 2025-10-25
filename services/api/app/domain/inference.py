import json
from typing import Any

import httpx

from app.config.settings import settings


class AzureMLInference:
    def __init__(self, client: httpx.AsyncClient):
        self.client = client
        self.uri = str(settings.SCORING_URI)
        self.api_key = settings.AML_API_KEY
        self.deployment = getattr(settings, "AML_DEPLOYMENT", None)
        self.timeout = settings.REQUEST_TIMEOUT

    def _headers(self) -> dict:
        h = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        if self.deployment:
            h["azureml-model-deployment"] = self.deployment
        return h

    async def score(self, payload: dict) -> dict[str, Any]:
        resp = await self.client.post(
            self.uri,
            content=json.dumps(payload),
            headers=self._headers(),
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()
