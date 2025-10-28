import asyncio
import json
import os
from time import time
from typing import Any

import httpx

from app.config import settings

# optional (used only if AML_BEARER_TOKEN is not provided)
try:
    # Auth: UAMI in prod (set AZURE_CLIENT_ID), Azure CLI/SP locally
    from azure.identity import DefaultAzureCredential, ManagedIdentityCredential
except Exception:
    ManagedIdentityCredential = DefaultAzureCredential = None  # type: ignore

_AML_SCOPE = "https://ml.azure.com/.default"


class AzureMLInference:
    def __init__(self, client: httpx.AsyncClient):
        self.client = client
        self.uri = str(settings.SCORING_URI)
        self.deployment = getattr(settings, "AML_DEPLOYMENT", None)
        self.timeout = settings.REQUEST_TIMEOUT

        # prefer injected bearer (local/dev): `export AML_BEARER_TOKEN=...`
        self._static_token: str | None = os.getenv("AML_BEARER_TOKEN")

        # fallback for Azure (UAMI) or local az login via DefaultAzureCredential
        self._cred = None
        if not self._static_token:
            if ManagedIdentityCredential is None or DefaultAzureCredential is None:
                raise RuntimeError("azure-identity is required if AML_BEARER_TOKEN is not provided.")
            client_id = os.getenv("AZURE_CLIENT_ID")  # set on Container App for UAMI
            self._cred = (
                ManagedIdentityCredential(client_id=client_id)
                if client_id
                else DefaultAzureCredential(exclude_interactive_browser_credential=True)
            )

        # tiny cache to avoid fetching token every request
        self._tok: str | None = None
        self._exp: int | None = None
        self._lock = asyncio.Lock()

    async def _get_token(self) -> str:
        if self._static_token:
            return self._static_token
        # fetch once (or when expired)
        async with self._lock:
            if not self._tok or not self._exp or time() > (self._exp - 120):
                loop = asyncio.get_running_loop()
                at = await loop.run_in_executor(None, self._cred.get_token, _AMLSCOPE := _AML_SCOPE)
                self._tok, self._exp = at.token, at.expires_on
            return self._tok

    async def _headers(self) -> dict:
        h = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {await self._get_token()}",
        }
        if self.deployment:
            h["azureml-model-deployment"] = self.deployment
        return h

    async def score(self, payload: dict) -> dict[str, Any]:
        _headers = await self._headers()
        resp = await self.client.post(self.uri, content=json.dumps(payload), headers=_headers, timeout=self.timeout)
        if resp.status_code == 401 and not self._static_token:  # refresh once if MI token expired mid-flight
            self._tok = self._exp = None
            resp = await self.client.post(
                self.uri, content=json.dumps(payload), headers=await self._headers(), timeout=self.timeout
            )
        resp.raise_for_status()
        return resp.json()
