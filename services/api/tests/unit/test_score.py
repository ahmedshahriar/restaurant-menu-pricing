# services/api/tests/unit/test_score.py
import sys
from pathlib import Path

import pytest
from app.api.v1 import score as score_module
from app.main import app
from fastapi.testclient import TestClient

# todo better way to handle imports?
# --- Make 'app' importable whether pytest runs from repo root or services/api ---
API_DIR = Path(__file__).resolve().parents[2]  # -> .../services/api
sys.path.insert(0, str(API_DIR))


# ----- dependency override (avoid using app.state.http_client) -----
class DummyAML:
    async def score(self, payload: dict):
        return [9.5]


def override_dep():
    return DummyAML()


@pytest.fixture(autouse=True)
def _override_deps():
    app.dependency_overrides[score_module.get_inference] = override_dep
    yield
    app.dependency_overrides.clear()


COLUMNS = [
    "price_range",
    "state_id",
    "city",
    "density",
    "category",
    "ingredients",
    "cost_of_living_index",
]
VALID_ROW = [
    "cheap",
    "Wisconsin",
    "appleton",
    1156.0,
    "Salads",
    ["lettuce", "tomato"],
    89.5,
]
REQUEST_BODY = {"input_data": {"columns": COLUMNS, "data": [VALID_ROW]}}


def test_score_success():
    with TestClient(app) as client:  # runs lifespan
        resp = client.post("/api/v1/score", json=REQUEST_BODY)
        assert resp.status_code == 200, resp.text
        assert resp.json().get("result") == [9.5]


def test_score_validation_error():
    with TestClient(app) as client:
        bad = {"input_data": {"columns": COLUMNS, "data": [["bad", 123]]}}
        resp = client.post("/api/v1/score", json=bad)
        assert resp.status_code in (400, 422)
