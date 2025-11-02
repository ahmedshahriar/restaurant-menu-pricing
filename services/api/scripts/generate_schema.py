# Generate JSON schema for ScoreRequest model
# Usage (from repo root):
#   ( cd services/api && python -m scripts.generate_schema )
# This writes payload.schema.json into services/api/
import json

from app.domain import ScoreRequest

with open("payload.schema.json", "w") as f:
    json.dump(ScoreRequest.model_json_schema(), f, indent=2)
