# Generate JSON schema for ScoreRequest model
# Usage (from repo root):
#   ( cd services/api && python -m scripts.generate_schema )
# This writes payload.schema.json into services/api/
import json

from app.domain import ScoreRequest

json.dump(ScoreRequest.model_json_schema(), open("payload.schema.json", "w"), indent=2)
