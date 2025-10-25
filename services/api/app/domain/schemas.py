from typing import Any

from pydantic import BaseModel, ConfigDict, field_validator

EXPECTED_COLUMNS = [
    "price_range",
    "state_id",
    "city",
    "density",
    "category",
    "ingredients",
    "cost_of_living_index",
]


class ScoringInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    columns: list[str]
    data: list[list[Any]]

    @field_validator("columns")
    @classmethod
    def check_columns(cls, v: list[str]) -> list[str]:
        if v != EXPECTED_COLUMNS:
            raise ValueError(f"columns must equal {EXPECTED_COLUMNS}")
        return v

    @field_validator("data")
    @classmethod
    def check_rows(cls, rows: list[list[Any]]) -> list[list[Any]]:
        for i, row in enumerate(rows):
            if len(row) != len(EXPECTED_COLUMNS):
                raise ValueError(f"row {i} length {len(row)} != {len(EXPECTED_COLUMNS)}")
            if not isinstance(row[0], str):
                raise ValueError(f"row {i} price_range must be string")
            if not isinstance(row[1], str):
                raise ValueError(f"row {i} state_id must be string")
            if not isinstance(row[2], str):
                raise ValueError(f"row {i} city must be string")
            try:
                row[3] = float(row[3])
            except Exception as e:
                raise ValueError(f"row {i} density must be numeric") from e
            if not isinstance(row[4], str):
                raise ValueError(f"row {i} category must be string")
            if not (isinstance(row[5], list) and all(isinstance(x, str) for x in row[5])):
                raise ValueError(f"row {i} ingredients must be list[str]")
            try:
                row[6] = float(row[6])
            except Exception as e:
                raise ValueError(f"row {i} cost_of_living_index must be numeric") from e
        return rows


class ScoreRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    input_data: ScoringInput


class ScoreResponse(BaseModel):
    # AML returns arbitrary JSON; keep open for MVP
    result: Any
