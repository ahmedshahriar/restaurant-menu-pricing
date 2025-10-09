from .schema import ModelSchema, schema
from .transformers import build_preprocessor

# convenient module-level constants (pulled from schema)
NUMERIC_COLS = list(schema.numeric)
CATEGORICAL_COLS = list(schema.categorical)
TEXT_COLS = list(schema.text)
TARGET_COL = schema.target
DATA_SPLIT_COL = schema.data_split_col

__all__ = [
    "build_preprocessor",
    "ModelSchema",
    "schema",
    "NUMERIC_COLS",
    "CATEGORICAL_COLS",
    "TEXT_COLS",
    "TARGET_COL",
    "DATA_SPLIT_COL",
]
