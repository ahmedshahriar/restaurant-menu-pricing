# run from the repo root
# python services/common/export_state_metadata.py
# This script reads the train sample dataset and exports a JSON mapping of states to their cities with metadata.
# The output JSON is saved to services/common/data/state_city_map.json
import json
from pathlib import Path

import pandas as pd

# Get repo root: go two levels up from this script
ROOT_DIR = Path(__file__).resolve().parents[2]
DATASETS_DIR = ROOT_DIR / "datasets"
COMMON_DATA_DIR = Path(__file__).resolve().parent / "data"

COMMON_DATA_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = DATASETS_DIR / "sampled-final-data.csv"
OUTPUT_PATH = COMMON_DATA_DIR / "state_city_map.json"

print(f"📂 Loading dataset: {CSV_PATH}")

# Read only the needed columns + coerce types early
df = pd.read_csv(CSV_PATH, usecols=["state_id", "city", "density", "cost_of_living_index"])

# Normalize strings and clean rows
df["state_id"] = df["state_id"].astype(str).str.strip().str.lower()
df["city"] = df["city"].astype(str).str.strip().str.lower()
df["density"] = pd.to_numeric(df["density"], errors="coerce")
df["cost_of_living_index"] = pd.to_numeric(df["cost_of_living_index"], errors="coerce")

df = df.dropna(subset=["state_id", "city", "density", "cost_of_living_index"])
df = df.drop_duplicates(subset=["state_id", "city"])  # keep first occurrence per city

# Build nested mapping (vectorized):
# state_id -> { city -> {"density": x, "cost_of_living_index": y} }
# 1) index by (state_id, city), keep only the two value cols
idxed = df.set_index(["state_id", "city"])[["density", "cost_of_living_index"]].sort_index()

# 2) group by state (level 0), convert each subframe to dict(city -> meta)
state_city_map = {state: group.droplevel(0).to_dict(orient="index") for state, group in idxed.groupby(level=0)}

with OUTPUT_PATH.open("w", encoding="utf-8") as f:
    json.dump(state_city_map, f, ensure_ascii=False, indent=2)

print(f"✅ Saved nested state_city_map.json to: {OUTPUT_PATH.resolve()}")
print(f"📊 States exported: {len(state_city_map)}")
print(f"🏙️ Total cities: {sum(len(c) for c in state_city_map.values())}")
