from __future__ import annotations

import pandas as pd
from loguru import logger

from application.utils.misc import convert_entities_to_list


# extract ingredients using NER pipeline
def extract_ingredients_series(descriptions: pd.Series, ner_pipeline) -> pd.Series:
    """Extract ingredients from a pandas Series of text descriptions using the provided NER pipeline."""

    def _extract(text: str) -> list[str]:
        ner_entity_result = ner_pipeline(text, aggregation_strategy="simple")
        return convert_entities_to_list(text, ner_entity_result)

    try:
        return descriptions.progress_apply(_extract)
    except Exception as e:
        logger.warning("Progress bar failed, falling back to standard apply: {}", e)
        return descriptions.apply(_extract)


# === Attach Cost of Living Index to DataFrame ===
def attach_cost_index(df: pd.DataFrame, df_cost_index: pd.DataFrame) -> pd.DataFrame:
    """Attach cost of living index to the dataframe based on city and state_id."""
    out = pd.merge(
        df,
        df_cost_index[["state_id", "city", "cost_of_living_index"]],
        how="left",
        on=["city", "state_id"],
    ).copy()
    out = out[out.city != "layton"].copy()
    out.reset_index(drop=True, inplace=True)
    return out


# === DataFrame Utilities ===


def merge_density(df_restaurant_with_address: pd.DataFrame, df_density: pd.DataFrame) -> pd.DataFrame:
    """Merge city density data into the restaurant dataframe based on city and state_id."""
    logger.info("Merging city density data")
    temp_df = df_density[["city", "density", "state_id"]].apply(lambda x: x.astype(str).str.lower().str.strip())
    out = pd.merge(df_restaurant_with_address, temp_df, how="left", on=["city", "state_id"]).copy()
    before = len(out)
    out = out.dropna(subset=["density"]).copy()
    out["density"] = out.density.astype("int32")
    logger.info("Density merge: kept {} / {} rows with density", len(out), before)
    return out


def filter_to_top_states(df_res_density: pd.DataFrame, states):
    """Filter the dataframe to only include restaurants in the specified states."""
    states = tuple(s.lower() for s in states)
    logger.info("Filtering out states not in {}", states)
    return df_res_density[df_res_density.state_id.isin(states)].copy()


def load_states_name_dict(df_states: pd.DataFrame) -> dict:
    """Load a mapping from state abbreviations to full state names."""
    drop_cols = [c for c in df_states.columns if c.lower().startswith("unnamed")]
    if drop_cols:
        df_states.drop(columns=drop_cols, inplace=True)
    df_states.Abbreviation = df_states.Abbreviation.map(str.lower)
    return df_states.set_index("Abbreviation")["State"].to_dict()
