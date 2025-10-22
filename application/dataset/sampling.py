from __future__ import annotations

import time

import numpy as np
import pandas as pd
from loguru import logger

from application.networks import NERModelSingleton

from . import io, processing
from .config import Config

_DEFAULT_CFG = Config()


def load_base_frames(cfg: Config):
    t0 = time.time()
    # Load datasets

    # selected columns from restaurants.csv
    # zip code varies in states, hence removed, we can get location from coordinates or full address
    # search position doesn't give any context in price prediction, hence removed
    # name is removed as it's not helpful in price prediction
    res_cols = ["id", "score", "ratings", "category", "price_range", "full_address", "lat", "lng"]

    df_restaurant = io.load_kaggle_dataset(
        cfg.RESTAURANTS_DS,
        cfg.RESTAURANTS_FILE,
        pandas_kwargs={
            "skipinitialspace": True,
            "usecols": res_cols,
            "converters": {
                "id": np.int32,
                "position": np.int32,
                # 'score': lambda x: np.float32(x) if x != None else np.nan,
                # 'score':np.float32, 'ratings':np.float32,
                "lat": np.float32,
                "lng": np.float32,
            },
        },
    )
    df_menu = io.load_kaggle_dataset(cfg.MENUS_DS, cfg.MENUS_FILE, pandas_kwargs={"skipinitialspace": True})
    df_index = io.load_kaggle_dataset(cfg.INDEX_DS, cfg.INDEX_FILE, pandas_kwargs={"skipinitialspace": True})
    df_density = io.load_kaggle_dataset(cfg.DENSITY_DS, cfg.DENSITY_FILE, pandas_kwargs={"skipinitialspace": True})
    df_states = io.load_kaggle_dataset(cfg.STATES_DS, cfg.STATES_FILE, pandas_kwargs={"skipinitialspace": True})
    logger.info(
        "Loaded: restaurants={} rows, menus={} rows, index={} rows, density={} rows in {:.2f}s",
        len(df_restaurant),
        len(df_menu),
        len(df_index),
        len(df_density),
        time.time() - t0,
    )
    return df_restaurant, df_menu, df_index, df_density, df_states


def generate_training_sample(cfg: Config = _DEFAULT_CFG) -> pd.DataFrame:
    # Load
    df_restaurant, df_menu_raw, df_index, df_density, df_states = load_base_frames(cfg)

    # Preprocess + sync
    df_menu = processing.preprocess_menu(df_menu_raw)
    df_restaurant_synced, df_menu_synced = processing.sync_restaurants_and_menus(df_restaurant, df_menu)

    # Address + density
    df_addr = processing.build_address_fields(df_restaurant_synced)

    # Intersect only cities present in density
    density_cities = set(df_density.city.str.strip().str.lower())
    df_addr = df_addr[df_addr.city.isin(density_cities)].copy()
    df_res_density = processing.merge_density(df_addr, df_density)

    # Filter to selected states
    df_top_state = processing.filter_to_top_states(df_res_density, cfg.top_states_filter)
    # generate states name dict
    states_name_dict = processing.load_states_name_dict(df_states)

    # Category/city selection
    df_res_ext, top_categories = processing.compute_top_categories(
        df_menu_synced, df_top_state, cfg.top_categories_per_city
    )

    # Category -> cities
    top_cities = processing.pick_top_cities(top_categories, cfg.focus_categories, cfg.top_cities_per_state)

    # Final base frame
    df_final = processing.build_final_menu_frame(df_menu_synced, df_res_ext, top_cities, cfg.focus_categories)

    # Price outliers + NER ingredients
    df_sampled = processing.remove_price_outliers_iqr(df_final, price_col="price", whisker=1.5)
    logger.info("Remaining rows: {}", len(df_sampled))

    ner_pipeline = NERModelSingleton().get_pipeline()
    df_sampled["ingredients"] = processing.extract_ingredients_series(df_sampled["description"], ner_pipeline)
    df_sampled = df_sampled.drop(columns=["description"])  # drop after extracting
    df_sampled = processing.clean_ingredients_column(df_sampled, col="ingredients")

    # Enrichment & normalize
    if "restaurant_id" in df_sampled.columns:
        df_sampled = df_sampled.drop(columns=["restaurant_id"])  # mirrors original

    df_sampled = processing.attach_cost_index(df_sampled, df_index)
    logger.info("Cost-of-living index attached")

    # Normalize price_range to buckets
    df_sampled = processing.normalize_price_range(df_sampled, col="price_range")
    logger.info("Normalized price_range to buckets (cheap/moderate/expensive)")

    # Replace state_id with full state names
    df_sampled["state_id"] = df_sampled["state_id"].replace(states_name_dict)
    logger.info("Replaced state_id with full state names")

    # Final clean-up
    df_sampled.dropna(inplace=True)

    # Persist
    df_sampled.to_csv(cfg.FINAL_SAMPLED_DATA_PATH, index=False)
    logger.success("Wrote {} rows -> {}", len(df_sampled), cfg.FINAL_SAMPLED_DATA_PATH)
    return df_sampled
