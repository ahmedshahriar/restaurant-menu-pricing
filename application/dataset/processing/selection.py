from __future__ import annotations

from collections.abc import Iterable

import pandas as pd
from loguru import logger


def compute_top_categories(df_menu: pd.DataFrame, df_top_state_restaurants: pd.DataFrame, top_n_per_city: int):
    """Compute the top menu categories per city from the menu and restaurant dataframes."""
    logger.info("Computing top categories per city...")
    df_res_ext = pd.merge(
        df_menu[["restaurant_id", "category"]],
        df_top_state_restaurants,
        left_on="restaurant_id",
        right_on="id",
        how="left",
    ).copy()

    df_res_ext.rename(columns={"category_x": "menu_category", "category_y": "restaurant_category"}, inplace=True)
    if "id" in df_res_ext.columns:
        df_res_ext.drop(["id"], axis=1, inplace=True)

    category_counts = df_res_ext.groupby(["state_id", "city", "menu_category"]).size().reset_index(name="count")
    sorted_categories = category_counts.sort_values("count", ascending=False)
    top_categories = sorted_categories.groupby(["state_id", "city"]).head(top_n_per_city)
    logger.info("Top categories computed: {} rows", len(top_categories))
    return df_res_ext, top_categories


def pick_top_cities(top_categories: pd.DataFrame, focus_categories: Iterable[str], top_cities_per_state: int):
    """Select the top cities per state based on the focus categories."""
    logger.info("Selecting top cities per state for focus categories {}...", ", ".join(focus_categories))
    top_filtered_categories = top_categories[top_categories.menu_category.isin(tuple(focus_categories))].copy()
    state_city_counts = top_filtered_categories.groupby(["state_id", "city"])["count"].sum().reset_index()
    state_city_counts = state_city_counts.sort_values(["state_id", "count"], ascending=[True, False])
    top_cities = state_city_counts.groupby("state_id").head(top_cities_per_state).copy()
    logger.info("Picked {} top cities across states", len(top_cities))
    return top_cities


def build_final_menu_frame(
    df_menu: pd.DataFrame, df_res_ext: pd.DataFrame, top_cities: pd.DataFrame, focus_categories: Iterable[str]
) -> pd.DataFrame:
    """Build the final menu dataframe filtered by top cities and focus categories."""
    logger.info("Building final menu frame...")
    df_res_ext_nonnull = df_res_ext.dropna(subset=["price_range"]).copy()
    res_filtered_df = pd.merge(
        df_res_ext_nonnull,
        top_cities[["state_id", "city"]],
        on=["state_id", "city"],
    ).copy()
    res_filtered_df.drop_duplicates(inplace=True)

    df_final = pd.merge(
        res_filtered_df[["restaurant_id", "price_range", "state_id", "city", "density"]],
        df_menu[["restaurant_id", "category", "description", "price"]],
        on=["restaurant_id"],
    ).copy()

    df_final = df_final[df_final.category.isin(tuple(focus_categories))].copy()
    df_final.drop_duplicates(inplace=True)
    logger.info("Final pre-NER frame: {} rows", df_final.shape[0])
    return df_final
