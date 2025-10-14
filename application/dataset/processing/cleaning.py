from __future__ import annotations

import html

import pandas as pd
from bs4 import BeautifulSoup
from loguru import logger

from application.utils.misc import unescape_html


def preprocess_menu(df_menu: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the menu DataFrame by cleaning text fields, handling missing values, and formatting prices."""
    logger.info("Preprocessing menus...")
    df = df_menu.copy()
    df["description"] = df.description.str.strip()
    before = len(df)
    df.dropna(subset=["description"], inplace=True)

    df["category"] = df.category.map(unescape_html).str.strip().str.normalize("NFKD")
    df["description"] = df.description.map(unescape_html).str.strip().str.normalize("NFKD")

    df = df[df.description.str.len() > 0].drop_duplicates()

    # Coerce numeric price (remove currency suffixes like "USD")
    df["price"] = df["price"].replace({"USD": ""}, regex=True).astype(float)
    df = df[df["price"] != 0]

    # Remove the meta-category
    df = df[df.category != "Picked for you"].copy()
    logger.info("Menus: {} -> {} after cleaning", before, len(df))
    return df


def sync_restaurants_and_menus(df_restaurant: pd.DataFrame, df_menu: pd.DataFrame):
    """Ensure that restaurants and menus are synchronized, removing any restaurants without menus and vice versa."""
    logger.info("Syncing restaurants and menus...")
    df_res = df_restaurant.copy()
    df_mnu = df_menu.copy()
    df_res = df_res.dropna(subset=["price_range", "full_address"])  # keep only those we can use
    before_res, before_mnu = len(df_res), len(df_menu)

    no_menu_res_list = set(df_res.id.unique()) - set(df_mnu.restaurant_id.unique())
    df_res = df_res.loc[~df_res["id"].isin(no_menu_res_list)].copy()

    no_res_menu_list = set(df_mnu.restaurant_id.unique()) - set(df_res.id.unique())
    df_mnu = df_mnu.loc[~df_mnu["restaurant_id"].isin(no_res_menu_list)].copy()
    logger.info(
        "Restaurants: {} -> {}, Menus: {} -> {} after syncing",
        before_res,
        len(df_res),
        before_mnu,
        len(df_mnu),
    )
    return df_res, df_mnu


def build_address_fields(df_restaurant: pd.DataFrame) -> pd.DataFrame:
    """Extract city and state_id from the full_address field in the restaurant DataFrame."""
    logger.info("Extracting city/state_id from full_address...")
    df = df_restaurant.copy()
    df = df[df.full_address.str.split(",").apply(lambda x: len(x[-2].strip())) == 2].copy()
    has_city = df.full_address.str.strip().str.lower().str.split(",").apply(lambda x: len(x[-3])) != 0
    df = df[has_city].copy()
    df["city"] = df.full_address.str.strip().str.lower().str.split(",").str[-3].str.strip()
    df["state_id"] = df.full_address.str.strip().str.lower().str.split(",").str[-2].str.strip()
    logger.info("Address fields added: {} rows", len(df))
    return df


def remove_price_outliers_iqr(df: pd.DataFrame, price_col: str = "price", whisker: float = 1.5) -> pd.DataFrame:
    """Remove outliers from the price column using the IQR method."""
    q1 = df[price_col].quantile(0.25)
    q3 = df[price_col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - whisker * iqr
    upper = q3 + whisker * iqr
    logger.info("Removed {} outliers using IQR ({}).", price_col, whisker)
    return df[(df[price_col] >= lower) & (df[price_col] <= upper)].copy()


def clean_ingredients_column(df: pd.DataFrame, col: str = "ingredients") -> pd.DataFrame:
    """
    Clean the ingredients column by removing
    empty lists, lowercasing, deduplicating, stripping whitespace, and unescaping HTML.
    Requires BeautifulSoup and html for HTML parsing and unescaping.
    """
    out = df.copy()
    before = len(out)
    out = out[out[col].apply(len) > 0].copy()

    out[col] = out[col].map(lambda xs: list(map(lambda y: y.lower(), xs)))
    out[col] = out[col].map(lambda xs: list(set(xs)))
    out[col] = out[col].map(lambda xs: list(map(lambda y: y.strip(), xs)))
    out[col] = out[col].map(lambda xs: list(map(lambda y: BeautifulSoup(y, "html.parser").get_text(strip=True), xs)))
    out[col] = out[col].map(lambda xs: list(map(lambda y: html.unescape(y).strip(), xs)))
    logger.info("Ingredients cleaned: {} -> {} rows", before, len(out))
    return out


def normalize_price_range(df: pd.DataFrame, col: str = "price_range") -> pd.DataFrame:
    """Normalize the price_range column from symbols to categorical labels."""
    out = df.copy()
    mapping = {"$": "cheap", "$$": "moderate"}
    out[col] = out[col].apply(lambda x: mapping.get(x, "expensive"))
    return out
