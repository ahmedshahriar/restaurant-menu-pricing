import pathlib
from typing import Any

import pandas as pd
from loguru import logger
from pymongo import ASCENDING
from pymongo.collection import Collection
from tqdm import tqdm

from core import settings
from infrastructure.db.mongo import get_client


def get_collection() -> Collection:
    client = get_client()
    db = client.get_database(settings.DATABASE_NAME)
    return db[settings.DATABASE_COLLECTION]


def fetch_all_docs(
    query: dict[str, Any] | None = None, projection: dict[str, int] | None = None
) -> list[dict[str, Any]]:
    """Fetch all docs from a MongoDB collection with tqdm progress, sorted for determinism."""
    coll = get_collection()
    try:
        if not query:
            try:
                total = coll.estimated_document_count()
            except Exception:
                total = coll.count_documents({})
        else:
            total = coll.count_documents(query)

        logger.info(f"Fetching ~{total} docs from '{settings.DATABASE_NAME}.{settings.DATABASE_COLLECTION}'")
        cursor = (
            coll.find(query or {}, projection or {})
            .sort([("_id", ASCENDING)])  # <- deterministic order
            .batch_size(1500)  # <- memory-friendlier network batches
        )
        docs: list[dict[str, Any]] = []
        for doc in tqdm(cursor, total=total, desc=f"Fetching {settings.DATABASE_COLLECTION}"):
            docs.append(doc)
        logger.success(f"Fetched {len(docs)} documents.")
        return docs
    except Exception as e:
        logger.exception(f"Error fetching documents: {e}")
        raise


def build_tables(docs: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Flatten restaurants and explode menu_items into two dataframes with stable surrogate keys."""
    logger.info("Building restaurant and menu tables...")
    DROP_COLS = ["_id", "task_id", "url", "phone", "image_url"]

    if not docs:
        logger.warning("No documents found; returning empty DataFrames.")
        return pd.DataFrame(), pd.DataFrame()

    df_restaurant = pd.json_normalize(docs)

    # Stable surrogate id from _id (factorize is deterministic per run given sorted _id)
    if "_id" in df_restaurant.columns:
        rid = pd.factorize(df_restaurant["_id"])[0] + 1
        df_restaurant.insert(0, "id", rid)
    else:
        # fallback (shouldn't happen if projection kept _id)
        df_restaurant.insert(0, "id", df_restaurant.index + 1)

    # Build menu table
    if "menu_items" in df_restaurant.columns:
        exploded = df_restaurant[["id", "menu_items"]].explode("menu_items").dropna()
        if not exploded.empty:
            df_menu = pd.json_normalize(exploded["menu_items"])
            df_menu.insert(0, "restaurant_id", exploded["id"].to_numpy())
            del exploded
        else:
            df_menu = pd.DataFrame()
            logger.warning("No non-empty 'menu_items' to extract.")
    else:
        df_menu = pd.DataFrame()
        logger.warning("No 'menu_items' column found in restaurant data.")

    # Drop unneeded columns (including _id)
    df_restaurant.drop(columns=[c for c in DROP_COLS if c in df_restaurant.columns], inplace=True, errors="ignore")
    if "menu_items" in df_restaurant.columns:
        df_restaurant.drop(columns=["menu_items"], inplace=True)

    logger.success(f"Built restaurant table ({len(df_restaurant)} rows) and menu table ({len(df_menu)} rows).")
    return df_restaurant, df_menu


def save_data(df_restaurant: pd.DataFrame, df_menu: pd.DataFrame, out_dir: str, *, compress: bool = False) -> None:
    """Save restaurant and menu DataFrames to CSV files, optionally compressed."""
    logger.info(f"Saving CSV files to: {out_dir}")
    out_dir_path = pathlib.Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    r_path = out_dir_path / (settings.RESTAURANT_DATA_PATH + (".gz" if compress else ""))
    m_path = out_dir_path / (settings.MENU_DATA_PATH + (".gz" if compress else ""))

    logger.info(f"Writing restaurant data -> {r_path}")
    df_restaurant.to_csv(r_path, index=False, compression="gzip" if compress else "infer")

    if not df_menu.empty:
        logger.info(f"Writing menu data -> {m_path}")
        df_menu.to_csv(m_path, index=False, compression="gzip" if compress else "infer")
    else:
        logger.warning("No menu data to write.")
    logger.success("CSV export complete.")
