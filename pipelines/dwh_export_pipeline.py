import gc
from typing import Any

from application.dataset import build_tables, fetch_all_docs, save_data
from core import settings


def dwh_export_pipeline():
    """
    Exports restaurant and menu data from the data warehouse (MongoDB) to CSV files.
    - Output directory: settings.DWH_EXPORT_DIR
    - Restaurant data file: settings.RESTAURANT_DATA_PATH
    - Menu data file: settings.MENU_DATA_PATH
    """
    # empty query to fetch all documents
    query: dict[str, Any] = {}
    # keep _id to build stable ids, drop it later
    projection: dict[str, int] = {"task_id": 0, "url": 0, "phone": 0, "image_url": 0}

    docs = fetch_all_docs(query, projection)
    df_restaurant, df_menu = build_tables(docs)
    save_data(df_restaurant, df_menu, settings.DWH_EXPORT_DIR, compress=False)
    del docs, df_restaurant, df_menu  # free memory
    gc.collect()
