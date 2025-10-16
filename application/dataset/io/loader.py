import ast

import pandas as pd
from kagglehub import KaggleDatasetAdapter, dataset_load
from loguru import logger


def _safe_eval_ingredients(v):
    if pd.isna(v):
        return []
    try:
        parsed = ast.literal_eval(v)
        return parsed if isinstance(parsed, list) else []
    except Exception:
        return []


def load_kaggle_dataset(dataset_handle: str, dataset_path: str, pandas_kwargs: dict | None = None) -> pd.DataFrame:
    """Load a dataset from Kaggle using kagglehub with specified adapter and pandas options.
    Args:
        dataset_handle (str): The Kaggle dataset handle in the format "owner/dataset-name".
        dataset_path (str): The specific file path within the dataset to load (e.g., "data.csv").
        pandas_kwargs (dict, optional): Additional keyword arguments to pass to pandas read_csv.
    Returns:
        pd.DataFrame: The loaded dataset as a pandas DataFrame.
    """
    if pandas_kwargs is None:
        pandas_kwargs = {}
    logger.info(f"Loading dataset from {dataset_handle}...")

    return dataset_load(
        adapter=KaggleDatasetAdapter.PANDAS, handle=dataset_handle, path=dataset_path, pandas_kwargs=pandas_kwargs
    )


# -------------------- Data --------------------
def load_model_data(path: str) -> pd.DataFrame:
    """
    Load CSV and apply minimal parsing consistent with the notebook.
    - parses `ingredients` as a Python list (tokenized) if present.
    Args:
        path (str): Path to the CSV file.
    Returns:
        pd.DataFrame: Loaded DataFrame with parsed ingredients if applicable.
    """
    df = pd.read_csv(path)
    if "ingredients" in df.columns:
        df["ingredients"] = df["ingredients"].apply(_safe_eval_ingredients)
        logger.info(f"Loaded {len(df)} ingredients from {path}")
    return df.reset_index(drop=True)
