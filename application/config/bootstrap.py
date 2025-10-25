from __future__ import annotations

import os
import random
import warnings
from pathlib import Path

import numpy as np

# Choose a backend before importing pyplot to avoid GUI deps in servers
if "MPLBACKEND" not in os.environ:
    os.environ["MPLBACKEND"] = "Agg"

from loguru import logger
from matplotlib import pyplot as plt  # noqa: E402

try:
    import torch  # type: ignore
except Exception:
    torch = None  # type: ignore

try:
    from tqdm.auto import tqdm
except ImportError:
    from tqdm import tqdm

from core.settings import settings  # pydantic settings


def apply_global_settings() -> None:
    """
    Apply global runtime config:
      - reproducibility (numpy/python/torch)
      - matplotlib defaults
      - warnings filtering
    Safe to call multiple times.
    """
    # --- tqdm pandas integration ---
    tqdm.pandas()

    # --- Reproducibility ---
    # NOTE: PYTHONHASHSEED must be set before Python starts.
    hash_seed = os.environ.get("PYTHONHASHSEED")
    if hash_seed:
        logger.debug(f"PYTHONHASHSEED={hash_seed} (set at process start)")
    else:
        logger.info("PYTHONHASHSEED not set at launch; hash randomization may be nondeterministic.")

    random.seed(settings.SEED)
    np.random.seed(settings.SEED)

    if torch is not None:
        try:
            torch.manual_seed(settings.SEED)
            # Set threads only if configured and API is available
            n_threads = getattr(settings, "TORCH_NUM_THREADS", None)
            if n_threads:
                try:
                    torch.set_num_threads(n_threads)
                except Exception as e:
                    logger.debug(f"torch.set_num_threads failed: {e}")
        except Exception as e:
            logger.warning(f"Torch seeding/config failed: {e}")

    # --- Matplotlib defaults ---
    try:
        if getattr(settings, "MPL_FIGSIZE", None):
            plt.rcParams["figure.figsize"] = settings.MPL_FIGSIZE
        if getattr(settings, "MPL_DPI", None):
            plt.rcParams["figure.dpi"] = settings.MPL_DPI
    except Exception as e:
        logger.warning(f"Matplotlib configuration failed: {e}")

    # artifact directory
    directory_path = Path(settings.ARTIFACT_DIR)
    directory_path.mkdir(parents=True, exist_ok=True)

    # --- Warnings ---
    if settings.IGNORE_DEPRECATION_WARNINGS:
        warnings.filterwarnings("ignore", category=DeprecationWarning)
    if settings.IGNORE_FUTURE_WARNINGS:
        warnings.filterwarnings("ignore", category=FutureWarning)

    logger.info(f"Environment initialized with seed={settings.SEED}")

    # Configure Kaggle credentials from environment variables:
    #   - KAGGLE_USERNAME
    #   - KAGGLE_KEY
    os.environ["KAGGLE_USERNAME"] = settings.KAGGLE_USERNAME
    os.environ["KAGGLE_KEY"] = settings.KAGGLE_KEY

    if not settings.KAGGLE_USERNAME or not settings.KAGGLE_KEY:
        raise RuntimeError("KAGGLE_USERNAME and KAGGLE_KEY must be set in environment variables.")

    logger.info(f"Kaggle credentials set for user: {settings.KAGGLE_USERNAME}")


# ---------------------------
# NEW: MLflow backend switch
# ---------------------------
def configure_mlflow_backend() -> str:
    """
    Configure MLflow tracking for either local or Azure ML, based on env:
      - MLFLOW_BACKEND=local: use MLFLOW_TRACKING_URI as-is, clear tokens
      - MLFLOW_BACKEND=azure: resolve AML tracking URI + set AAD bearer token
    Returns the tracking URI that was set.
    """
    import mlflow

    if not settings.MLFLOW_BACKEND:
        raise RuntimeError("MLFLOW_BACKEND is required (must be 'local' or 'azure')")

    backend = settings.MLFLOW_BACKEND.lower()
    logger.info(f"MLFLOW_BACKEND={backend}")

    # Always start clean: if we aren't using Azure, don't leak an AAD token to local servers
    def _set_token(token: str | None):
        if token:
            os.environ["MLFLOW_TRACKING_TOKEN"] = token
        else:
            os.environ.pop("MLFLOW_TRACKING_TOKEN", None)

    if backend == "local":
        logger.info("Using local MLflow backend")
        if not settings.MLFLOW_TRACKING_URI:
            raise RuntimeError("MLFLOW_TRACKING_URI is required when MLFLOW_BACKEND=local")
        _set_token(None)
        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
        logger.info(f"MLflow backend=local uri={settings.MLFLOW_TRACKING_URI}")
        return settings.MLFLOW_TRACKING_URI

    # Azure path
    from azure.ai.ml import MLClient
    from azure.identity import DefaultAzureCredential

    cred = DefaultAzureCredential(exclude_interactive_browser_credential=True)

    if not (settings.AZURE_SUBSCRIPTION_ID and settings.AZURE_RESOURCE_GROUP and settings.AZURE_ML_WORKSPACE_NAME):
        raise RuntimeError(
            "AZURE_SUBSCRIPTION_ID, AZURE_RESOURCE_GROUP, AZURE_ML_WORKSPACE_NAME are required for backend=azure"
        )

    ml_client = MLClient(
        credential=cred,
        subscription_id=settings.AZURE_SUBSCRIPTION_ID,
        resource_group_name=settings.AZURE_RESOURCE_GROUP,
        workspace_name=settings.AZURE_ML_WORKSPACE_NAME,
    )

    # get the correct tracking URI from the workspace
    tracking_uri = ml_client.workspaces.get(settings.AZURE_ML_WORKSPACE_NAME).mlflow_tracking_uri
    mlflow.set_tracking_uri(tracking_uri)

    # acquire a short-lived AAD token for the AML MLflow server
    token = cred.get_token("https://ml.azure.com/.default").token
    _set_token(token)

    logger.info(f"MLflow backend=azure uri={tracking_uri}")
    return tracking_uri


# Optional: separate seeding helper to reuse it standalone
def seed_everything(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
