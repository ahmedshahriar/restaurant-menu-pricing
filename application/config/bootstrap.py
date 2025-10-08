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

from core.settings import settings  # pydantic settings


def apply_global_settings() -> None:
    """
    Apply global runtime config:
      - reproducibility (numpy/python/torch)
      - matplotlib defaults
      - warnings filtering
    Safe to call multiple times.
    """

    # --- Reproducibility ---
    # NOTE: PYTHONHASHSEED must be set before Python starts.
    hash_seed = os.environ.get("PYTHONHASHSEED")
    if hash_seed:
        logger.debug("PYTHONHASHSEED=%s (set at process start)", hash_seed)
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

    print(f"✅ Environment initialized with seed={settings.SEED}")


# Optional: separate seeding helper to reuse it standalone
def seed_everything(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
