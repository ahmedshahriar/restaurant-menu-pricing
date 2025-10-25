import importlib
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

# Ensure repo root on path (so "import application.*" works)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _install_minimal_stubs():
    """
    Minimal stubs required for unit tests (no network, no external services).
    """
    # ----- core.settings (module + settings object) -----
    core_pkg = sys.modules.get("core") or types.ModuleType("core")
    core_settings_mod = sys.modules.get("core.settings") or types.ModuleType("core.settings")

    defaults = dict(
        INDEX_DS="owner/index-ds",
        INDEX_FILE="index.csv",
        DENSITY_DS="owner/density-ds",
        DENSITY_FILE="density.csv",
        STATES_DS="owner/states-ds",
        STATES_FILE="states.csv",
        COST_OF_INDEX_UPDATED_FILE="cost_index.csv",
        SAMPLED_DATA_PATH="data/sampled-final-data.csv",
        TEST_SIZE=0.2,
        SEED=33,
        DATABASE_HOST="mongodb://localhost:27017",
        DATABASE_NAME="db",
        DATABASE_COLLECTION="restaurants",
        RESTAURANT_DATA_PATH="restaurants.csv",
        MENU_DATA_PATH="restaurant-menus.csv",
        NER_MODEL="Dizex/InstaFoodRoBERTa-NER",
        # NOTE: intentionally NOT setting MLFLOW_BACKEND here.
    )

    # 1) fill missing attributes on the module (don’t overwrite existing ones)
    for k, v in defaults.items():
        if not hasattr(core_settings_mod, k):
            setattr(core_settings_mod, k, v)

    # 2) ensure a .settings namespace exists and mirrors any missing fields
    if not hasattr(core_settings_mod, "settings") or not isinstance(core_settings_mod.settings, SimpleNamespace):
        core_settings_mod.settings = SimpleNamespace()

    for k in defaults:
        if not hasattr(core_settings_mod.settings, k):
            setattr(core_settings_mod.settings, k, getattr(core_settings_mod, k))

    # 3) register without replacing existing modules (prevents clobbering CLI values)
    sys.modules["core"] = core_pkg
    sys.modules["core.settings"] = core_settings_mod
    core_pkg.settings = core_settings_mod

    # ----- mlflow (avoid importing real mlflow in unit tests) -----
    mlflow = types.ModuleType("mlflow")
    mlflow.set_tracking_uri = lambda *a, **k: None
    mlflow.set_experiment = lambda *a, **k: None
    # useful no-ops if training code calls them inside a run:
    mlflow.start_run = lambda *a, **k: types.SimpleNamespace(__enter__=lambda s: s, __exit__=lambda s, *args: None)
    mlflow.end_run = lambda *a, **k: None
    mlflow.log_params = mlflow.log_param = lambda *a, **k: None
    mlflow.log_metrics = mlflow.log_metric = lambda *a, **k: None
    mlflow.set_tag = mlflow.set_tags = lambda *a, **k: None
    sys.modules["mlflow"] = mlflow

    mlflow_sklearn = types.ModuleType("mlflow.sklearn")
    mlflow_sklearn.autolog = lambda *a, **k: None
    sys.modules["mlflow.sklearn"] = mlflow_sklearn

    mlflow_models = types.ModuleType("mlflow.models")
    mlflow_models.infer_signature = lambda *a, **k: None
    sys.modules["mlflow.models"] = mlflow_models

    # ----- kagglehub (block accidental downloads) -----
    kagglehub = types.ModuleType("kagglehub")

    class _Adapter:
        PANDAS = object()

    def dataset_load(adapter, handle, path, pandas_kwargs=None):
        raise RuntimeError("dataset_load stub should not be called in unit tests.")

    kagglehub.KaggleDatasetAdapter = _Adapter
    kagglehub.dataset_load = dataset_load
    sys.modules["kagglehub"] = kagglehub

    # ----- application.preprocessing constants used by splitter -----
    # ----- application.preprocessing (stubbed) -----
    app_pre = types.ModuleType("application.preprocessing")
    app_pre.DATA_SPLIT_COL = "category"
    app_pre.TARGET_COL = "price"

    def _stub_build_preprocessor():
        # minimal placeholder so pipelines.autotune can import it;
        # tests will monkeypatch this symbol in pipelines.autotune anyway.
        return object()

    app_pre.build_preprocessor = _stub_build_preprocessor
    sys.modules["application.preprocessing"] = app_pre


@pytest.fixture(scope="session", autouse=True)
def _load_utils_misc():
    # Make sure we execute the on-disk misc.py so coverage counts it
    sys.modules.pop("application.utils.misc", None)
    sys.modules.pop("application.utils", None)
    importlib.import_module("application.utils.misc")


@pytest.fixture(scope="session", autouse=True)
def _unset_cli_stubbed_modules_for_unit():
    # If CLI suite injected stubs, remove them so we can import the on-disk package
    for name in ("pipelines", "model"):
        sys.modules.pop(name, None)


@pytest.fixture(scope="session", autouse=True)
def _install_unit_stubs():
    _install_minimal_stubs()


# ---------- Shared tiny DataFrames for multiple tests ----------


@pytest.fixture
def df_menu_raw():
    return pd.DataFrame(
        [
            {"restaurant_id": 1, "category": "Picked for you", "description": " &nbsp; ", "price": "0"},
            {"restaurant_id": 1, "category": "Salads", "description": "Tomato &amp; Basil", "price": "12.50USD"},
            {"restaurant_id": 2, "category": "Sandwiches", "description": "Bacon, Lettuce, Tomato", "price": "9.99"},
        ]
    )


@pytest.fixture
def df_restaurant_base():
    return pd.DataFrame(
        [
            {"id": 1, "price_range": "$$", "full_address": "123 Main, Appleton, WI 54911", "lat": 0.0, "lng": 0.0},
            {"id": 2, "price_range": "$", "full_address": "45 Oak, San Diego, CA 92101", "lat": 0.0, "lng": 0.0},
            {"id": 3, "price_range": None, "full_address": "No Menu, Austin, TX 73301", "lat": 0.0, "lng": 0.0},
        ]
    )


@pytest.fixture
def df_density():
    return pd.DataFrame(
        [
            {"city": "appleton", "state_id": "wi", "density": "1156"},
            {"city": "san diego", "state_id": "ca", "density": "4300"},
        ]
    )


@pytest.fixture
def df_states():
    return pd.DataFrame(
        [
            {"Abbreviation": "wi", "State": "Wisconsin"},
            {"Abbreviation": "ca", "State": "California"},
            {"Abbreviation": "tx", "State": "Texas"},
        ]
    )


@pytest.fixture
def tmp_cost_index_csv(tmp_path):
    df = pd.DataFrame(
        [
            {"state_id": "wi", "city": "appleton", "cost_of_living_index": 92.0},
            {"state_id": "ca", "city": "san diego", "cost_of_living_index": 145.0},
        ]
    )
    p = tmp_path / "cost_index.csv"
    df.to_csv(p, index=False)
    return str(p)
