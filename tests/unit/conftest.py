import importlib
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _install_minimal_stubs():
    """
    Provide minimal external modules your dataset code imports so unit tests
    can run without the full stack (no real env, no network).
    """
    # ----- core package with a proper `core.settings` module -----
    core_pkg = types.ModuleType("core")  # package
    core_settings_mod = types.ModuleType("core.settings")  # submodule

    # The underlying settings object (if your code ever imports it directly)
    settings_obj = SimpleNamespace(
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
    )

    # (a) expose a `settings` variable inside the module
    core_settings_mod.settings = settings_obj
    # (b) also mirror all values as module-level attributes
    for k, v in settings_obj.__dict__.items():
        setattr(core_settings_mod, k, v)

    # Register package and submodule, and make package point to submodule for `from core import settings`
    sys.modules["core"] = core_pkg
    sys.modules["core.settings"] = core_settings_mod
    core_pkg.settings = core_settings_mod  # allow `from core import settings` to resolve to the submodule

    # ----- kagglehub (so loader import works even if pkg missing) -----
    kagglehub = types.ModuleType("kagglehub")

    class _Adapter:
        PANDAS = object()

    def dataset_load(adapter, handle, path, pandas_kwargs=None):
        raise RuntimeError("dataset_load stub should not be called in unit tests.")

    kagglehub.KaggleDatasetAdapter = _Adapter
    kagglehub.dataset_load = dataset_load
    sys.modules["kagglehub"] = kagglehub

    # ----- application.preprocessing constants used by splitter -----
    app_pre = types.ModuleType("application.preprocessing")
    app_pre.DATA_SPLIT_COL = "category"
    app_pre.TARGET_COL = "price"
    sys.modules["application.preprocessing"] = app_pre


@pytest.fixture(scope="session", autouse=True)
def _load_utils_misc():
    # ensure any old stub is gone, then load the real file so coverage sees it
    sys.modules.pop("application.utils.misc", None)
    sys.modules.pop("application.utils", None)
    importlib.import_module("application.utils.misc")


@pytest.fixture(scope="session", autouse=True)
def _stubs_installed():
    _install_minimal_stubs()


# ---------- Shared tiny DataFrames ----------


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
    # Include ZIPs and uppercase state codes so build_address_fields() extracts successfully
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


# =========================
# CLI stubs for tools.run.py
# =========================


def _install_cli_stubs():
    """
    Stubs for CLI-only imports used by tools/run.py:
      - application.config.apply_global_settings()
      - application.dataset.generate_training_sample()
      - pipelines.autotune_pipeline(), pipelines.dwh_export_pipeline()
      - model.REGISTRY
      - core.__version__ (and ensure core.settings has CLI-required attrs)
      - no-op mlflow in case it's not available
    """
    # -- application.config
    application = sys.modules.get("application") or types.ModuleType("application")

    from pathlib import Path

    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    application.__path__ = [str(PROJECT_ROOT / "application")]  # make it a package proxy

    application.config = types.ModuleType("application.config")

    def apply_global_settings():
        return None

    application.config.apply_global_settings = apply_global_settings
    sys.modules["application"] = application
    sys.modules["application.config"] = application.config

    # -- application.dataset
    application.dataset = sys.modules.get("application.dataset") or types.ModuleType("application.dataset")
    _cli_state.generate_calls = 0

    def generate_training_sample():
        _cli_state.generate_calls += 1
        return {"ok": True}

    application.dataset.generate_training_sample = generate_training_sample
    sys.modules["application.dataset"] = application.dataset

    # -- pipelines
    pipelines = types.ModuleType("pipelines")
    _cli_state.autotune_calls = []
    _cli_state.dwh_export_calls = 0

    def autotune_pipeline(model_names, data_path, n_trials, cv_folds, scoring, best_model_registry_name):
        _cli_state.autotune_calls.append(
            dict(
                model_names=tuple(model_names),
                data_path=data_path,
                n_trials=n_trials,
                cv_folds=cv_folds,
                scoring=scoring,
                best_model_registry_name=best_model_registry_name,
            )
        )
        return {"best_model_name": model_names[0] if model_names else "dummy"}

    def dwh_export_pipeline():
        _cli_state.dwh_export_calls += 1
        return {"ok": True}

    pipelines.autotune_pipeline = autotune_pipeline
    pipelines.dwh_export_pipeline = dwh_export_pipeline
    sys.modules["pipelines"] = pipelines

    # -- model registry
    model = types.ModuleType("model")
    model.REGISTRY = {"lr": object(), "dtree": object(), "xgboost": object()}
    sys.modules["model"] = model

    # -- core.__version__ and ensure settings has CLI fields
    core_pkg = sys.modules.get("core")
    if not core_pkg:
        core_pkg = types.ModuleType("core")
        sys.modules["core"] = core_pkg

    # attach __version__
    core_pkg.__version__ = "0.0-test"

    # ensure we have a 'core.settings' module with a 'settings' var and module-level attrs
    core_settings_mod = sys.modules.get("core.settings")
    if not core_settings_mod:
        core_settings_mod = types.ModuleType("core.settings")
        sys.modules["core.settings"] = core_settings_mod
        core_pkg.settings = core_settings_mod  # allow "from core import settings"

    # extend existing settings with CLI-specific keys
    base = getattr(core_settings_mod, "settings", None)
    if base is None:
        base = SimpleNamespace()
        core_settings_mod.settings = base

    for k, v in {
        "SAMPLED_DATA_PATH": getattr(base, "SAMPLED_DATA_PATH", "data/sampled-final-data.csv"),
        "N_TRIALS": getattr(base, "N_TRIALS", 3),
        "CV_FOLDS": getattr(base, "CV_FOLDS", 2),
        "SCORING": getattr(base, "SCORING", "neg_root_mean_squared_error"),
        "BEST_MODEL_REGISTRY_NAME": getattr(base, "BEST_MODEL_REGISTRY_NAME", "ubereats-menu-price-predictor"),
        "MLFLOW_TRACKING_URI": getattr(base, "MLFLOW_TRACKING_URI", "file:/tmp/mlruns"),
        "MLFLOW_EXPERIMENT_NAME": getattr(base, "MLFLOW_EXPERIMENT_NAME", "restaurant_price_exp"),
        "TRAINING_DATA_SAMPLE_PATH": getattr(base, "TRAINING_DATA_SAMPLE_PATH", "data/sampled-final-data.csv"),
        "DWH_EXPORT_DIR": getattr(base, "DWH_EXPORT_DIR", "data/dwh"),
        "RESTAURANT_DATA_PATH": getattr(base, "RESTAURANT_DATA_PATH", "data/dwh/restaurants.csv"),
        "MENU_DATA_PATH": getattr(base, "MENU_DATA_PATH", "data/dwh/menus.csv"),
        "MODEL_SERVE_PORT": getattr(base, "MODEL_SERVE_PORT", 5000),
    }.items():
        setattr(base, k, v)
        setattr(core_settings_mod, k, v)

    # Optional: provide a no-op mlflow for environments lacking it
    if "mlflow" not in sys.modules:
        mlflow_stub = types.ModuleType("mlflow")

        def _noop(*a, **k):
            return None

        mlflow_stub.set_tracking_uri = _noop
        mlflow_stub.set_experiment = _noop
        sys.modules["mlflow"] = mlflow_stub


# shared state for assertions in test_cli_run
_cli_state = SimpleNamespace(generate_calls=0, dwh_export_calls=0, autotune_calls=[])


@pytest.fixture(scope="session")
def cli_stub_state():
    _install_cli_stubs()
    return _cli_state
