import importlib
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

# --- Ensure project root is importable so `import tools.run` works ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Shared state object the tests assert against
_CLI_STATE = SimpleNamespace(
    generate_calls=0,
    dwh_export_calls=0,
    autotune_calls=[],
)


@pytest.fixture(scope="session", autouse=True)
def _install_cli_stubs():
    """
    Install CLI-only stubs without breaking the real package layout.
    Import the on-disk `application` package and attach/override the pieces
    tools.run/tools.serve use.
    """
    # ----- import the real application package (not a synthetic module) -----
    application = importlib.import_module("application")

    # application.config.apply_global_settings
    try:
        config_mod = importlib.import_module("application.config")
    except ModuleNotFoundError:
        config_mod = types.ModuleType("application.config")
        sys.modules["application.config"] = config_mod
    config_mod.apply_global_settings = lambda: None
    application.config = config_mod

    # application.dataset.generate_training_sample
    dataset_mod = importlib.import_module("application.dataset")

    def generate_training_sample():
        _CLI_STATE.generate_calls += 1
        return {"ok": True}

    dataset_mod.generate_training_sample = generate_training_sample
    application.dataset = dataset_mod

    # ----- pipelines (standalone module) -----
    pipelines = types.ModuleType("pipelines")

    def autotune_pipeline(model_names, data_path, n_trials, cv_folds, scoring, best_model_registry_name):
        _CLI_STATE.autotune_calls.append(
            dict(
                model_names=tuple(model_names),
                data_path=data_path,
                n_trials=n_trials,
                cv_folds=cv_folds,
                scoring=scoring,
                best_model_registry_name=best_model_registry_name,
            )
        )
        return {"best_model_name": (model_names or ["dummy"])[0]}

    def dwh_export_pipeline():
        _CLI_STATE.dwh_export_calls += 1
        return {"ok": True}

    pipelines.autotune_pipeline = autotune_pipeline
    pipelines.dwh_export_pipeline = dwh_export_pipeline
    sys.modules["pipelines"] = pipelines

    # ----- model registry -----
    model = types.ModuleType("model")
    model.REGISTRY = {"lr": object(), "dtree": object(), "xgboost": object()}
    sys.modules["model"] = model

    # ----- core.__version__ and core.settings -----
    core = sys.modules.get("core") or types.ModuleType("core")
    core.__version__ = "0.0-test"
    sys.modules["core"] = core

    core_settings = sys.modules.get("core.settings") or types.ModuleType("core.settings")
    # tests/cli/conftest.py  (inside _install_cli_stubs, in settings_values)
    settings_values = {
        "TRAINING_DATA_SAMPLE_PATH": "data/sampled-final-data.csv",
        "SAMPLED_DATA_PATH": "data/sampled-final-data.csv",
        "N_TRIALS": 3,
        "CV_FOLDS": 2,
        "SCORING": "neg_root_mean_squared_error",
        # match the test's expectation:
        "BEST_MODEL_REGISTRY_NAME": "ubereats-menu-price-predictor",
        # force local backend for CLI tests:
        "MLFLOW_BACKEND": "local",
        "MLFLOW_TRACKING_URI": "file:/tmp/mlruns",
        "MLFLOW_EXPERIMENT_NAME": "restaurant_price_exp",
        "MODEL_SERVE_PORT": 5000,
        "DWH_EXPORT_DIR": "data/dwh",
        "RESTAURANT_DATA_PATH": "data/dwh/restaurants.csv",
        "MENU_DATA_PATH": "data/dwh/menus.csv",
    }
    for k, v in settings_values.items():
        setattr(core_settings, k, v)
    core_settings.settings = SimpleNamespace(**settings_values)
    core.settings = core_settings
    sys.modules["core.settings"] = core_settings

    # --- ensure hostile env vars don't override stub settings ---
    for k in (
        "MLFLOW_BACKEND",
        "AZURE_SUBSCRIPTION_ID",
        "AZURE_RESOURCE_GROUP",
        "AZURE_ML_WORKSPACE_NAME",
        "MLFLOW_TRACKING_URI",
        "MLFLOW_TRACKING_TOKEN",
    ):
        os.environ.pop(k, None)

    # ----- no-op mlflow (if missing) -----
    if "mlflow" not in sys.modules:
        mlflow = sys.modules.get("mlflow") or types.ModuleType("mlflow")
        mlflow.__version__ = "2.14.0"
        mlflow.set_tracking_uri = getattr(mlflow, "set_tracking_uri", lambda *a, **k: None)
        mlflow.set_experiment = getattr(mlflow, "set_experiment", lambda *a, **k: None)

        # some code calls mlflow.tracking.get_tracking_uri()
        mlflow_tracking = sys.modules.get("mlflow.tracking") or types.ModuleType("mlflow.tracking")
        if not hasattr(mlflow_tracking, "get_tracking_uri"):
            mlflow_tracking.get_tracking_uri = lambda *a, **k: "file:/tmp/mlruns"
        sys.modules["mlflow.tracking"] = mlflow_tracking
        mlflow_sklearn = sys.modules.get("mlflow.sklearn") or types.ModuleType("mlflow.sklearn")
        if not hasattr(mlflow_sklearn, "autolog"):
            mlflow_sklearn.autolog = lambda *a, **k: None
        sys.modules["mlflow"] = mlflow
        sys.modules["mlflow.sklearn"] = mlflow_sklearn


@pytest.fixture
def cli_stub_state():
    """Shared call counters/state for assertions in CLI tests."""
    return _CLI_STATE
