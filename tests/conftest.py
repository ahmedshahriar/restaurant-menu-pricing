import sys
import types
from types import SimpleNamespace

import pandas as pd
import pytest


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

    # ----- application.utils.misc (HTML + NER helpers) -----
    app_utils = types.ModuleType("application.utils")
    app_misc = types.ModuleType("application.utils.misc")

    def unescape_html(x: str) -> str:
        return x

    def convert_entities_to_list(text: str, ents):
        out = []
        for e in ents:
            tok = e.get("word") or e.get("entity_group") or ""
            if tok:
                out.append(str(tok).strip())
        return out

    app_misc.unescape_html = unescape_html
    app_misc.convert_entities_to_list = convert_entities_to_list
    app_utils.misc = app_misc
    sys.modules["application.utils"] = app_utils
    sys.modules["application.utils.misc"] = app_misc

    # ----- application.preprocessing constants used by splitter -----
    app_pre = types.ModuleType("application.preprocessing")
    app_pre.DATA_SPLIT_COL = "category"
    app_pre.TARGET_COL = "price"
    sys.modules["application.preprocessing"] = app_pre


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
