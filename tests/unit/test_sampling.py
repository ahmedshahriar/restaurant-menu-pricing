import numpy as np
import pandas as pd
import pytest


def test_load_base_frames_uses_injected_loader(monkeypatch):
    """
    Unit test only for load_base_frames: monkeypatch Kaggle loaders to return tiny frames.
    No HF model / NER / CSV writes here.
    """
    from application.dataset.sampling import Config, load_base_frames

    def fake_loader(handle, path, pandas_kwargs=None):
        if path.endswith("restaurants.csv"):
            return pd.DataFrame(
                [
                    {
                        "id": 1,
                        "score": 4.5,
                        "ratings": 10,
                        "category": "Salads",
                        "price_range": "$$",
                        "full_address": "1 St, Appleton, WI 54911",
                        "lat": 0.0,
                        "lng": 0.0,
                    }
                ]
            )
        if path.endswith("restaurant-menus.csv"):
            return pd.DataFrame([{"restaurant_id": 1, "category": "Salads", "description": "Tomato", "price": "9.0"}])
        if path.endswith("index.csv"):
            return pd.DataFrame([{"dummy": 1}])
        if path.endswith("density.csv"):
            return pd.DataFrame([{"city": "appleton", "state_id": "wi", "density": "1156"}])
        if path.endswith("states.csv"):
            return pd.DataFrame([{"Abbreviation": "wi", "State": "Wisconsin"}])
        return pd.DataFrame()

    # Patch BOTH the submodule and the package-level re-export
    import application.dataset.io as io_mod
    import application.dataset.io.loader as loader_mod

    monkeypatch.setattr(loader_mod, "load_kaggle_dataset", fake_loader, raising=True)
    monkeypatch.setattr(io_mod, "load_kaggle_dataset", fake_loader, raising=True)

    cfg = Config(
        RESTAURANTS_FILE="restaurants.csv",
        MENUS_FILE="restaurant-menus.csv",
        INDEX_FILE="index.csv",
        DENSITY_FILE="density.csv",
        STATES_FILE="states.csv",
    )
    frames = load_base_frames(cfg)
    assert len(frames) == 5
    df_restaurant, df_menu, df_index, df_density, df_states = frames
    assert not df_restaurant.empty and not df_menu.empty and not df_density.empty and not df_states.empty


def test_sampling_load_base_frames_smoke(monkeypatch):
    sm = pytest.importorskip("application.dataset.sampling", reason="sampling module not found")

    def fake_loader(handle, path, pandas_kwargs=None):
        if path.endswith("restaurants.csv"):
            return pd.DataFrame(
                [
                    {
                        "id": 1,
                        "price_range": "$$",
                        "category": "Salads",
                        "full_address": "1 St, Appleton, WI 54911",
                        "lat": 0.0,
                        "lng": 0.0,
                    }
                ]
            )
        if path.endswith("restaurant-menus.csv"):
            return pd.DataFrame([{"restaurant_id": 1, "category": "Salads", "description": "Tomato", "price": "9.0"}])
        if path.endswith("index.csv"):
            return pd.DataFrame([{"dummy": 1}])
        if path.endswith("density.csv"):
            return pd.DataFrame([{"city": "appleton", "state_id": "wi", "density": "1156"}])
        if path.endswith("states.csv"):
            return pd.DataFrame([{"Abbreviation": "wi", "State": "Wisconsin"}])
        return pd.DataFrame()

    # patch both exports, in case sampling imports either
    io = pytest.importorskip("application.dataset.io")
    loader_mod = pytest.importorskip("application.dataset.io.loader")
    monkeypatch.setattr(io, "load_kaggle_dataset", fake_loader, raising=True)
    monkeypatch.setattr(loader_mod, "load_kaggle_dataset", fake_loader, raising=True)

    frames = sm.load_base_frames(
        sm.Config(
            RESTAURANTS_FILE="restaurants.csv",
            MENUS_FILE="restaurant-menus.csv",
            INDEX_FILE="index.csv",
            DENSITY_FILE="density.csv",
            STATES_FILE="states.csv",
        )
    )
    assert len(frames) == 5
    for f in frames:
        assert f is not None and not f.empty


def _tiny_frames():
    """Return tiny but schema-correct frames matching load_base_frames expectations."""
    df_restaurant = pd.DataFrame(
        [{"id": 1, "name": "A", "price_range": "$$", "full_address": "123 Main, Appleton, WI 54911"}]
    )
    df_menu = pd.DataFrame([{"restaurant_id": 1, "category": "Salads", "description": "Tomato & Basil", "price": 9.0}])
    df_index = pd.DataFrame([{"state_id": "wi", "city": "appleton", "cost_of_living_index": 92.0}])
    df_density = pd.DataFrame([{"city": "appleton", "state_id": "wi", "density": "1156"}])
    df_states = pd.DataFrame(
        [
            {"Abbreviation": "wi", "State": "Wisconsin"},
            {"Abbreviation": "ca", "State": "California"},
        ]
    )
    return df_restaurant, df_menu, df_index, df_density, df_states


@pytest.mark.unit
def test_generate_training_sample_minimal(monkeypatch, tmp_path, tmp_cost_index_csv):
    """
    Executes sampling.generate_training_sample with tiny in-memory data by
    monkeypatching IO and processing functions. This avoids NER downloads and
    heavy transforms, but still validates the orchestration & output.
    """
    import application.dataset.sampling as sm

    # --- 1) Patch IO: load_kaggle_dataset returns tiny frames
    r, m, idx, den, st = _tiny_frames()

    def fake_loader(handle, path, pandas_kwargs=None):
        if path.endswith("restaurants.csv"):
            return r.copy()
        if path.endswith("restaurant-menus.csv"):
            return m.copy()
        if path.endswith("index.csv"):
            return idx.copy()
        if path.endswith("density.csv"):
            return den.copy()
        if path.endswith("states.csv"):
            return st.copy()
        return pd.DataFrame()

    import application.dataset.io as io_mod
    import application.dataset.io.loader as loader_mod

    monkeypatch.setattr(io_mod, "load_kaggle_dataset", fake_loader, raising=True)
    monkeypatch.setattr(loader_mod, "load_kaggle_dataset", fake_loader, raising=True)

    # --- 2) Patch processing functions to lightweight behavior
    proc = pytest.importorskip("application.dataset.processing", reason="processing module not found")

    # preprocess: keep as-is (or no-op)
    monkeypatch.setattr(proc, "preprocess_menu", lambda df: df.copy(), raising=True)

    # sync: pass-through frames
    monkeypatch.setattr(proc, "sync_restaurants_and_menus", lambda dr, dm: (dr.copy(), dm.copy()), raising=True)

    # address fields: directly seed city/state_id from the restaurant row
    def fake_build_address_fields(dr):
        out = dr.copy()
        out["city"] = ["appleton"]
        out["state_id"] = ["wi"]
        return out

    monkeypatch.setattr(proc, "build_address_fields", fake_build_address_fields, raising=True)

    # merge density: attach density as int
    def fake_merge_density(addr, density):
        out = addr.copy()
        out["density"] = 1156
        out["density"] = out["density"].astype(np.int32)
        return out

    monkeypatch.setattr(proc, "merge_density", fake_merge_density, raising=True)

    # filter to states: no-op but ensure types
    monkeypatch.setattr(proc, "filter_to_top_states", lambda df, states: df.copy(), raising=True)

    # states mapping
    monkeypatch.setattr(proc, "load_states_name_dict", lambda df_states: {"wi": "Wisconsin"}, raising=True)

    # compute_top_categories: produce df_res_ext + top_cats minimal
    def fake_compute_top_categories(df_menu, df_top_state, top_n_per_city):
        df_res_ext = pd.DataFrame(
            [
                {
                    "restaurant_id": 1,
                    "price_range": "$$",
                    "state_id": "wi",
                    "city": "appleton",
                    "density": 1156,
                }
            ]
        )
        top_cats = pd.DataFrame([{"state_id": "wi", "city": "appleton", "menu_category": "Salads", "count": 1}])
        return df_res_ext, top_cats

    monkeypatch.setattr(proc, "compute_top_categories", fake_compute_top_categories, raising=True)

    # pick top cities: keep appleton
    monkeypatch.setattr(
        proc,
        "pick_top_cities",
        lambda top_cats, focus_categories, top_cities_per_state: pd.DataFrame([{"state_id": "wi", "city": "appleton"}]),
        raising=True,
    )

    # final menu frame: produce a single-row frame with expected columns
    def fake_build_final_menu_frame(df_menu, df_res_ext, top_cities, focus_categories):
        return pd.DataFrame(
            [
                {
                    "restaurant_id": 1,
                    "price_range": "$$",
                    "state_id": "wi",
                    "city": "appleton",
                    "density": 1156,
                    "category": "Salads",
                    "description": "Tomato & Basil",
                    "price": 9.0,
                }
            ]
        )

    monkeypatch.setattr(proc, "build_final_menu_frame", fake_build_final_menu_frame, raising=True)

    # outlier removal: keep
    monkeypatch.setattr(proc, "remove_price_outliers_iqr", lambda df, price_col, whisker: df.copy(), raising=True)

    # NER pipeline + extraction: dummy ingredients
    monkeypatch.setattr(
        proc,
        "extract_ingredients_series",
        lambda s, ner: pd.Series([["tomato", "basil"] for _ in range(len(s))], index=s.index),
        raising=True,
    )
    monkeypatch.setattr(proc, "clean_ingredients_column", lambda df, col="ingredients": df, raising=True)

    # cost index attach: read from provided CSV
    def fake_attach_cost_index(df, df_cost):
        # emulate a left-merge on the expected keys
        return df.merge(df_cost, on=["state_id", "city"], how="left")

    monkeypatch.setattr(proc, "attach_cost_index", fake_attach_cost_index, raising=True)

    # normalize price range: map to 'moderate'
    def fake_normalize_price_range(df, col="price_range"):
        out = df.copy()
        out[col] = "moderate"
        return out

    monkeypatch.setattr(proc, "normalize_price_range", fake_normalize_price_range, raising=True)

    # --- 3) Build a Config pointing FINAL_SAMPLED_DATA_PATH into tmp_path
    from application.dataset.config import Config

    cfg = Config(
        RESTAURANTS_FILE="restaurants.csv",
        MENUS_FILE="restaurant-menus.csv",
        INDEX_FILE="index.csv",
        DENSITY_FILE="density.csv",
        STATES_FILE="states.csv",
        FINAL_SAMPLED_DATA_PATH=str(tmp_path / "sampled-final-data.csv"),
        # the following are read by load_base_frames; values don't matter for our fake_loader
        RESTAURANTS_DS="owner/restaurants",
        MENUS_DS="owner/menus",
        INDEX_DS="owner/index",
        DENSITY_DS="owner/density",
        STATES_DS="owner/states",
        # selection params (use the defaults if your Config has them)
        top_states_filter=("wi",),
        top_categories_per_city=5,
        focus_categories=("Salads", "Sandwiches"),
        top_cities_per_state=2,
        # ner
        NER_MODEL="Dizex/InstaFoodRoBERTa-NER",
    )

    # 🔧 stub the NER singleton used by sampling.py so we don't download a model
    class _FakeNERSingleton:
        def __init__(self):
            pass

        def get_pipeline(self):  # sampling passes this into extract_ingredients_series; our stub ignores it
            return object()

    monkeypatch.setattr(sm, "NERModelSingleton", _FakeNERSingleton, raising=True)
    # fallback if sampling ever changes imports
    monkeypatch.setattr("application.networks.ner.NERModelSingleton", _FakeNERSingleton, raising=False)

    # --- 4) Run and assert
    out = sm.generate_training_sample(cfg)

    # CSV persisted
    csv_path = tmp_path / "sampled-final-data.csv"
    assert csv_path.exists()

    saved = pd.read_csv(csv_path)
    assert set(saved.columns) >= {
        "price_range",
        "state_id",
        "city",
        "density",
        "category",
        "price",
        "ingredients",
        "cost_of_living_index",
    }
    assert len(saved) == len(out) == 1

    # Schema + values
    assert {
        "price_range",
        "state_id",
        "city",
        "density",
        "category",
        "price",
        "ingredients",
        "cost_of_living_index",
    } <= set(out.columns)
    assert len(out) == 1
    assert out["price_range"].iloc[0] == "moderate"
    assert out["state_id"].iloc[0] in ("wi", "Wisconsin")  # allow either, depending on your normalize order
    assert isinstance(out["ingredients"].iloc[0], list) and "tomato" in out["ingredients"].iloc[0]
