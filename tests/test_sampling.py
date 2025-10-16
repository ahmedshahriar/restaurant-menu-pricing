import pandas as pd
import pytest

pytestmark = pytest.mark.unit


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
