# tests/unit/test_pipelines_dwh_export.py
import importlib
import sys
from types import SimpleNamespace

import pandas as pd


def test_dwh_export_pipeline_minimal(monkeypatch, tmp_path):
    # Ensure we get the submodule, not the package attribute
    sys.modules.pop("pipelines.dwh_export_pipeline", None)
    dp = importlib.import_module("pipelines.dwh_export_pipeline")

    # Patch the module’s settings object used inside the function
    monkeypatch.setattr(dp, "settings", SimpleNamespace(DWH_EXPORT_DIR=str(tmp_path)), raising=False)

    calls = {"fetch": 0, "build": 0, "save": []}

    def fake_fetch(query, projection):
        calls["fetch"] += 1
        return [
            {"_id": "a1", "name": "Place A", "menu_items": [{"title": "Soup", "price": 5.0}]},
            {"_id": "b2", "name": "Place B", "menu_items": []},
        ]

    monkeypatch.setattr(dp, "fetch_all_docs", fake_fetch, raising=True)

    def fake_build(docs):
        calls["build"] += 1
        df_rest = pd.DataFrame([{"id": "a1"}, {"id": "b2"}])
        df_menu = pd.DataFrame([{"restaurant_id": "a1", "title": "Soup", "price": 5.0}])
        return df_rest, df_menu

    monkeypatch.setattr(dp, "build_tables", fake_build, raising=True)

    def fake_save(df_rest, df_menu, out_dir, *, compress=False):
        calls["save"].append(dict(rows_rest=len(df_rest), rows_menu=len(df_menu), out_dir=out_dir, compress=compress))

    monkeypatch.setattr(dp, "save_data", fake_save, raising=True)

    # Call the function from the module
    dp.dwh_export_pipeline()

    # Assertions
    assert calls["fetch"] == 1
    assert calls["build"] == 1
    assert len(calls["save"]) == 1
    s0 = calls["save"][0]
    assert s0["rows_rest"] == 2 and s0["rows_menu"] == 1
    assert s0["out_dir"] == str(tmp_path)
    assert s0["compress"] is False
