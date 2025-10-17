import pytest


def test_build_tables_creates_restaurant_and_menu_frames():
    from application.dataset.dwh_export import build_tables

    docs = [
        {
            "_id": "a1",
            "name": "R1",
            "menu_items": [
                {"title": "Salad", "price": 10.0},
                {"title": "Burger", "price": 12.0},
            ],
        },
        {"_id": "a2", "name": "R2", "menu_items": []},
    ]
    df_rest, df_menu = build_tables(docs)
    assert "id" in df_rest.columns and "_id" not in df_rest.columns
    assert "restaurant_id" in df_menu.columns
    assert len(df_menu) == 2


def test_build_tables_handles_missing_and_empty():
    dwh = pytest.importorskip("application.dataset.dwh_export")

    docs = [
        {"_id": "x1", "name": "NoMenu"},  # no menu_items key
        {"_id": "x2", "name": "EmptyMenu", "menu_items": []},
        {
            "_id": "x3",
            "name": "SomeMenu",
            "menu_items": [{"title": "Salad", "price": 10.0}, {"title": None, "price": None}],
        },
    ]
    df_rest, df_menu = dwh.build_tables(docs)
    # restaurants table built, _id removed
    assert "_id" not in df_rest.columns and "id" in df_rest.columns
    # menu exploded (with backref) but skips/bends null rows safely
    assert "restaurant_id" in df_menu.columns
    assert len(df_menu) >= 1


def test_build_tables_skips_non_dict_menu_items():
    from application.dataset.dwh_export import build_tables

    docs = [{"_id": "a1", "name": "Weird", "menu_items": ["not-a-dict", {"title": "Ok", "price": 1.0}]}]
    df_rest, df_menu = build_tables(docs)

    # Restaurant table: 1 row, _id normalized to 'id'
    assert len(df_rest) == 1
    assert "_id" not in df_rest.columns and "id" in df_rest.columns

    # Menu table: one NaN row (from non-dict) + one valid dict row.
    assert len(df_menu) == 2
    assert "restaurant_id" in df_menu.columns

    valid = df_menu[df_menu["title"].notna()]
    assert len(valid) == 1
    assert valid["title"].iloc[0] == "Ok"
    assert valid["price"].iloc[0] == 1.0


def test_fetch_all_docs_no_query_uses_estimate(monkeypatch):
    from application.dataset.dwh_export import fetch_all_docs

    class FakeCursor(list):
        def sort(self, *_):
            return self

        def batch_size(self, *_):
            return self

    class FakeColl:
        def estimated_document_count(self):
            return 3

        def count_documents(self, *_):
            return 3

        def find(self, *_):
            return FakeCursor([{"_id": 1}, {"_id": 2}, {"_id": 3}])

    class FakeDB(dict):
        def __getitem__(self, k):
            return FakeColl()

    class FakeClient:
        def get_database(self, *_):
            return FakeDB()

    # patch client + settings-based getter
    import application.dataset.dwh_export as mod

    monkeypatch.setattr(mod, "get_client", lambda: FakeClient(), raising=True)

    docs = fetch_all_docs()  # no query -> estimated_document_count path
    assert len(docs) == 3
    assert docs[0]["_id"] == 1


def test_fetch_all_docs_with_query_uses_count_documents(monkeypatch):
    from application.dataset.dwh_export import fetch_all_docs

    calls = {"count": 0}

    class FakeCursor(list):
        def sort(self, *_):
            return self

        def batch_size(self, *_):
            return self

    class FakeColl:
        def estimated_document_count(self):
            return 0  # should NOT be used

        def count_documents(self, *_):
            calls["count"] += 1
            return 2

        def find(self, *_):
            return FakeCursor([{"_id": "a"}, {"_id": "b"}])

    class FakeDB(dict):
        def __getitem__(self, k):
            return FakeColl()

    class FakeClient:
        def get_database(self, *_):
            return FakeDB()

    import application.dataset.dwh_export as mod

    monkeypatch.setattr(mod, "get_client", lambda: FakeClient(), raising=True)

    docs = fetch_all_docs(query={"x": 1})
    assert calls["count"] == 1
    assert [d["_id"] for d in docs] == ["a", "b"]


def test_fetch_all_docs_error_is_logged_and_raised(monkeypatch):
    from application.dataset.dwh_export import fetch_all_docs

    class FakeColl:
        def estimated_document_count(self):
            return 1

        def count_documents(self, *_):
            return 1

        def find(self, *_):
            raise RuntimeError("boom")

    class FakeDB(dict):
        def __getitem__(self, k):
            return FakeColl()

    class FakeClient:
        def get_database(self, *_):
            return FakeDB()

    import application.dataset.dwh_export as mod

    monkeypatch.setattr(mod, "get_client", lambda: FakeClient(), raising=True)

    with pytest.raises(RuntimeError):
        fetch_all_docs()


def test_save_data_handles_compress_and_empty_menu(tmp_path, monkeypatch):
    import pandas as pd

    import application.dataset.dwh_export as mod

    # Patch the settings object that save_data actually uses
    monkeypatch.setattr(mod.settings, "RESTAURANT_DATA_PATH", "restaurants.csv", raising=False)
    monkeypatch.setattr(mod.settings, "MENU_DATA_PATH", "restaurant-menus.csv", raising=False)

    df_rest = pd.DataFrame([{"id": 1, "name": "R"}])
    df_menu = pd.DataFrame()  # exercise "No menu data to write."

    # write gzipped
    mod.save_data(df_rest, df_menu, str(tmp_path), compress=True)

    # both files use configured names with optional .gz
    r_path = tmp_path / "restaurants.csv.gz"
    m_path = tmp_path / "restaurant-menus.csv.gz"

    assert r_path.exists()
    # menu not written when empty
    assert not m_path.exists()
