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
