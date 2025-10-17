import pandas as pd


def test_pick_top_cities_selects_expected_pairs():
    from application.dataset.processing.selection import pick_top_cities

    # Minimal, explicit "top categories" frame the function expects
    top_cats = pd.DataFrame(
        [
            {"state_id": "wi", "city": "appleton", "menu_category": "Salads", "count": 12},
            {"state_id": "wi", "city": "appleton", "menu_category": "Sandwiches", "count": 8},
            {"state_id": "ca", "city": "san diego", "menu_category": "Salads", "count": 15},
            {"state_id": "tx", "city": "austin", "menu_category": "Pizzas", "count": 9},
        ]
    )

    result = pick_top_cities(
        top_cats,
        focus_categories=("Salads", "Sandwiches"),
        top_cities_per_state=2,
    )

    assert {"state_id", "city"}.issubset(result.columns)
    # At least the focused pairs should survive
    pairs = set(zip(result["state_id"], result["city"], strict=True))
    assert ("wi", "appleton") in pairs
    assert ("ca", "san diego") in pairs


def test_build_final_menu_frame_joins_and_outputs_columns(df_restaurant_base, df_menu_raw):
    import pandas as pd

    from application.dataset.processing.selection import build_final_menu_frame

    # Pretend we already extended restaurants with geo features
    df_res_ext = df_restaurant_base.copy()

    # 🔧 Ensure the column matches what the code expects
    df_res_ext = df_res_ext.rename(columns={"id": "restaurant_id"})

    df_res_ext["city"] = ["appleton", "san diego", "austin"]
    df_res_ext["state_id"] = ["wi", "ca", "tx"]
    df_res_ext["price_range"] = ["$$", "$", "$$"]
    df_res_ext["density"] = [1156, 4300, 3100]

    # Top cities we want to keep (output from pick_top_cities)
    top_cities = pd.DataFrame(
        [
            {"state_id": "wi", "city": "appleton"},
            {"state_id": "ca", "city": "san diego"},
        ]
    )

    # Menu with expected column names
    df_menu = df_menu_raw.rename(columns={"description": "description"}).copy()

    final = build_final_menu_frame(
        df_menu=df_menu,
        df_res_ext=df_res_ext,
        top_cities=top_cities,
        focus_categories=("Salads", "Sandwiches"),
    )

    assert not final.empty
    assert {"price_range", "state_id", "city", "category", "description", "price"}.issubset(final.columns)


def test_compute_top_categories_minimal(df_menu_raw, df_restaurant_base):
    from application.dataset.processing.selection import compute_top_categories

    # Restaurants keep 'id' (right_on key), and add geo + restaurant category
    res = df_restaurant_base.copy()
    res["city"] = ["appleton", "san diego", "austin"]
    res["state_id"] = ["wi", "ca", "tx"]
    res["density"] = [1156, 4300, 3100]

    # add restaurant-side category so merge yields category_x/category_y
    res["category"] = ["House", "House", "House"]

    # Menu already has restaurant_id/category
    df_menu = df_menu_raw.copy()

    # Keep it deterministic/small
    df_res_ext, top_cats = compute_top_categories(df_menu, res, top_n_per_city=3)

    # Expectations: menu_category exists and grouping worked
    assert {"restaurant_id", "city", "state_id", "density"}.issubset(df_res_ext.columns)
    assert "menu_category" in df_res_ext.columns
    assert not top_cats.empty
    assert {"state_id", "city", "menu_category", "count"}.issubset(top_cats.columns)
