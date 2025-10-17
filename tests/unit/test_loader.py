def test_load_model_data_parses_ingredients_list(tmp_path):
    import pandas as pd

    from application.dataset.io.loader import load_model_data

    p = tmp_path / "sample.csv"
    df = pd.DataFrame({"ingredients": [str(["Tomato", "Basil"]), str([])], "x": [1, 2]})
    df.to_csv(p, index=False)

    out = load_model_data(str(p))
    assert isinstance(out.loc[0, "ingredients"], list)
    assert out.loc[0, "ingredients"][0] == "Tomato"
    assert out.loc[1, "ingredients"] == []


def test_load_model_data_handles_malformed_ingredients(tmp_path):
    import pandas as pd

    from application.dataset.io.loader import load_model_data

    p = tmp_path / "bad.csv"
    # ingredients is not a list-looking string -> function should not crash; coerce to list or NaN
    pd.DataFrame({"ingredients": ["not a list", None], "x": [1, 2]}).to_csv(p, index=False)
    out = load_model_data(str(p))
    assert len(out) == 2  # smoke
    # accept either [] or NaN for bad entries; the important part is no exception
    bad = out.loc[0, "ingredients"]
    assert isinstance(bad, list) or pd.isna(bad)


def test_load_model_data_no_ingredients_column(tmp_path):
    import pandas as pd

    from application.dataset.io.loader import load_model_data

    p = tmp_path / "no_ing.csv"
    pd.DataFrame({"x": [1, 2]}).to_csv(p, index=False)

    out = load_model_data(str(p))
    # Should return frame unchanged (no error)
    assert list(out.columns) == ["x"]
    assert len(out) == 2
