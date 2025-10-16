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
