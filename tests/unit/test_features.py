import pandas as pd


def test_attach_cost_index_adds_column(tmp_cost_index_csv):
    import pandas as pd

    from application.dataset.processing.features import attach_cost_index

    df = pd.DataFrame(
        [
            {"city": "appleton", "state_id": "wi", "price": 10.0},
            {"city": "san diego", "state_id": "ca", "price": 12.0},
        ]
    )
    df_cost = pd.read_csv(tmp_cost_index_csv)

    out = attach_cost_index(df, df_cost)
    assert "cost_of_living_index" in out.columns
    assert out.loc[out.city.eq("appleton"), "cost_of_living_index"].iloc[0] == 92.0


def test_merge_density_adds_int_density_and_filter_to_top_states(df_restaurant_base, df_density):
    from application.dataset.processing.features import filter_to_top_states, merge_density

    # Seed address cols directly to avoid dependency on the address regex.
    addr = df_restaurant_base.copy()
    addr["city"] = ["appleton", "san diego", "austin"]
    addr["state_id"] = ["wi", "ca", "tx"]

    merged = merge_density(addr, df_density)
    assert "density" in merged.columns
    assert str(merged["density"].dtype).startswith("int")

    wi_only = filter_to_top_states(merged, states=("wi",))
    assert set(wi_only["state_id"].astype(str).str.lower()) == {"wi"}


def test_load_states_name_dict_maps_abbrev_to_name(df_states):
    from application.dataset.processing.features import load_states_name_dict

    mapping = load_states_name_dict(df_states.copy())
    assert mapping["wi"] == "Wisconsin"
    assert mapping["ca"] == "California"


def test_merge_density_drops_unmatched_and_casts_to_int(df_density):
    from application.dataset.processing.features import merge_density

    addr = pd.DataFrame(
        [
            {"id": 1, "city": "nowhere", "state_id": "zz"},
            {"id": 2, "city": "appleton", "state_id": "wi"},
        ]
    )

    merged = merge_density(addr, df_density)

    # Inner-join semantics: unmatched rows are dropped
    assert "nowhere" not in set(merged["city"].astype(str).str.lower())

    # Density present and cast to integer-ish dtype
    assert "density" in merged.columns
    assert str(merged["density"].dtype).startswith("int")

    # Sanity check a matched row
    assert int(merged.loc[merged["city"].str.lower() == "appleton", "density"].iloc[0]) == 1156


def test_filter_to_top_states_returns_same_when_all_states_requested():
    from application.dataset.processing.features import filter_to_top_states

    df = pd.DataFrame([{"state_id": "wi"}, {"state_id": "ca"}])
    # Pass all present states explicitly (function requires an iterable)
    same = filter_to_top_states(df, states=("wi", "ca"))
    assert same.equals(df)


def test_load_states_name_dict_tolerates_mixed_case(df_states):
    from application.dataset.processing.features import load_states_name_dict

    # Mixed casing -> mapping should normalize keys
    df = df_states.copy()
    df.loc[0, "Abbreviation"] = "Wi"
    mapping = load_states_name_dict(df)
    assert mapping["wi"] == "Wisconsin"
