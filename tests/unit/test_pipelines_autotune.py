def test_autotune_pipeline_happy_path(monkeypatch):
    import pandas as pd

    import pipelines.autotune as p

    # --- fakes & call capture ---
    calls = {
        "load": 0,
        "split": 0,
        "build": 0,
        "tune": [],
        "compare": [],
    }

    # dataset
    def fake_load(path):
        calls["load"] += 1
        return pd.DataFrame({"x": [1, 2, 3], "y": [10, 20, 30]})

    monkeypatch.setattr(p, "load_model_data", fake_load, raising=True)

    # split -> fixed outputs
    def fake_split(df):
        calls["split"] += 1
        Xtr = pd.DataFrame({"x": [1, 2]})
        Xte = pd.DataFrame({"x": [3]})
        ytr = pd.Series([10, 20], name="y")
        yte = pd.Series([30], name="y")
        return Xtr, Xte, ytr, yte

    monkeypatch.setattr(p, "split_data", fake_split, raising=True)

    # preprocessor
    def fake_build():
        calls["build"] += 1
        return object()

    monkeypatch.setattr(p, "build_preprocessor", fake_build, raising=True)

    # tune (should be skipped for "lr", called for others)
    def fake_tune(Xtr, ytr, Xte, yte, preprocessor, *, model_name, n_trials, cv_folds, scoring_criterion):
        calls["tune"].append(
            dict(model_name=model_name, n_trials=n_trials, cv_folds=cv_folds, scoring=scoring_criterion)
        )
        return {"alpha": 0.1}, 0.123  # best_params, best_metric

    monkeypatch.setattr(p, "tune_model", fake_tune, raising=True)

    # compare
    def fake_compare(tuned_models, Xtr, ytr, Xte, yte, preprocessor, *, scoring_criterion, best_model_registry_name):
        calls["compare"].append(
            dict(tuned_models=tuned_models, scoring=scoring_criterion, registry=best_model_registry_name)
        )
        results = {"lr": {"rmse": 1.0}, "xgboost": {"rmse": 0.9}}
        return results, "xgboost", "models:/xgboost/1"

    monkeypatch.setattr(p, "train_and_compare", fake_compare, raising=True)

    # --- run ---
    out = p.autotune_pipeline(
        model_names=["lr", "xgboost"],
        data_path="data/sampled-final-data.csv",
        n_trials=5,
        cv_folds=3,
        scoring="neg_root_mean_squared_error",
        best_model_registry_name="ubereats-menu-price-predictor",
    )

    # --- asserts ---
    assert calls["load"] == 1
    assert calls["split"] == 1
    assert calls["build"] == 1

    # "lr" is skipped -> tune only called once for "xgboost"
    assert len(calls["tune"]) == 1
    t0 = calls["tune"][0]
    assert t0["model_name"] == "xgboost"
    assert t0["n_trials"] == 5 and t0["cv_folds"] == 3
    assert t0["scoring"] == "neg_root_mean_squared_error"

    # compare invoked with tuned_models for xgboost and {} for lr
    assert len(calls["compare"]) == 1
    tuned = calls["compare"][0]["tuned_models"]
    assert "lr" in tuned and tuned["lr"] == {}
    assert "xgboost" in tuned and tuned["xgboost"] == {"alpha": 0.1}
    assert calls["compare"][0]["registry"] == "ubereats-menu-price-predictor"

    # returned payload shape
    assert set(out.keys()) == {"results", "best_model_name", "model_uri"}
    assert out["best_model_name"] == "xgboost"
    assert isinstance(out["results"], dict) and isinstance(out["model_uri"], str)
