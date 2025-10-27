import os

from click.testing import CliRunner

for k in (
    "MLFLOW_BACKEND",
    "AZURE_SUBSCRIPTION_ID",
    "AZURE_RESOURCE_GROUP",
    "AZURE_ML_WORKSPACE_NAME",
    "MLFLOW_TRACKING_URI",
    "MLFLOW_TRACKING_TOKEN",
):
    os.environ.pop(k, None)


def _import_cli():
    import importlib

    return importlib.import_module("tools.run")


def test_cli_help(cli_stub_state):
    run_mod = _import_cli()
    runner = CliRunner()
    res = runner.invoke(run_mod.cli, ["--help"])
    assert res.exit_code == 0
    assert "Restaurant Menu Pricing CLI" in res.output


def test_cli_list_models(cli_stub_state):
    run_mod = _import_cli()
    runner = CliRunner()
    res = runner.invoke(run_mod.cli, ["--list-models"])
    assert res.exit_code == 0
    assert "dtree" in res.output and "lr" in res.output


def test_cli_models_validation_too_few(cli_stub_state):
    run_mod = _import_cli()
    runner = CliRunner()
    res = runner.invoke(run_mod.cli, ["--models", "lr"])
    assert res.exit_code == 2
    assert "at least two models" in res.output.lower()


def test_cli_models_validation_invalid_name(cli_stub_state):
    run_mod = _import_cli()
    runner = CliRunner()
    res = runner.invoke(run_mod.cli, ["--models", "invalid,lr"])
    assert res.exit_code == 2
    assert "invalid model name" in res.output.lower()


def test_cli_dry_run_prints_plan(cli_stub_state, monkeypatch):
    run_mod = _import_cli()
    # silence mlflow side-effects if real mlflow is present
    monkeypatch.setattr(run_mod.mlflow, "set_tracking_uri", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(run_mod.mlflow, "set_experiment", lambda *a, **k: None, raising=False)

    runner = CliRunner()
    res = runner.invoke(run_mod.cli, ["--models", "lr,dtree", "--dry-run"])
    assert res.exit_code == 0
    assert "Plan:" in res.output
    assert "Models: ['lr', 'dtree']" in res.output


def test_cli_top_level_runs_autotune_pipeline(cli_stub_state, monkeypatch):
    import os

    import application.config.bootstrap as boot
    from core import settings as S

    # 1) clear any env that could force azure
    os.environ.pop("MLFLOW_BACKEND", None)
    os.environ.pop("AZURE_SUBSCRIPTION_ID", None)
    os.environ.pop("AZURE_RESOURCE_GROUP", None)
    os.environ.pop("AZURE_ML_WORKSPACE_NAME", None)
    os.environ.pop("MLFLOW_TRACKING_URI", None)
    os.environ.pop("MLFLOW_TRACKING_TOKEN", None)

    # 2) set on core.settings (module)
    monkeypatch.setattr(S, "MLFLOW_BACKEND", "local", raising=False)
    #    also set on nested settings object if present
    if hasattr(S, "settings"):
        monkeypatch.setattr(S.settings, "MLFLOW_BACKEND", "local", raising=False)

    # 3) ensure bootstrap is looking at the same settings object and value
    monkeypatch.setattr(boot, "settings", S, raising=False)
    monkeypatch.setattr(boot.settings, "MLFLOW_BACKEND", "local", raising=False)

    run_mod = _import_cli()

    # silence mlflow side-effects if real mlflow is present
    monkeypatch.setattr(run_mod.mlflow, "set_tracking_uri", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(run_mod.mlflow, "set_experiment", lambda *a, **k: None, raising=False)

    runner = CliRunner()
    res = runner.invoke(run_mod.cli, [])

    assert res.exit_code == 0, res.output
    assert len(cli_stub_state.autotune_calls) == 1
    call = cli_stub_state.autotune_calls[0]
    assert len(call["model_names"]) >= 2
    assert call["best_model_registry_name"] == "ubereats-menu-price-predictor"


def test_subcommand_generate_train_sample_calls_dataset(cli_stub_state, monkeypatch):
    run_mod = _import_cli()
    monkeypatch.setattr(run_mod.mlflow, "set_tracking_uri", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(run_mod.mlflow, "set_experiment", lambda *a, **k: None, raising=False)

    runner = CliRunner()
    res = runner.invoke(run_mod.cli, ["generate-train-sample"])
    assert res.exit_code == 0, res.output
    assert cli_stub_state.generate_calls == 1


def test_subcommand_dwh_export_calls_pipeline(cli_stub_state, monkeypatch):
    run_mod = _import_cli()
    monkeypatch.setattr(run_mod.mlflow, "set_tracking_uri", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(run_mod.mlflow, "set_experiment", lambda *a, **k: None, raising=False)

    runner = CliRunner()
    res = runner.invoke(run_mod.cli, ["dwh-export"])
    assert res.exit_code == 0, res.output
    assert cli_stub_state.dwh_export_calls == 1


def test_cli_top_level_wrapped_exception(cli_stub_state, monkeypatch):
    # --- ensure backend is 'local' BEFORE importing tools.run ---
    monkeypatch.delenv("MLFLOW_BACKEND", raising=False)

    # Some code paths read from env, others from core.settings;
    # make both say "local" *before* import-time side effects.
    monkeypatch.setenv("MLFLOW_BACKEND", "local")
    from core import settings as S

    monkeypatch.setattr(S, "MLFLOW_BACKEND", "local", raising=False)

    # Import CLI only after the backend is pinned
    run_mod = _import_cli()

    # Safety: completely no-op the bootstrapper regardless of how it's referenced
    # so backend checks never interfere with this negative-path test.
    # Your CLI may call configure_mlflow_backend directly or via a 'bootstrap' module.
    try:
        monkeypatch.setattr(run_mod, "configure_mlflow_backend", lambda: None, raising=False)
    except AttributeError:
        pass
    try:
        bootstrap = getattr(run_mod, "bootstrap", None)
        if bootstrap is not None:
            monkeypatch.setattr(bootstrap, "configure_mlflow_backend", lambda: None, raising=False)
    except AttributeError:
        pass

    # Silence real mlflow calls if mlflow is present
    monkeypatch.setattr(run_mod.mlflow, "set_tracking_uri", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(run_mod.mlflow, "set_experiment", lambda *a, **k: None, raising=False)

    # Force autotune failure to exercise ClickException branch
    def boom(**kwargs):
        raise RuntimeError("autotune failed")

    monkeypatch.setattr(run_mod, "autotune_pipeline", boom, raising=False)

    from click.testing import CliRunner

    res = CliRunner().invoke(run_mod.cli, [])
    assert res.exit_code != 0
    # Click will print the message (possibly prefixed by "Error: ")
    assert "autotune failed" in res.output.lower()
