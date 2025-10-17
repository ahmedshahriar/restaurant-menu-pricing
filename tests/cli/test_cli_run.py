import pytest
from click.testing import CliRunner

pytestmark = [pytest.mark.unit, pytest.mark.cli]


def _import_cli():
    import tools.run as run_mod

    return run_mod


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
    run_mod = _import_cli()
    # silence mlflow in case it's real
    monkeypatch.setattr(run_mod.mlflow, "set_tracking_uri", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(run_mod.mlflow, "set_experiment", lambda *a, **k: None, raising=False)

    runner = CliRunner()
    res = runner.invoke(run_mod.cli, [])
    assert res.exit_code == 0, res.output
    # Ensure autotune was called with >= 2 models (defaults to all)
    assert len(cli_stub_state.autotune_calls) == 1
    call = cli_stub_state.autotune_calls[0]
    assert len(call["model_names"]) >= 2
    assert call["best_model_registry_name"] == "restaurant_best_model"


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
    run_mod = _import_cli()
    # silence mlflow
    monkeypatch.setattr(run_mod.mlflow, "set_tracking_uri", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(run_mod.mlflow, "set_experiment", lambda *a, **k: None, raising=False)

    # force autotune failure to exercise ClickException branch
    def boom(**kwargs):
        raise RuntimeError("autotune failed")

    monkeypatch.setattr(run_mod, "autotune_pipeline", boom, raising=False)
    from click.testing import CliRunner

    res = CliRunner().invoke(run_mod.cli, [])
    assert res.exit_code != 0
    assert "autotune failed" in res.output.lower()
