import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def _reset_env(monkeypatch):
    # Make sure tests don't leak env vars between cases
    for k in list(os.environ.keys()):
        if k.startswith(("KAGGLE_", "MLFLOW_TRACKING_")) or k in {"PYTHONHASHSEED"}:
            monkeypatch.delenv(k, raising=False)


def _base_settings(tmp_path: Path) -> SimpleNamespace:
    # Common settings object; tests override fields case-by-case
    return SimpleNamespace(
        SEED=33,
        TORCH_NUM_THREADS=1,
        MPL_FIGSIZE=(3, 2),
        MPL_DPI=90,
        ARTIFACT_DIR=str(tmp_path / "artifacts"),
        IGNORE_DEPRECATION_WARNINGS=True,
        IGNORE_FUTURE_WARNINGS=True,
        # Kaggle credentials (present for "happy" tests)
        KAGGLE_USERNAME="user",
        KAGGLE_KEY="secret",
        # MLflow default values (tests override as needed)
        MLFLOW_BACKEND="local",
        MLFLOW_TRACKING_URI="file:/tmp/mlruns",
        # Azure fields (used only in azure tests)
        AZURE_SUBSCRIPTION_ID=None,
        AZURE_RESOURCE_GROUP=None,
        AZURE_ML_WORKSPACE_NAME=None,
    )


# ---------------------------------------------------------------------
# apply_global_settings
# ---------------------------------------------------------------------
def test_apply_global_settings_happy_path(tmp_path, monkeypatch, _reset_env):
    import application.config.bootstrap as boot

    # settings
    S = _base_settings(tmp_path)
    monkeypatch.setattr(boot, "settings", S, raising=False)

    # fake torch with debug branch for set_num_threads
    class _Torch:
        def manual_seed(self, x):  # noqa: D401
            self._seed = x

        def set_num_threads(self, n):
            raise RuntimeError("nope")  # exercise exception path

    monkeypatch.setattr(boot, "torch", _Torch(), raising=False)

    # run
    boot.apply_global_settings()

    # artifact dir created
    assert Path(S.ARTIFACT_DIR).exists()
    # Kaggle env set
    assert os.environ["KAGGLE_USERNAME"] == "user"
    assert os.environ["KAGGLE_KEY"] == "secret"
    # Matplotlib backend forced
    assert os.environ.get("MPLBACKEND") == "Agg"


def test_apply_global_settings_raises_without_kaggle(tmp_path, monkeypatch, _reset_env):
    import application.config.bootstrap as boot

    S = _base_settings(tmp_path)
    S.KAGGLE_USERNAME = ""
    S.KAGGLE_KEY = ""
    monkeypatch.setattr(boot, "settings", S, raising=False)
    # No torch needed for this path
    monkeypatch.setattr(boot, "torch", None, raising=False)

    with pytest.raises(RuntimeError, match="KAGGLE_USERNAME and KAGGLE_KEY"):
        boot.apply_global_settings()


# ---------------------------------------------------------------------
# configure_mlflow_backend
# ---------------------------------------------------------------------
def _install_mlflow_stub(monkeypatch):
    import types

    mlflow = types.ModuleType("mlflow")
    calls = {"uri": None}
    mlflow.set_tracking_uri = lambda uri: calls.__setitem__("uri", uri)
    sys.modules["mlflow"] = mlflow
    return calls


def test_configure_mlflow_backend_local_sets_uri(tmp_path, monkeypatch):
    import application.config.bootstrap as boot

    calls = _install_mlflow_stub(monkeypatch)

    S = _base_settings(tmp_path)
    S.MLFLOW_BACKEND = "local"
    S.MLFLOW_TRACKING_URI = "file:/some/dir"
    monkeypatch.setattr(boot, "settings", S, raising=False)

    uri = boot.configure_mlflow_backend()
    assert uri == "file:/some/dir"
    assert calls["uri"] == "file:/some/dir"
    # token is cleared for local
    assert "MLFLOW_TRACKING_TOKEN" not in os.environ


def test_configure_mlflow_backend_azure_requires_env(monkeypatch, tmp_path):
    import types

    import application.config.bootstrap as boot

    _install_mlflow_stub(monkeypatch)

    # ---- stub azure modules + required symbols ----
    az_ml = types.ModuleType("azure.ai.ml")

    class _MLClient:  # minimal placeholder
        def __init__(self, *a, **k):
            pass

    az_ml.MLClient = _MLClient

    az_id = types.ModuleType("azure.identity")

    class _Cred:
        def __init__(self, *a, **k):
            pass

        def get_token(self, *a, **k):
            return types.SimpleNamespace(token="AAD_TOKEN")

    az_id.DefaultAzureCredential = _Cred

    sys.modules["azure"] = types.ModuleType("azure")
    sys.modules["azure.ai"] = types.ModuleType("azure.ai")
    sys.modules["azure.ai.ml"] = az_ml
    sys.modules["azure.identity"] = az_id

    # ---- settings: choose azure backend but leave AZURE_* unset ----
    S = _base_settings(tmp_path)
    S.MLFLOW_BACKEND = "azure"
    S.AZURE_SUBSCRIPTION_ID = None
    S.AZURE_RESOURCE_GROUP = None
    S.AZURE_ML_WORKSPACE_NAME = None
    monkeypatch.setattr(boot, "settings", S, raising=False)

    # now the function should import fine, then fail on missing env
    with pytest.raises(RuntimeError, match="required for backend=azure"):
        boot.configure_mlflow_backend()


def test_configure_mlflow_backend_azure_success(monkeypatch, tmp_path):
    import types

    import application.config.bootstrap as boot

    calls = _install_mlflow_stub(monkeypatch)

    # ----- stub azure classes -----
    class _WS:
        mlflow_tracking_uri = "azure://mlflow-ws"

    class _Workspaces:
        def get(self, name):
            return _WS()

    class _MLClient:
        def __init__(self, credential, subscription_id, resource_group_name, workspace_name):
            self.workspaces = _Workspaces()

    class _Cred:
        def __init__(self, *a, **k):
            pass

        def get_token(self, scope):
            return types.SimpleNamespace(token="AAD_TOKEN")

    az_ml = types.ModuleType("azure.ai.ml")
    az_ml.MLClient = _MLClient
    az_id = types.ModuleType("azure.identity")
    az_id.DefaultAzureCredential = _Cred

    sys.modules["azure"] = types.ModuleType("azure")
    sys.modules["azure.ai"] = types.ModuleType("azure.ai")
    sys.modules["azure.ai.ml"] = az_ml
    sys.modules["azure.identity"] = az_id

    # ----- settings -----
    S = _base_settings(tmp_path)
    S.MLFLOW_BACKEND = "azure"
    S.AZURE_SUBSCRIPTION_ID = "sub"
    S.AZURE_RESOURCE_GROUP = "rg"
    S.AZURE_ML_WORKSPACE_NAME = "ws"
    monkeypatch.setattr(boot, "settings", S, raising=False)

    uri = boot.configure_mlflow_backend()
    assert uri == "azure://mlflow-ws"
    assert calls["uri"] == "azure://mlflow-ws"
    # token set by DefaultAzureCredential
    assert os.environ.get("MLFLOW_TRACKING_TOKEN") == "AAD_TOKEN"


# ---------------------------------------------------------------------
# seed_everything
# ---------------------------------------------------------------------
def test_seed_everything_sets_env_and_calls_torch(monkeypatch):
    import application.config.bootstrap as boot

    # fake torch to record manual_seed
    class _Torch:
        def __init__(self):
            self.seeds = []

        def manual_seed(self, s):
            self.seeds.append(s)

    t = _Torch()
    monkeypatch.setattr(boot, "torch", t, raising=False)

    boot.seed_everything(33)
    assert os.environ["PYTHONHASHSEED"] == "33"
    assert t.seeds == [33]
