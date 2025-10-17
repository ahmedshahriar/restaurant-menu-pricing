import os

from click.testing import CliRunner


def _import_serve_main():
    import tools.serve as serve_mod

    return serve_mod.main


class _FakeProcessOK:
    def __init__(self, return_code=0):
        self._code = return_code
        self.pid = 12345

    def wait(self):
        return self._code

    def poll(self):
        return self._code

    def terminate(self):  # pragma: no cover - not used for OK path
        self._code = -15

    def kill(self):  # pragma: no cover - not used for OK path
        self._code = -9


class _FakeProcessRaisesKeyboardInterrupt:
    def __init__(self):
        self.pid = 9999
        self._code = None

    def wait(self):
        raise KeyboardInterrupt

    def poll(self):
        return self._code

    def terminate(self):
        self._code = -15

    def kill(self):
        self._code = -9


class _FakeProcessError:
    def __init__(self, code=2):
        self._code = code
        self.pid = 4242

    def wait(self):
        return self._code

    def poll(self):
        return self._code


def test_serve_child_exits_with_nonzero(monkeypatch):
    from click.testing import CliRunner

    import tools.serve as serve_mod

    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/mlflow")
    monkeypatch.setattr("subprocess.Popen", lambda *a, **k: _FakeProcessError(code=2))
    res = CliRunner().invoke(serve_mod.main, ["--port", "5556"])
    assert res.exit_code == 2
    assert "exited with code 2" in res.output.lower()


def test_serve_popen_filenotfound(monkeypatch):
    from click.testing import CliRunner

    import tools.serve as serve_mod

    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/mlflow")

    def _raise(*a, **k):
        raise FileNotFoundError("mlflow vanished")

    monkeypatch.setattr("subprocess.Popen", _raise)
    res = CliRunner().invoke(serve_mod.main, [])
    assert res.exit_code == 127
    assert "command not found" in res.output.lower()


def test_serve_errors_when_mlflow_cli_missing(monkeypatch):
    main = _import_serve_main()

    # Simulate missing 'mlflow' on PATH
    monkeypatch.setattr("shutil.which", lambda _: None)

    runner = CliRunner()
    res = runner.invoke(main, [])
    assert res.exit_code == 127
    assert "mlflow" in res.output.lower() and "not found" in res.output.lower()


def test_serve_starts_and_exits_success(monkeypatch):
    main = _import_serve_main()

    # Pretend 'mlflow' exists
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/mlflow")
    # Fake subprocess.Popen returning immediately with code 0
    monkeypatch.setattr("subprocess.Popen", lambda *a, **k: _FakeProcessOK(return_code=0))

    runner = CliRunner()
    res = runner.invoke(main, ["--port", "5555"])
    assert res.exit_code == 0
    assert "starting mlflow model server" in res.output.lower()
    assert "exited successfully" in res.output.lower()
    # The CLI should set MLFLOW_TRACKING_URI env var from core.settings
    assert os.environ.get("MLFLOW_TRACKING_URI") == "file:/tmp/mlruns"


def test_serve_keyboard_interrupt_graceful_shutdown(monkeypatch):
    main = _import_serve_main()

    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/mlflow")
    monkeypatch.setattr("subprocess.Popen", lambda *a, **k: _FakeProcessRaisesKeyboardInterrupt())

    runner = CliRunner()
    res = runner.invoke(main, [])
    # Ctrl+C path returns 130 (conventional)
    assert res.exit_code == 130
    assert "stopping mlflow model server" in res.output.lower()
    assert "server stopped" in res.output.lower()


def test_serve_unexpected_exception(monkeypatch):
    from click.testing import CliRunner

    import tools.serve as serve_mod

    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/mlflow")

    def _raise(*a, **k):
        raise RuntimeError("unexpected")

    monkeypatch.setattr("subprocess.Popen", _raise)
    res = CliRunner().invoke(serve_mod.main, [])
    # generic error path exits with 1
    assert res.exit_code == 1
    assert "unexpected error" in res.output.lower()
