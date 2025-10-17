import pytest


def test_get_client_success_and_singleton(monkeypatch):
    from infrastructure.db.mongo import MongoDatabaseConnector, get_client

    calls = {"ctor": 0, "ping": 0}

    class FakeAdmin:
        def command(self, name):
            assert name == "ping"
            calls["ping"] += 1
            return {"ok": 1}

    class FakeClient:
        def __init__(self, *a, **k):
            calls["ctor"] += 1

        @property
        def admin(self):
            return FakeAdmin()

    # Ensure clean slate for the singleton
    import infrastructure.db.mongo as mod

    MongoDatabaseConnector._instance = None  # <-- reset
    monkeypatch.setattr(mod, "MongoClient", FakeClient, raising=True)
    # in case it reads DATABASE_HOST
    monkeypatch.setattr(mod.settings, "DATABASE_HOST", "mongodb://localhost:27017", raising=False)

    c1 = get_client()
    c2 = get_client()
    assert c1 is c2  # singleton
    assert calls["ctor"] == 1
    assert calls["ping"] == 1


def test_get_client_failure_raises(monkeypatch):
    from pymongo.errors import ConnectionFailure

    import infrastructure.db.mongo as mod
    from infrastructure.db.mongo import MongoDatabaseConnector, get_client

    class FailClient:
        def __init__(self, *args, **kwargs):  # <-- accept host + options
            pass

        @property
        def admin(self):
            class _A:
                def command(self, *_):
                    raise ConnectionFailure("no host")

            return _A()

    # reset singleton and patch
    MongoDatabaseConnector._instance = None
    monkeypatch.setattr(mod, "MongoClient", FailClient, raising=True)
    monkeypatch.setattr(mod.settings, "DATABASE_HOST", "mongodb://localhost:27017", raising=False)

    with pytest.raises(ConnectionFailure):
        _ = get_client()
