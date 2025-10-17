def test_ner_singleton_initializes_with_fake_hf(monkeypatch):
    # Patch torch device checks
    import application.networks.ner as ner_mod

    monkeypatch.setattr(ner_mod.torch.cuda, "is_available", lambda: False, raising=False)

    class _Backends:
        @staticmethod
        def is_available():
            return False

    monkeypatch.setattr(ner_mod.torch, "backends", type("B", (), {"mps": _Backends})(), raising=False)

    # Fake HF components
    class FakeTok:
        @classmethod
        def from_pretrained(cls, *a, **k):
            return cls()

    class FakeModel:
        @classmethod
        def from_pretrained(cls, *a, **k):
            return cls()

        def to(self, *_):
            return self

    def fake_pipeline(**kwargs):
        # return a callable that mimics the NER pipeline interface used in features.extract_ingredients_series
        def _call(text, aggregation_strategy=None):
            return [{"word": "tomato"}, {"word": "basil"}]

        return _call

    monkeypatch.setattr(ner_mod, "AutoTokenizer", FakeTok, raising=True)
    monkeypatch.setattr(ner_mod, "AutoModelForTokenClassification", FakeModel, raising=True)
    monkeypatch.setattr(ner_mod, "pipeline", fake_pipeline, raising=True)

    # Ensure a model name exists in settings (the conftest already sets NER_MODEL)
    # Instantiate twice → same instance
    from application.networks.ner import NERModelSingleton

    n1 = NERModelSingleton()
    n2 = NERModelSingleton()
    assert n1 is n2

    pipe = n1.get_pipeline()
    assert callable(pipe)
    # call the fake pipeline and confirm shape
    out = pipe("caprese salad", aggregation_strategy="simple")
    assert isinstance(out, list) and {"word"} <= set(out[0].keys())
