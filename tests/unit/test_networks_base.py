def test_singleton_meta_reuses_instance_and_ignores_args():
    from application.networks.base import SingletonMeta

    class Dummy(metaclass=SingletonMeta):
        def __init__(self, value):
            self.value = value

    a = Dummy(1)
    b = Dummy(2)  # should return the same instance as 'a'
    assert a is b
    assert a.value == 1  # init args after first call don't overwrite

    # cleanup so other tests aren’t affected (optional)
    SingletonMeta._instances.pop(Dummy, None)
