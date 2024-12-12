import contextlib


@contextlib.contextmanager
def monkeypatched(object, name, patch):
    old = getattr(object, name)
    setattr(object, name, patch)
    yield object
    setattr(object, name, old)
