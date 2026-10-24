import logging

import pytest

from vaice import responders


class Marker:
    def __init__(self, name):
        self.name = name


@pytest.fixture
def builders(monkeypatch):
    calls = []

    def native():
        calls.append("native")
        return Marker("native")

    def cascade():
        calls.append("cascade")
        return Marker("cascade")

    monkeypatch.setattr(responders, "_build_native", native)
    monkeypatch.setattr(responders, "_build_cascade", cascade)
    return calls


def test_auto_prefers_native(builders):
    assert responders.create_responder("auto").name == "native"
    assert builders == ["native"]


def test_auto_falls_back_to_cascade_with_warning(builders, monkeypatch, caplog):
    def broken():
        raise RuntimeError("no metal device")

    monkeypatch.setattr(responders, "_build_native", broken)
    with caplog.at_level(logging.WARNING):
        assert responders.create_responder("auto").name == "cascade"
    assert "no metal device" in caplog.text
    # the traceback is kept: the same path hides real bugs in the native backend
    assert caplog.records[-1].exc_info is not None


def test_explicit_native_does_not_fall_back(builders, monkeypatch):
    def broken():
        raise RuntimeError("boom")

    monkeypatch.setattr(responders, "_build_native", broken)
    with pytest.raises(RuntimeError, match="boom"):
        responders.create_responder("native")
    assert builders == []


def test_explicit_cascade_never_loads_native(builders):
    assert responders.create_responder("cascade").name == "cascade"
    assert builders == ["cascade"]


def test_unknown_backend_rejected(builders):
    with pytest.raises(ValueError, match="unknown backend"):
        responders.create_responder("bogus")


def test_native_disabled_when_model_empty(monkeypatch):
    monkeypatch.setattr(responders.config, "NATIVE_MODEL", "")
    with pytest.raises(RuntimeError, match="no native model"):
        responders._build_native()
