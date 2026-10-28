import logging

import pytest

from vaice import responders


class Marker:
    def __init__(self, name):
        self.name = name


@pytest.fixture
def builders(monkeypatch):
    """Records which backends were built, and with which strictness."""
    calls = []

    def native():
        calls.append("native")
        return Marker("native")

    def llamacpp(*, strict):
        calls.append(f"llamacpp(strict={strict})")
        return Marker("llamacpp")

    monkeypatch.setattr(responders, "_build_native", native)
    monkeypatch.setattr(responders, "_build_llamacpp", llamacpp)
    return calls


def test_auto_prefers_llamacpp_and_checks_the_server_strictly(builders):
    assert responders.create_responder("auto").name == "llamacpp"
    assert builders == ["llamacpp(strict=True)"]  # native is never loaded when llama.cpp is there


def test_auto_falls_back_to_native_with_warning(builders, monkeypatch, caplog):
    def broken(*, strict):
        raise RuntimeError("connection refused")

    monkeypatch.setattr(responders, "_build_llamacpp", broken)
    with caplog.at_level(logging.WARNING):
        assert responders.create_responder("auto").name == "native"
    assert "connection refused" in caplog.text
    assert "falling back to native" in caplog.text
    # the traceback is kept: the same path hides real bugs in the llamacpp backend
    assert caplog.records[-1].exc_info is not None
    assert builders == ["native"]


def test_auto_fails_when_native_is_the_last_resort_and_cannot_load(builders, monkeypatch):
    def no_server(*, strict):
        raise RuntimeError("no server")

    def no_mlx():
        raise RuntimeError("no metal device")

    monkeypatch.setattr(responders, "_build_llamacpp", no_server)
    monkeypatch.setattr(responders, "_build_native", no_mlx)
    with pytest.raises(RuntimeError, match="no metal device"):
        responders.create_responder("auto")


def test_explicit_llamacpp_never_loads_native_and_does_not_fall_back(builders, monkeypatch):
    assert responders.create_responder("llamacpp").name == "llamacpp"
    assert builders == ["llamacpp(strict=False)"]  # a server that is merely down is only a warning

    def broken(*, strict):
        raise RuntimeError("the model cannot speak")

    monkeypatch.setattr(responders, "_build_llamacpp", broken)
    with pytest.raises(RuntimeError, match="cannot speak"):
        responders.create_responder("llamacpp")
    assert builders == ["llamacpp(strict=False)"]


def test_cascade_is_the_old_name_of_llamacpp(builders):
    assert responders.create_responder("cascade").name == "llamacpp"
    assert builders == ["llamacpp(strict=False)"]


def test_explicit_native_does_not_fall_back_and_never_touches_llamacpp(builders, monkeypatch):
    assert responders.create_responder("native").name == "native"
    assert builders == ["native"]

    def broken():
        raise RuntimeError("boom")

    monkeypatch.setattr(responders, "_build_native", broken)
    with pytest.raises(RuntimeError, match="boom"):
        responders.create_responder("native")
    assert builders == ["native"]


def test_unknown_backend_rejected(builders):
    with pytest.raises(ValueError, match="unknown backend"):
        responders.create_responder("bogus")


def test_the_choices_name_the_old_backend_too():
    assert responders.BACKENDS == ("auto", "llamacpp", "native")
    assert "cascade" in responders.BACKEND_CHOICES
    assert set(responders.BACKEND_CHOICES) - set(responders.BACKENDS) == set(responders.BACKEND_ALIASES)


def test_native_disabled_when_model_empty(monkeypatch):
    monkeypatch.setattr(responders.config, "NATIVE_MODEL", "")
    with pytest.raises(RuntimeError, match="no native model"):
        responders._build_native()


def test_building_llamacpp_checks_the_server_first(monkeypatch):
    from vaice.responders import llamacpp

    checked = []
    monkeypatch.setattr(llamacpp, "check_server", lambda strict: checked.append(strict))
    assert responders._build_llamacpp(strict=True).name == "llamacpp"
    assert checked == [True]
