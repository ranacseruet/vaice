import importlib

import pytest

from vaice import config


def test_bad_number_is_a_clean_error_not_a_traceback(monkeypatch):
    monkeypatch.setenv("VAICE_PORT", "78x0")
    with pytest.raises(SystemExit, match="VAICE_PORT must be a number"):
        config._number("VAICE_PORT", default="7860", kind=int)


def test_number_default_and_override(monkeypatch):
    monkeypatch.delenv("VAICE_LLAMACPP_TIMEOUT", raising=False)
    assert config._number("VAICE_LLAMACPP_TIMEOUT", default="120", kind=float) == 120.0
    monkeypatch.setenv("VAICE_LLAMACPP_TIMEOUT", "30.5")
    assert config._number("VAICE_LLAMACPP_TIMEOUT", default="120", kind=float) == 30.5


def test_env_default_and_override(monkeypatch):
    monkeypatch.delenv("VAICE_LLAMACPP_PROMPT", raising=False)
    assert config._env("VAICE_LLAMACPP_PROMPT", default="be brief") == "be brief"
    monkeypatch.setenv("VAICE_LLAMACPP_PROMPT", "be terse")
    assert config._env("VAICE_LLAMACPP_PROMPT", default="be brief") == "be terse"
    monkeypatch.setenv("VAICE_LLAMACPP_PROMPT", "")
    # an empty prompt is a choice, not "unset"
    assert config._env("VAICE_LLAMACPP_PROMPT", default="be brief") == ""


@pytest.mark.parametrize(
    ("given", "base"),
    [
        ("http://localhost:8001", "http://localhost:8001"),
        ("http://localhost:8001/v1/realtime", "http://localhost:8001"),
        ("http://localhost:8001/", "http://localhost:8001"),
        ("https://llama.example.com/v1", "https://llama.example.com"),
        ("127.0.0.1:8002", "http://127.0.0.1:8002"),  # no scheme: plain http
    ],
)
def test_the_server_url_keeps_only_scheme_host_and_port(given, base):
    assert config.server_base(given) == base


@pytest.mark.parametrize("given", ["", "http://", "ftp://host/x", "ws://host:1"])
def test_a_server_url_that_is_not_http_is_refused(given):
    with pytest.raises(ValueError, match="http"):
        config.server_base(given)


def test_server_url_default_and_override(monkeypatch):
    monkeypatch.delenv("VAICE_LLAMACPP_URL", raising=False)
    assert config._server_url("VAICE_LLAMACPP_URL", default="http://localhost:8001") == "http://localhost:8001"
    monkeypatch.setenv("VAICE_LLAMACPP_URL", "http://other:7000")
    assert config._server_url("VAICE_LLAMACPP_URL", default="http://localhost:8001") == "http://other:7000"


def test_a_bad_server_url_names_the_variable(monkeypatch):
    monkeypatch.setenv("VAICE_LLAMACPP_URL", "nonsense://")
    with pytest.raises(SystemExit, match="VAICE_LLAMACPP_URL must be a URL"):
        config._server_url("VAICE_LLAMACPP_URL", default="http://localhost:8001")


def test_the_settings_are_read_from_the_environment_at_import(monkeypatch):
    monkeypatch.setenv("VAICE_LLAMACPP_URL", "http://box:9000/v1")
    monkeypatch.setenv("VAICE_LLAMACPP_PROMPT", "custom prompt")
    monkeypatch.setenv("VAICE_LLAMACPP_TIMEOUT", "7")
    try:
        reloaded = importlib.reload(config)
        assert (reloaded.LLAMACPP_URL, reloaded.LLAMACPP_PROMPT, reloaded.LLAMACPP_TIMEOUT) == (
            "http://box:9000",
            "custom prompt",
            7.0,
        )
    finally:
        monkeypatch.undo()
        importlib.reload(config)
