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


def test_a_bad_number_names_the_alias_that_was_set(monkeypatch):
    monkeypatch.delenv("VAICE_LLAMACPP_TIMEOUT", raising=False)
    monkeypatch.setenv("VAICE_LLM_TIMEOUT", "soon")
    with pytest.raises(SystemExit, match="VAICE_LLM_TIMEOUT must be a number"):
        config._number("VAICE_LLAMACPP_TIMEOUT", "VAICE_LLM_TIMEOUT", default="120", kind=float)


def test_the_new_name_wins_over_the_old_one(monkeypatch):
    monkeypatch.setenv("VAICE_LLAMACPP_TIMEOUT", "40")
    monkeypatch.setenv("VAICE_LLM_TIMEOUT", "90")
    assert config._number("VAICE_LLAMACPP_TIMEOUT", "VAICE_LLM_TIMEOUT", default="120", kind=float) == 40.0
    monkeypatch.setenv("VAICE_LLAMACPP_PROMPT", "new")
    monkeypatch.setenv("VAICE_LLM_PROMPT", "old")
    assert config._env("VAICE_LLAMACPP_PROMPT", "VAICE_LLM_PROMPT", "VAICE_PROMPT", default="d") == "new"


def test_the_old_name_still_works(monkeypatch):
    monkeypatch.delenv("VAICE_LLAMACPP_PROMPT", raising=False)
    monkeypatch.delenv("VAICE_LLM_PROMPT", raising=False)
    monkeypatch.setenv("VAICE_PROMPT", "be terse")
    assert config._env("VAICE_LLAMACPP_PROMPT", "VAICE_LLM_PROMPT", "VAICE_PROMPT", default="d") == "be terse"
    monkeypatch.delenv("VAICE_LLAMACPP_TIMEOUT", raising=False)
    monkeypatch.setenv("VAICE_LLM_TIMEOUT", "90")
    assert config._number("VAICE_LLAMACPP_TIMEOUT", "VAICE_LLM_TIMEOUT", default="120", kind=float) == 90.0


def test_neither_name_gives_the_default(monkeypatch):
    monkeypatch.delenv("VAICE_LLAMACPP_PROMPT", raising=False)
    monkeypatch.delenv("VAICE_LLM_PROMPT", raising=False)
    monkeypatch.delenv("VAICE_PROMPT", raising=False)
    assert config._env("VAICE_LLAMACPP_PROMPT", "VAICE_LLM_PROMPT", "VAICE_PROMPT", default="be brief") == "be brief"
    monkeypatch.setenv("VAICE_LLAMACPP_PROMPT", "")
    # an empty prompt is a choice, not "unset"
    assert config._env("VAICE_LLAMACPP_PROMPT", "VAICE_LLM_PROMPT", "VAICE_PROMPT", default="be brief") == ""


@pytest.mark.parametrize(
    ("given", "base"),
    [
        ("http://localhost:8001", "http://localhost:8001"),
        ("http://localhost:8001/v1/chat/completions", "http://localhost:8001"),  # the old VAICE_LLM_URL
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


def test_the_server_url_comes_from_the_new_name_or_the_old_one(monkeypatch):
    monkeypatch.delenv("VAICE_LLAMACPP_URL", raising=False)
    monkeypatch.delenv("VAICE_LLM_URL", raising=False)
    assert config._server_url("VAICE_LLAMACPP_URL", "VAICE_LLM_URL", default="http://localhost:8001") == (
        "http://localhost:8001"
    )
    monkeypatch.setenv("VAICE_LLM_URL", "http://box:9000/v1/chat/completions")
    assert config._server_url("VAICE_LLAMACPP_URL", "VAICE_LLM_URL", default="x") == "http://box:9000"
    monkeypatch.setenv("VAICE_LLAMACPP_URL", "http://other:7000")
    assert config._server_url("VAICE_LLAMACPP_URL", "VAICE_LLM_URL", default="x") == "http://other:7000"


def test_a_bad_server_url_names_the_variable(monkeypatch):
    monkeypatch.delenv("VAICE_LLAMACPP_URL", raising=False)
    monkeypatch.setenv("VAICE_LLM_URL", "nonsense://")
    with pytest.raises(SystemExit, match="VAICE_LLM_URL must be a URL"):
        config._server_url("VAICE_LLAMACPP_URL", "VAICE_LLM_URL", default="http://localhost:8001")


def test_the_settings_are_read_from_the_environment_at_import(monkeypatch):
    for name in ("VAICE_LLAMACPP_URL", "VAICE_LLAMACPP_PROMPT", "VAICE_LLAMACPP_TIMEOUT", "VAICE_PROMPT"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("VAICE_LLM_URL", "http://box:9000/v1/chat/completions")
    monkeypatch.setenv("VAICE_LLM_PROMPT", "old prompt")
    monkeypatch.setenv("VAICE_LLM_TIMEOUT", "7")
    try:
        reloaded = importlib.reload(config)
        assert (reloaded.LLAMACPP_URL, reloaded.LLAMACPP_PROMPT, reloaded.LLAMACPP_TIMEOUT) == (
            "http://box:9000",
            "old prompt",
            7.0,
        )
    finally:
        monkeypatch.undo()
        importlib.reload(config)
