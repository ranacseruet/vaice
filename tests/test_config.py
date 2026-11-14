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


def test_a_bad_flag_is_a_clean_error_not_a_traceback(monkeypatch):
    monkeypatch.setenv("VAICE_CAN_INTERRUPT", "maybe")
    with pytest.raises(SystemExit, match="VAICE_CAN_INTERRUPT must be a boolean"):
        config._flag("VAICE_CAN_INTERRUPT", default=True)


def test_tools_are_off_when_the_variable_is_unset(monkeypatch):
    monkeypatch.delenv("VAICE_LLAMACPP_TOOLS", raising=False)
    assert config._tools("VAICE_LLAMACPP_TOOLS") is None


def test_tools_builtin_means_every_tool_the_server_exposes(monkeypatch):
    monkeypatch.setenv("VAICE_LLAMACPP_TOOLS", "builtin")
    assert config._tools("VAICE_LLAMACPP_TOOLS") == []


def test_a_tool_list_is_parsed_and_trimmed(monkeypatch):
    monkeypatch.setenv("VAICE_LLAMACPP_TOOLS", "get_info, read_file")
    assert config._tools("VAICE_LLAMACPP_TOOLS") == ["get_info", "read_file"]


def test_a_bad_tool_list_is_a_clean_error_not_a_traceback(monkeypatch):
    for raw in ("get info", "Get_Info", "read_file,,get_info", ""):
        monkeypatch.setenv("VAICE_LLAMACPP_TOOLS", raw)
        with pytest.raises(SystemExit, match="VAICE_LLAMACPP_TOOLS"):
            config._tools("VAICE_LLAMACPP_TOOLS")


def test_flag_default_and_override(monkeypatch):
    monkeypatch.delenv("VAICE_CAN_INTERRUPT", raising=False)
    assert config._flag("VAICE_CAN_INTERRUPT", default=True) is True
    for raw in ("0", "false", "No", "off"):
        monkeypatch.setenv("VAICE_CAN_INTERRUPT", raw)
        assert config._flag("VAICE_CAN_INTERRUPT", default=True) is False
    for raw in ("1", "true", "YES", "on"):
        monkeypatch.setenv("VAICE_CAN_INTERRUPT", raw)
        assert config._flag("VAICE_CAN_INTERRUPT", default=True) is True


def test_a_dotenv_in_the_working_directory_fills_unset_variables(monkeypatch, tmp_path):
    (tmp_path / ".env").write_text("VAICE_PORT=7999\nVAICE_VAD_THRESHOLD=0.75\n")
    monkeypatch.delenv("VAICE_PORT", raising=False)
    monkeypatch.setenv("VAICE_VAD_THRESHOLD", "0.65")  # the real environment wins over .env
    monkeypatch.chdir(tmp_path)
    try:
        reloaded = importlib.reload(config)
        assert (reloaded.PORT, reloaded.VAD_THRESHOLD) == (7999, 0.65)
    finally:
        monkeypatch.undo()
        importlib.reload(config)


def test_the_settings_are_read_from_the_environment_at_import(monkeypatch):
    monkeypatch.setenv("VAICE_LLAMACPP_URL", "http://box:9000/v1")
    monkeypatch.setenv("VAICE_LLAMACPP_PROMPT", "custom prompt")
    monkeypatch.setenv("VAICE_LLAMACPP_TIMEOUT", "7")
    monkeypatch.setenv("VAICE_VAD_THRESHOLD", "0.8")
    monkeypatch.setenv("VAICE_CAN_INTERRUPT", "0")
    try:
        reloaded = importlib.reload(config)
        assert (reloaded.LLAMACPP_URL, reloaded.LLAMACPP_PROMPT, reloaded.LLAMACPP_TIMEOUT) == (
            "http://box:9000",
            "custom prompt",
            7.0,
        )
        assert (reloaded.VAD_THRESHOLD, reloaded.CAN_INTERRUPT) == (0.8, False)
    finally:
        monkeypatch.undo()
        importlib.reload(config)


def test_an_out_of_range_vad_threshold_is_a_clean_error(monkeypatch):
    monkeypatch.setenv("VAICE_VAD_THRESHOLD", "5")
    try:
        with pytest.raises(SystemExit, match="VAICE_VAD_THRESHOLD must be between 0 and 1"):
            importlib.reload(config)
    finally:
        monkeypatch.undo()
        importlib.reload(config)
