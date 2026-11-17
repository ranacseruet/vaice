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


def test_tools_builtin_means_every_tool_the_server_advertises(monkeypatch):
    monkeypatch.setenv("VAICE_LLAMACPP_TOOLS", "builtin")
    assert config._tools("VAICE_LLAMACPP_TOOLS") is True


def test_a_tool_list_is_rejected_the_set_is_the_server_s_decision(monkeypatch):
    monkeypatch.setenv("VAICE_LLAMACPP_TOOLS", "get_info, read_file")
    with pytest.raises(SystemExit, match="start llama-server with --tools"):
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


def test_tool_compaction_caps_are_read_from_the_environment(monkeypatch):
    monkeypatch.setenv("VAICE_LLAMACPP_TOOL_VALUE_CHARS", "120")
    monkeypatch.setenv("VAICE_LLAMACPP_TOOL_OUTPUT_CHARS", "600")
    try:
        reloaded = importlib.reload(config)
        assert (reloaded.LLAMACPP_TOOL_VALUE_CHARS, reloaded.LLAMACPP_TOOL_OUTPUT_CHARS) == (120, 600)
    finally:
        monkeypatch.undo()
        importlib.reload(config)


def test_a_non_positive_tool_compaction_cap_is_a_clean_error(monkeypatch):
    monkeypatch.setenv("VAICE_LLAMACPP_TOOL_VALUE_CHARS", "0")
    try:
        with pytest.raises(SystemExit, match="VAICE_LLAMACPP_TOOL_VALUE_CHARS"):
            importlib.reload(config)
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


@pytest.mark.parametrize("name", ["VAICE_VAD_MIN_SILENCE_MS", "VAICE_VAD_SPEECH_PAD_MS"])
def test_a_negative_vad_value_is_a_clean_error(monkeypatch, name):
    monkeypatch.setenv(name, "-1")
    try:
        with pytest.raises(SystemExit, match="must be >= 0"):
            importlib.reload(config)
    finally:
        monkeypatch.undo()
        importlib.reload(config)


def test_an_empty_vaice_data_dir_disables_persistence(monkeypatch):
    monkeypatch.setenv("VAICE_DATA_DIR", "")
    try:
        reloaded = importlib.reload(config)
        assert reloaded.DATA_DIR is None and reloaded.CONVERSATION_FILE is None
    finally:
        monkeypatch.undo()
        importlib.reload(config)


def test_an_empty_or_blank_vaice_data_dir_is_not_the_default(monkeypatch):
    monkeypatch.setenv("VAICE_DATA_DIR", "   ")  # whitespace-only also means memory only
    try:
        reloaded = importlib.reload(config)
        assert reloaded.DATA_DIR is None and reloaded.CONVERSATION_FILE is None
    finally:
        monkeypatch.undo()
        importlib.reload(config)


def test_the_server_start_settings_have_defaults(monkeypatch):
    for name in ("VAICE_LLAMACPP_MODEL", "VAICE_LLAMACPP_SERVER_ARGS", "VAICE_LLAMACPP_START_TIMEOUT"):
        monkeypatch.delenv(name, raising=False)
    try:
        reloaded = importlib.reload(config)
        assert reloaded.LLAMACPP_MODEL == "LiquidAI/LFM3-Audio-2B-GGUF:Q4_K_M"
        assert reloaded.LLAMACPP_SERVER_ARGS == []
        assert reloaded.LLAMACPP_START_TIMEOUT == 900.0
    finally:
        monkeypatch.undo()
        importlib.reload(config)


def test_the_server_start_settings_can_be_overridden(monkeypatch):
    monkeypatch.setenv("VAICE_LLAMACPP_MODEL", "me/other-GGUF:Q8_0")
    monkeypatch.setenv("VAICE_LLAMACPP_SERVER_ARGS", "--tools get_info,read_file --alias 'my model'")
    monkeypatch.setenv("VAICE_LLAMACPP_START_TIMEOUT", "30")
    try:
        reloaded = importlib.reload(config)
        assert reloaded.LLAMACPP_MODEL == "me/other-GGUF:Q8_0"
        assert reloaded.LLAMACPP_SERVER_ARGS == ["--tools", "get_info,read_file", "--alias", "my model"]
        assert reloaded.LLAMACPP_START_TIMEOUT == 30.0
    finally:
        monkeypatch.undo()
        importlib.reload(config)


def test_server_args_that_shlex_cannot_parse_name_the_variable(monkeypatch):
    monkeypatch.setenv("VAICE_LLAMACPP_SERVER_ARGS", "--alias 'unclosed")
    try:
        with pytest.raises(SystemExit, match="VAICE_LLAMACPP_SERVER_ARGS must be a shell-style argument list"):
            importlib.reload(config)
    finally:
        monkeypatch.undo()
        importlib.reload(config)


def test_the_mode_is_chat_unless_set_and_the_assistant_settings_have_defaults(monkeypatch):
    for name in ("VAICE_MODE", "VAICE_ASSISTANT_PROMPT", "VAICE_VOICE"):
        monkeypatch.delenv(name, raising=False)
    try:
        reloaded = importlib.reload(config)
        assert reloaded.MODE == "chat"
        assert reloaded.VOICE == "female"
        assert reloaded.ASSISTANT_PROMPT.startswith("You are a voice assistant on the user's Mac.")
        assert "Use a tool only when the question is about this computer" in reloaded.ASSISTANT_PROMPT
    finally:
        monkeypatch.undo()
        importlib.reload(config)


def test_shell_commands_are_off_unless_asked_for(monkeypatch):
    monkeypatch.delenv("VAICE_ASSISTANT_SHELL", raising=False)
    try:
        assert importlib.reload(config).ASSISTANT_SHELL is False
        for raw in ("1", "true", "on"):
            monkeypatch.setenv("VAICE_ASSISTANT_SHELL", raw)
            assert importlib.reload(config).ASSISTANT_SHELL is True
        monkeypatch.setenv("VAICE_ASSISTANT_SHELL", "off")
        assert importlib.reload(config).ASSISTANT_SHELL is False
        monkeypatch.setenv("VAICE_ASSISTANT_SHELL", "maybe")
        with pytest.raises(SystemExit, match="VAICE_ASSISTANT_SHELL must be a boolean"):
            importlib.reload(config)
    finally:
        monkeypatch.undo()
        importlib.reload(config)


def test_the_assistant_settings_can_be_overridden(monkeypatch):
    monkeypatch.setenv("VAICE_MODE", "assistant")
    monkeypatch.setenv("VAICE_ASSISTANT_PROMPT", "Be terse.")
    monkeypatch.setenv("VAICE_VOICE", "male")
    try:
        reloaded = importlib.reload(config)
        assert (reloaded.MODE, reloaded.ASSISTANT_PROMPT, reloaded.VOICE) == ("assistant", "Be terse.", "male")
    finally:
        monkeypatch.undo()
        importlib.reload(config)


@pytest.mark.parametrize(("name", "value", "choices"), [
    ("VAICE_MODE", "voice", "chat or assistant"),
    ("VAICE_MODE", "", "chat or assistant"),
    ("VAICE_MODE", "Assistant", "chat or assistant"),
    ("VAICE_VOICE", "robot", "female or male"),
])
def test_a_bad_choice_names_the_variable_and_the_values(monkeypatch, name, value, choices):
    monkeypatch.setenv(name, value)
    try:
        with pytest.raises(SystemExit, match=f"vaice: {name} must be {choices}, got {value!r}"):
            importlib.reload(config)
    finally:
        monkeypatch.undo()
        importlib.reload(config)
