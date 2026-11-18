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
        assert reloaded.LLAMACPP_MODEL == "LiquidAI/LFM3-Omni-2B-GGUF:Q8_0"
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
        assert "use the tool again" in reloaded.ASSISTANT_PROMPT  # live eval: follow-ups are not answered from history
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


BRAIN_NAMES = ("VAICE_BRAIN_URL", "VAICE_BRAIN_MODEL", "VAICE_BRAIN_API_KEY", "VAICE_BRAIN_TIMEOUT")


def test_the_helper_brain_is_off_by_default(monkeypatch):
    for name in BRAIN_NAMES:
        monkeypatch.delenv(name, raising=False)
    try:
        reloaded = importlib.reload(config)
        assert (reloaded.BRAIN_URL, reloaded.BRAIN_MODEL, reloaded.BRAIN_API_KEY) == ("", "", "")
        assert reloaded.BRAIN_TIMEOUT == 60.0
    finally:
        monkeypatch.undo()
        importlib.reload(config)


def test_an_empty_brain_url_is_no_helper(monkeypatch):
    monkeypatch.setenv("VAICE_BRAIN_URL", "  ")
    try:
        assert importlib.reload(config).BRAIN_URL == ""
    finally:
        monkeypatch.undo()
        importlib.reload(config)


def test_the_helper_brain_settings_are_read_from_the_environment(monkeypatch):
    monkeypatch.setenv("VAICE_BRAIN_URL", "https://openrouter.ai/api/v1/")
    monkeypatch.setenv("VAICE_BRAIN_MODEL", " some/model ")
    monkeypatch.setenv("VAICE_BRAIN_API_KEY", "fake-key-123\n")
    monkeypatch.setenv("VAICE_BRAIN_TIMEOUT", "12.5")
    try:
        reloaded = importlib.reload(config)
        assert reloaded.BRAIN_URL == "https://openrouter.ai/api/v1"  # trailing slash stripped, path kept
        assert reloaded.BRAIN_MODEL == "some/model"
        assert reloaded.BRAIN_API_KEY == "fake-key-123"  # a pasted newline is not part of the key
        assert reloaded.BRAIN_TIMEOUT == 12.5
    finally:
        monkeypatch.undo()
        importlib.reload(config)


@pytest.mark.parametrize(("given", "base"), [
    ("http://localhost:1234/v1", "http://localhost:1234/v1"),
    ("http://localhost:1234/v1/", "http://localhost:1234/v1"),
    ("https://api.openai.com/v1", "https://api.openai.com/v1"),
    ("localhost:11434/v1", "http://localhost:11434/v1"),  # no scheme: plain http, as for the llama-server URL
    ("http://localhost:11434", "http://localhost:11434"),
])
def test_a_brain_base_url_keeps_its_path_and_loses_the_trailing_slash(given, base):
    assert config.brain_base(given) == base


@pytest.mark.parametrize("given", ["http://host/v1/chat/completions", "http://host/v1/chat/completions/"])
def test_the_endpoint_instead_of_the_base_is_a_clear_error(given):
    with pytest.raises(ValueError, match=r"base URL.*not the endpoint"):
        config.brain_base(given)


@pytest.mark.parametrize("given", ["ftp://host/v1", "http://", "ws://host:1/v1"])
def test_a_brain_url_that_is_not_http_is_refused(given):
    with pytest.raises(ValueError, match="http"):
        config.brain_base(given)


@pytest.mark.parametrize("given", ["https://user:fake-key-123@host/v1", "https://host/v1?key=fake-key-123",
                                   "https://host/v1#fake-key-123"])
def test_credentials_a_query_or_a_fragment_in_the_brain_url_are_refused_without_echoing_them(given):
    with pytest.raises(ValueError, match="credentials, a query or a fragment") as caught:
        config.brain_base(given)

    assert "fake-key-123" not in str(caught.value)


def test_a_bad_brain_url_names_the_variable_and_never_repeats_the_value(monkeypatch):
    monkeypatch.setenv("VAICE_BRAIN_URL", "https://host/v1?key=fake-key-123")
    try:
        with pytest.raises(SystemExit, match="vaice: VAICE_BRAIN_URL: it must not contain credentials") as caught:
            importlib.reload(config)
        assert "fake-key-123" not in str(caught.value)

        monkeypatch.setenv("VAICE_BRAIN_URL", "http://host/v1/chat/completions")
        with pytest.raises(SystemExit, match="VAICE_BRAIN_URL: give the base URL"):
            importlib.reload(config)
    finally:
        monkeypatch.undo()
        importlib.reload(config)


@pytest.mark.parametrize("value", ["0", "-5", "soon"])
def test_a_bad_brain_timeout_is_a_clean_error(monkeypatch, value):
    monkeypatch.setenv("VAICE_BRAIN_TIMEOUT", value)
    try:
        with pytest.raises(SystemExit, match="VAICE_BRAIN_TIMEOUT must be"):
            importlib.reload(config)
    finally:
        monkeypatch.undo()
        importlib.reload(config)


VISION_NAMES = ("VAICE_VISION_URL", "VAICE_VISION_MODEL", "VAICE_VISION_API_KEY", "VAICE_VISION_TIMEOUT",
                "VAICE_VISION_MAX_TOKENS")


def test_the_vision_skill_is_off_by_default(monkeypatch):
    for name in VISION_NAMES:
        monkeypatch.delenv(name, raising=False)
    try:
        reloaded = importlib.reload(config)
        assert (reloaded.VISION_URL, reloaded.VISION_MODEL, reloaded.VISION_API_KEY) == ("", "", "")
        assert reloaded.VISION_TIMEOUT == 30.0
        assert reloaded.VISION_MAX_TOKENS == 200
    finally:
        monkeypatch.undo()
        importlib.reload(config)


def test_the_vision_settings_are_read_from_the_environment(monkeypatch):
    monkeypatch.setenv("VAICE_VISION_URL", "http://127.0.0.1:8002/v1/")
    monkeypatch.setenv("VAICE_VISION_MODEL", " omni ")
    monkeypatch.setenv("VAICE_VISION_API_KEY", "fake-key-123\n")
    monkeypatch.setenv("VAICE_VISION_TIMEOUT", "12.5")
    try:
        reloaded = importlib.reload(config)
        assert reloaded.VISION_URL == "http://127.0.0.1:8002/v1"  # trailing slash stripped, /v1 kept
        assert reloaded.VISION_MODEL == "omni"
        assert reloaded.VISION_API_KEY == "fake-key-123"
        assert reloaded.VISION_TIMEOUT == 12.5
        assert reloaded.BRAIN_URL == ""  # the helper brain is a separate setting
    finally:
        monkeypatch.undo()
        importlib.reload(config)


def test_a_bad_vision_url_names_the_variable_and_never_repeats_the_value(monkeypatch):
    monkeypatch.setenv("VAICE_VISION_URL", "https://host/v1?key=fake-key-123")
    try:
        with pytest.raises(SystemExit, match="vaice: VAICE_VISION_URL: it must not contain credentials") as caught:
            importlib.reload(config)
        assert "VAICE_VISION_API_KEY" in str(caught.value) and "VAICE_BRAIN" not in str(caught.value)
        assert "fake-key-123" not in str(caught.value)

        monkeypatch.setenv("VAICE_VISION_URL", "http://host/v1/chat/completions")
        with pytest.raises(SystemExit, match="VAICE_VISION_URL: give the base URL"):
            importlib.reload(config)

        monkeypatch.setenv("VAICE_VISION_URL", "ftp://host/v1")
        with pytest.raises(SystemExit, match="VAICE_VISION_URL: expected an http"):
            importlib.reload(config)
    finally:
        monkeypatch.undo()
        importlib.reload(config)


def test_a_bad_brain_url_still_names_the_brain_key():
    with pytest.raises(ValueError, match="VAICE_BRAIN_API_KEY"):
        config.brain_base("https://user:pw@host/v1")


def test_the_vision_token_cap_can_be_changed_or_switched_off(monkeypatch):
    try:
        for value, expected in (("120", 120), ("0", 0)):
            monkeypatch.setenv("VAICE_VISION_MAX_TOKENS", value)
            reloaded = importlib.reload(config)
            assert expected == reloaded.VISION_MAX_TOKENS
    finally:
        monkeypatch.undo()
        importlib.reload(config)


@pytest.mark.parametrize("value", ["-1", "lots"])
def test_a_bad_vision_token_cap_is_a_clean_error(monkeypatch, value):
    monkeypatch.setenv("VAICE_VISION_MAX_TOKENS", value)
    try:
        with pytest.raises(SystemExit, match="VAICE_VISION_MAX_TOKENS must be"):
            importlib.reload(config)
    finally:
        monkeypatch.undo()
        importlib.reload(config)


@pytest.mark.parametrize("value", ["0", "-5", "soon"])
def test_a_bad_vision_timeout_is_a_clean_error(monkeypatch, value):
    monkeypatch.setenv("VAICE_VISION_TIMEOUT", value)
    try:
        with pytest.raises(SystemExit, match="VAICE_VISION_TIMEOUT must be"):
            importlib.reload(config)
    finally:
        monkeypatch.undo()
        importlib.reload(config)


def test_spoken_progress_is_on_after_two_and_a_half_seconds_and_can_be_turned_off(monkeypatch):
    monkeypatch.delenv("VAICE_ASSISTANT_PROGRESS_SECONDS", raising=False)
    try:
        assert importlib.reload(config).ASSISTANT_PROGRESS_SECONDS == 2.5
        monkeypatch.setenv("VAICE_ASSISTANT_PROGRESS_SECONDS", "0")
        assert importlib.reload(config).ASSISTANT_PROGRESS_SECONDS == 0
        monkeypatch.setenv("VAICE_ASSISTANT_PROGRESS_SECONDS", "4.5")
        assert importlib.reload(config).ASSISTANT_PROGRESS_SECONDS == 4.5
    finally:
        monkeypatch.undo()
        importlib.reload(config)


@pytest.mark.parametrize("value", ["-1", "soon", "nan"])
def test_a_bad_progress_setting_is_a_clean_error(monkeypatch, value):
    monkeypatch.setenv("VAICE_ASSISTANT_PROGRESS_SECONDS", value)
    try:
        with pytest.raises(SystemExit, match="VAICE_ASSISTANT_PROGRESS_SECONDS must be"):
            importlib.reload(config)
    finally:
        monkeypatch.undo()
        importlib.reload(config)


def test_assistant_memory_is_off_unless_switched_on(monkeypatch):
    monkeypatch.delenv("VAICE_ASSISTANT_MEMORY", raising=False)
    try:
        assert importlib.reload(config).ASSISTANT_MEMORY is False
        monkeypatch.setenv("VAICE_ASSISTANT_MEMORY", "true")
        assert importlib.reload(config).ASSISTANT_MEMORY is True
        for raw in ("0", "false", "off"):
            monkeypatch.setenv("VAICE_ASSISTANT_MEMORY", raw)
            assert importlib.reload(config).ASSISTANT_MEMORY is False
        monkeypatch.setenv("VAICE_ASSISTANT_MEMORY", "maybe")
        with pytest.raises(SystemExit, match="VAICE_ASSISTANT_MEMORY must be a boolean"):
            importlib.reload(config)
    finally:
        monkeypatch.undo()
        importlib.reload(config)


def test_the_notes_file_lives_in_the_data_dir_and_is_none_without_one(monkeypatch, tmp_path):
    monkeypatch.setenv("VAICE_DATA_DIR", str(tmp_path))
    try:
        assert tmp_path / "memory.json" == importlib.reload(config).MEMORY_FILE
        monkeypatch.setenv("VAICE_DATA_DIR", "")
        assert importlib.reload(config).MEMORY_FILE is None  # in memory only, like the conversation
    finally:
        monkeypatch.undo()
        importlib.reload(config)


def test_mcp_is_off_by_default(monkeypatch):
    for name in ("VAICE_MCP_CONFIG", "VAICE_MCP_TIMEOUT"):
        monkeypatch.delenv(name, raising=False)
    try:
        reloaded = importlib.reload(config)
        assert reloaded.MCP_CONFIG is None
        assert reloaded.MCP_TIMEOUT == 30.0
    finally:
        monkeypatch.undo()
        importlib.reload(config)


def test_the_mcp_settings_are_read_from_the_environment_and_the_path_is_expanded(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("VAICE_MCP_CONFIG", " ~/mcp.json ")
    monkeypatch.setenv("VAICE_MCP_TIMEOUT", "7.5")
    try:
        reloaded = importlib.reload(config)
        assert (tmp_path / "mcp.json") == reloaded.MCP_CONFIG
        assert reloaded.MCP_TIMEOUT == 7.5
    finally:
        monkeypatch.undo()
        importlib.reload(config)


def test_an_empty_mcp_config_means_off(monkeypatch):
    monkeypatch.setenv("VAICE_MCP_CONFIG", "  ")
    try:
        assert importlib.reload(config).MCP_CONFIG is None
    finally:
        monkeypatch.undo()
        importlib.reload(config)


@pytest.mark.parametrize("value", ["0", "-5", "soon"])
def test_a_bad_mcp_timeout_is_a_clean_error(monkeypatch, value):
    monkeypatch.setenv("VAICE_MCP_TIMEOUT", value)
    try:
        with pytest.raises(SystemExit, match="VAICE_MCP_TIMEOUT must be"):
            importlib.reload(config)
    finally:
        monkeypatch.undo()
        importlib.reload(config)
