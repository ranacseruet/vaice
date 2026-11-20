"""browse.py: the settings of Pi's config directory, the launcher and the environments the two presets give an agent,
the gate preflight, what is left to clean up after a killed browser agent, and the Pi version check. The agent itself is
tested in test_acp_agent.py; the real agents are checked by hand (see the PR), not here."""
import json
import logging
import os
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest

from vaice import browse
from vaice.browse import write_pi_home

KEY = "sk-test-secret-key"


# --- the presets' environments

def test_the_ungated_policy_runs_every_browser_tool_and_still_asks_about_anything_else():
    policy = browse.UNGATED_POLICY
    assert policy == {"mcp__chrome__*": "run", "default": "ask"}
    assert not any(key.startswith("mcp__chrome__") and value != "run" for key, value in policy.items())


PI_NONCE = "0123456789abcdef"


def test_the_pi_environment_is_an_allowlist_plus_the_gate_settings_and_the_key(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", "/usr/bin")
    monkeypatch.setenv("VAICE_VISION_API_KEY", "another-secret")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "aws-secret")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-secret")
    home = tmp_path / "pi"
    env = browse.pi_environment(home, ask_timeout_s=60, gate_policy=browse.GATE_POLICY, nonce=PI_NONCE, api_key=KEY)
    assert env["PATH"] == "/usr/bin"
    assert env["PI_CODING_AGENT_DIR"] == str(home) and env["HOME"] == str(home / "home")
    assert env[browse.API_KEY_ENV] == KEY  # the one secret passed on purpose
    assert json.loads(env["VAICE_GATE_POLICY"]) == browse.GATE_POLICY
    assert int(env["VAICE_GATE_TIMEOUT_MS"]) == 65000  # ask_timeout_s 60 + the grace
    assert env["VAICE_GATE_NONCE"] == PI_NONCE
    assert env["PI_ACP_PI_COMMAND"] == str(browse.launcher_path(home))
    assert (env["PI_OFFLINE"], env["PI_SKIP_VERSION_CHECK"], env["PI_TELEMETRY"]) == ("1", "1", "0")
    assert env["npm_config_cache"]  # HOME moves, the package cache does not
    for name in ("VAICE_VISION_API_KEY", "AWS_SECRET_ACCESS_KEY", "OPENAI_API_KEY"):
        assert name not in env
    assert set(env) <= {*browse._INHERITED_ENV, "PI_CODING_AGENT_DIR", "HOME", "PI_OFFLINE", "PI_SKIP_VERSION_CHECK",
                        "PI_TELEMETRY", "VAICE_GATE_POLICY", "VAICE_GATE_TIMEOUT_MS", "VAICE_GATE_NONCE",
                        "PI_ACP_PI_COMMAND", browse.API_KEY_ENV}


def test_the_pi_environment_without_a_key_has_none(tmp_path):
    env = browse.pi_environment(tmp_path / "pi", ask_timeout_s=1, gate_policy=browse.UNGATED_POLICY, nonce=PI_NONCE)
    assert browse.API_KEY_ENV not in env
    assert json.loads(env["VAICE_GATE_POLICY"]) == browse.UNGATED_POLICY


def test_the_acp_environment_is_the_environment_without_the_vaice_variables():
    environ = {"PATH": "/usr/bin", "HOME": "/home/u", "ANTHROPIC_API_KEY": "sk-ant-1234567890",
               "VAICE_VISION_API_KEY": "another-secret", "VAICE_BROWSE_AGENT": "opencode acp", "VAICE_PORT": "7860"}
    assert browse.acp_environment(environ) == {"PATH": "/usr/bin", "HOME": "/home/u",
                                               "ANTHROPIC_API_KEY": "sk-ant-1234567890"}


def test_secret_values_are_the_long_enough_values_of_credential_looking_names():
    env = {"OPENAI_API_KEY": "sk-1234567890", "GITHUB_TOKEN": "ghp_abcdefgh", "DB_PASSWORD": "short",
           "PATH": "/usr/bin", "MY_CREDENTIALS": "credential-value", "TERM": "xterm-256color"}
    assert sorted(browse.secret_values(env)) == sorted(["sk-1234567890", "ghp_abcdefgh", "credential-value"])


def test_the_nonce_is_random_and_the_marker_is_the_gate_command_then_the_nonce():
    assert browse.new_nonce() != browse.new_nonce() and len(browse.new_nonce()) >= 16
    assert browse.gate_marker(PI_NONCE) == f"vaice-gate {PI_NONCE}"


def test_the_pi_agent_is_pi_acp_pinned_with_the_gate_marker_and_its_key_scrubbed(tmp_path):
    pytest.importorskip("acp")  # the optional `browse` extra
    env = browse.pi_environment(tmp_path / "pi", ask_timeout_s=60, gate_policy=browse.GATE_POLICY, nonce=PI_NONCE,
                                api_key=KEY)
    agent = browse.pi_agent(tmp_path / "pi", env=env, nonce=PI_NONCE, log_path=tmp_path / "pi.log", ask_timeout_s=60)
    try:
        assert agent._argv == ["npx", "-y", browse.PI_ACP] and browse.PI_ACP == "pi-acp@0.0.34"
        assert agent._marker == f"vaice-gate {PI_NONCE}" and agent.permissions == "ask" and agent.pi_dialogs
        assert agent._mcp_servers == []  # pi-acp ignores them: Pi's own mcp.json holds the browser server
        assert agent._cwd == browse.work_dir(tmp_path / "pi") and agent._secrets == (KEY,)
        assert agent._scrub(f"Bearer {KEY}") == "Bearer ***"
    finally:
        agent.close()


def test_the_acp_agent_passes_the_pinned_browser_server_and_the_environment_without_vaice_variables(tmp_path):
    pytest.importorskip("acp")  # the optional `browse` extra
    environ = {"PATH": "/usr/bin", "ANTHROPIC_API_KEY": "sk-ant-1234567890", "VAICE_VISION_API_KEY": "another-secret"}
    agent = browse.acp_agent(["opencode", "acp"], work=tmp_path / "acp-work", log_path=tmp_path / "acp.log",
                             browser_args=["--userDataDir=/p"], ask_timeout_s=60, environ=environ)
    try:
        assert agent._argv == ["opencode", "acp"] and agent.permissions == "ask" and agent._marker is None
        assert agent._env == {"PATH": "/usr/bin", "ANTHROPIC_API_KEY": "sk-ant-1234567890"}
        assert agent._secrets == ("sk-ant-1234567890",)
        (chrome,) = agent._mcp_servers
        assert (chrome.name, chrome.command, chrome.env) == ("chrome", "npx", [])
        assert chrome.args == ["-y", browse.CHROME_DEVTOOLS_MCP, "--no-usage-statistics", "--no-performance-crux",
                               "--userDataDir=/p"]
        assert stat.S_IMODE((tmp_path / "acp-work").stat().st_mode) == 0o700
    finally:
        agent.close()


def test_the_acp_agent_allows_each_step_without_asking_when_confirmation_is_off(tmp_path):
    pytest.importorskip("acp")  # the optional `browse` extra
    agent = browse.acp_agent(["x"], work=tmp_path / "w", log_path=tmp_path / "log", browser_args=[], ask_timeout_s=60,
                             confirm=False, environ={})
    try:
        assert agent.permissions == "allow"
    finally:
        agent.close()


# --- the launcher


def test_the_launcher_runs_pi_with_the_isolation_flags_and_the_gate_extension_and_nothing_else(tmp_path):
    path = browse.launcher_path(tmp_path)
    browse.write_launcher(path)
    assert stat.S_IMODE(path.stat().st_mode) == 0o700
    text = path.read_text()
    assert text.startswith("#!/bin/sh\n") and text.rstrip().endswith('"$@"')
    for flag in ("--no-session", "--no-builtin-tools", "--no-context-files", "--no-skills", "--no-prompt-templates",
                 "--offline"):
        assert flag in text
    assert "--no-extensions" not in text  # that would turn off Pi's built-in MCP client
    assert f"-e {browse.GATE_FILE}" in text or f"-e '{browse.GATE_FILE}'" in text
    assert not list(tmp_path.glob("*.tmp"))


def test_the_launcher_quotes_a_path_with_spaces(tmp_path, monkeypatch):
    monkeypatch.setattr(browse, "GATE_FILE", tmp_path / "a dir" / "gate.ts")
    path = browse.launcher_path(tmp_path)
    browse.write_launcher(path)
    assert "'" + str(tmp_path / "a dir" / "gate.ts") + "'" in path.read_text()


# --- the gate preflight: a tiny `pi` that answers get_commands (or does not)


def stub_home(tmp_path, body: str) -> Path:
    """A Pi config directory whose launcher is `body` (a shell script) instead of the real one."""
    home = tmp_path / "pi"
    home.mkdir()
    launcher = browse.launcher_path(home)
    launcher.write_text(f"#!/bin/sh\n{body}\n")
    launcher.chmod(0o700)
    return home


def answering(commands: list[str]) -> str:
    """A launcher that reads the request, answers it with `commands` and exits (as Pi does when its stdin closes)."""
    data = json.dumps({"id": "g1", "type": "response", "command": "get_commands", "success": True,
                       "data": {"commands": [{"name": name} for name in commands]}})
    return f"cat > /dev/null\necho '{json.dumps({'type': 'noise'})}'\necho '{data}'"


def test_the_gate_check_is_true_when_pi_lists_the_gate_command(tmp_path):
    home = stub_home(tmp_path, answering(["compact", "vaice-gate"]))
    assert browse.check_gate(home, {"PATH": os.environ["PATH"]}, log_path=tmp_path / "pi.log") is True


def test_the_gate_check_is_false_when_the_gate_extension_did_not_load(tmp_path):
    home = stub_home(tmp_path, answering(["compact"]))
    assert browse.check_gate(home, {"PATH": os.environ["PATH"]}) is False


def test_the_gate_check_gets_the_get_commands_request_on_stdin_and_the_environment_it_was_given(tmp_path):
    record = tmp_path / "seen"
    home = stub_home(tmp_path, f'head -n 1 > {record}\necho "$VAICE_GATE_NONCE" >> {record}\n'
                                  + answering(["vaice-gate"]).split("\n", 1)[1])
    env = {"PATH": os.environ["PATH"], "VAICE_GATE_NONCE": PI_NONCE}
    assert browse.check_gate(home, env) is True
    request, nonce = record.read_text().splitlines()
    assert json.loads(request) == {"id": "g1", "type": "get_commands"} and nonce == PI_NONCE


@pytest.mark.parametrize("body", ["exit 3", "echo not json", "cat > /dev/null", "exec /nonexistent/pi"])
def test_the_gate_check_is_false_for_a_pi_that_fails_or_says_nothing_useful(tmp_path, body):
    assert browse.check_gate(stub_home(tmp_path, body), {"PATH": os.environ["PATH"]}) is False


def test_the_gate_check_is_false_when_the_launcher_is_missing(tmp_path):
    assert browse.check_gate(tmp_path / "pi", {"PATH": os.environ["PATH"]}) is False


def test_the_gate_check_ends_a_pi_that_does_not_answer_in_time(tmp_path, monkeypatch):
    monkeypatch.setattr(browse, "START_TIMEOUT", 0.5)
    monkeypatch.setattr(browse, "TERM_WAIT", 0.3)
    home = stub_home(tmp_path, "sleep 60")
    started = time.monotonic()
    assert browse.check_gate(home, {"PATH": os.environ["PATH"]}) is False
    assert time.monotonic() - started < 5


def test_the_gate_check_logs_pis_stderr_to_a_private_file(tmp_path):
    home = stub_home(tmp_path, "echo 'pi complaint' >&2\nexit 1")
    log = tmp_path / "pi.log"
    browse.check_gate(home, {"PATH": os.environ["PATH"]}, log_path=log)
    assert stat.S_IMODE(log.stat().st_mode) == 0o600 and "pi complaint" in log.read_text()


# --- the Pi's config directory and the policy


def test_write_pi_home_writes_private_files_without_the_key(tmp_path, monkeypatch):
    monkeypatch.setenv(browse.API_KEY_ENV, KEY)
    home = tmp_path / "pi"
    write_pi_home(home, base_url="http://localhost:1234/v1", model="lfm2.5-24b-a2b", api_key_env=browse.API_KEY_ENV,
                  browser_args=["--headless", "--isolated", "--chromeArg=--mute-audio"])
    models = json.loads((home / "models.json").read_text())
    provider = models["providers"]["vaice"]
    assert provider == {"baseUrl": "http://localhost:1234/v1", "api": "openai-completions",
                        "apiKey": "$VAICE_PI_API_KEY", "models": [{"id": "lfm2.5-24b-a2b"}]}
    settings = json.loads((home / "settings.json").read_text())
    assert settings["defaultProvider"] == "vaice" and settings["defaultModel"] == "lfm2.5-24b-a2b"
    chrome = json.loads((home / "mcp.json").read_text())["mcpServers"]["chrome"]
    assert chrome["command"] == "npx" and chrome["exposure"] == "direct"
    assert chrome["args"][:2] == ["-y", "chrome-devtools-mcp@1.13.0"]
    assert chrome["args"][-3:] == ["--headless", "--isolated", "--chromeArg=--mute-audio"]
    for name in ("models.json", "settings.json", "mcp.json"):
        assert stat.S_IMODE((home / name).stat().st_mode) == 0o600
        assert KEY not in (home / name).read_text()
    for directory in (home, home / "home", home / "work"):
        assert stat.S_IMODE(directory.stat().st_mode) == 0o700


def test_write_pi_home_without_a_key_writes_a_placeholder_and_rewrites_cleanly(tmp_path):
    home = tmp_path / "pi"
    write_pi_home(home, base_url="http://localhost:1234/v1", model="m1", api_key_env=None, browser_args=[])
    write_pi_home(home, base_url="http://localhost:1234/v1", model="m2", api_key_env=None,
                  browser_args=browse.default_browser_args(tmp_path / "profile"))
    provider = json.loads((home / "models.json").read_text())["providers"]["vaice"]
    assert provider["apiKey"] == "vaice" and provider["models"] == [{"id": "m2"}]
    args = json.loads((home / "mcp.json").read_text())["mcpServers"]["chrome"]["args"]
    assert args[-1] == f"--userDataDir={tmp_path / 'profile'}"
    assert "--headless" not in args and "--isolated" not in args  # the default is a visible window, a kept profile
    assert sorted(p.name for p in home.iterdir()) == ["home", "mcp.json", "models.json", "settings.json", "work"]
    assert json.loads((home / "settings.json").read_text())["quietStartup"] is True


@pytest.mark.parametrize("name", ["", "A-B", "1X", "A B", "$X"])
def test_write_pi_home_refuses_a_bad_key_variable_name(tmp_path, name):
    with pytest.raises(ValueError):
        write_pi_home(tmp_path / "pi", base_url="http://x/v1", model="m", api_key_env=name, browser_args=[])


def test_the_gate_policy_asks_for_everything_that_changes_a_page():
    policy = browse.GATE_POLICY
    assert policy["mcp__chrome__*"] == "run" and policy["default"] == "ask"
    mutating = ("click", "fill", "fill_form", "type_text", "press_key", "evaluate_script", "upload_file", "drag",
                "handle_dialog")
    assert {name: policy[f"mcp__chrome__{name}"] for name in mutating} == dict.fromkeys(mutating, "ask")
    assert set(policy) == {"mcp__chrome__*", "default", *(f"mcp__chrome__{n}" for n in mutating)}
    assert json.loads(json.dumps(policy)) == policy


def test_the_gate_extension_is_shipped_and_names_the_command_the_agent_checks():
    source = browse.GATE_FILE.read_text()
    assert f'registerCommand("{browse.GATE_COMMAND}"' in source
    assert "VAICE_GATE_POLICY" in source and "VAICE_GATE_TIMEOUT_MS" in source


# --- what a killed Pi leaves behind: the browser and its MCP server are in no group of Pi's

PROFILE_FLAGS = ("--user-data-dir", "--userDataDir")


def holder(flag: str, path: Path) -> subprocess.Popen:
    """A process whose command line holds `flag=path` (python ignores what follows `-c`'s script), in a group of
    its own like Chrome and chrome-devtools-mcp are."""
    return subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)", f"{flag}={path}"],
                            start_new_session=True)


def reaped(proc: subprocess.Popen, seconds: float = 5.0) -> bool:
    try:
        proc.wait(seconds)
    except subprocess.TimeoutExpired:
        return False
    return True


def test_profile_holders_are_matched_on_the_exact_path_with_either_flag_and_never_this_process(tmp_path):
    profile = tmp_path / "chrome-profile"
    listing = "\n".join([
        f"101 /Applications/Chrome --user-data-dir={profile} --remote-debugging-pipe",
        f"102 node chrome-devtools-mcp --no-usage-statistics --userDataDir={profile}",
        f"103 node chrome-devtools-mcp --userDataDir={profile}-2",  # a sibling profile: not ours
        f"104 /Applications/Chrome --user-data-dir={profile}/Default",  # a directory inside it: not the profile
        f"105 /Applications/Chrome --user-data-dir={tmp_path / 'other'}",
        f"106 vim {profile}",  # mentions the path without the flag
        f"{os.getpid()} python --user-data-dir={profile}",  # this very process
        f"107 /Applications/Chrome --user-data-dir={profile}",
        "garbage"])

    assert browse._profile_pids(profile, listing) == [101, 102, 107]


@pytest.mark.parametrize("flag", PROFILE_FLAGS)
def test_what_still_holds_the_profile_is_ended_and_a_sibling_profile_and_a_bystander_are_not(tmp_path, flag):
    profile = tmp_path / "chrome-profile"
    ours, sibling = holder(flag, profile), holder(flag, tmp_path / "chrome-profile-2")
    bystander = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True)
    try:
        time.sleep(0.3)  # let them exec, so `ps` shows their arguments

        found = browse.end_profile_processes(profile)

        assert found == [ours.pid] and reaped(ours)
        assert sibling.poll() is None and bystander.poll() is None
    finally:
        for proc in (ours, sibling, bystander):
            proc.kill()
            proc.wait()


def test_a_holder_that_ignores_sigterm_is_killed(tmp_path):
    profile = tmp_path / "chrome-profile"
    script = "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)"
    stubborn = subprocess.Popen([sys.executable, "-c", script, f"--user-data-dir={profile}"], start_new_session=True)
    try:
        time.sleep(0.3)
        started = time.monotonic()

        browse.end_profile_processes(profile)

        assert reaped(stubborn) and time.monotonic() - started < browse.TERM_WAIT + 2
    finally:
        stubborn.kill()
        stubborn.wait()


def test_a_failing_ps_is_a_warning_and_signals_nothing(tmp_path, monkeypatch, caplog):
    def broken(*args, **kwargs):
        raise OSError("no ps")

    monkeypatch.setattr(browse.subprocess, "run", broken)
    monkeypatch.setattr(browse.os, "kill", lambda *a: pytest.fail("signalled something"))
    with caplog.at_level(logging.WARNING, logger="vaice.browse"):
        assert browse.end_profile_processes(tmp_path / "chrome-profile") == []

    assert "could not look for processes" in caplog.text


# --- the version check

def script_printing(tmp_path, text, code=0):
    path = tmp_path / "pi"
    path.write_text(f"#!/bin/sh\necho '{text}'\nexit {code}\n")
    path.chmod(0o755)
    return str(path)


@pytest.mark.parametrize(("printed", "version"), [("1.5.2", (1, 5, 2)), ("pi v1.10.0 (abc)", (1, 10, 0)),
                                                  ("0.9.1-beta", (0, 9, 1)), ("garbage", None), ("1.5", None)])
def test_the_version_is_read_from_pi_dash_dash_version(tmp_path, printed, version):
    assert browse.pi_version(script_printing(tmp_path, printed)) == version


def test_an_unrunnable_pi_has_no_version(tmp_path):
    assert browse.pi_version(str(tmp_path / "missing")) is None


def test_versions_compare_as_tuples():
    assert (1, 10, 0) > browse.MIN_PI_VERSION > (1, 5, 1)
