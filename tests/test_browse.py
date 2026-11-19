"""browse.py against tests/fixtures/fake_pi.py: a stand-in for `pi --mode rpc` that speaks the protocol measured on
Pi 1.5.2 and records what it receives. The real Pi is checked by hand (see the PR), not here."""
import atexit
import json
import logging
import os
import stat
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
from threads import run_on_thread, start_thread

from vaice import browse
from vaice.browse import BrowseDeclined, BrowseError, BrowseStopped, PiAgent, write_pi_home

FAKE = Path(__file__).parent / "fixtures" / "fake_pi.py"
KEY = "sk-test-secret-key"


@pytest.fixture(autouse=True)
def quick(monkeypatch):
    """Short waits, so the failure paths take a fraction of a second."""
    monkeypatch.setattr(browse, "START_TIMEOUT", 5.0)
    monkeypatch.setattr(browse, "NEW_SESSION_TIMEOUT", 2.0)
    monkeypatch.setattr(browse, "ABORT_WAIT", 0.5)
    monkeypatch.setattr(browse, "CLOSE_WAIT", 1.0)
    monkeypatch.setattr(browse, "TERM_WAIT", 0.3)
    monkeypatch.setattr(browse, "KILL_WAIT", 2.0)


class Harness:
    """A PiAgent on the fake, its record of the fake's received commands, and the callbacks' calls."""

    def __init__(self, tmp_path: Path, **env: str):
        self.home = tmp_path / "pi"
        self.record = tmp_path / "fake.log"
        self.stderr_log = tmp_path / "logs" / "pi.log"
        self.agent = PiAgent(self.home, log_path=self.stderr_log, command=[sys.executable, str(FAKE)],
                             env={"FAKE_PI_LOG": str(self.record), **env})
        self.progress: list[tuple[str, dict]] = []
        self.asked: list[tuple[str, str]] = []
        self.answer = True
        self.stop_flag = threading.Event()

    def on_progress(self, name, args):
        self.progress.append((name, args))

    def on_ask(self, title, message, stop):
        self.asked.append((title, message))
        return self.answer

    def run(self, goal="Open x.test and read the title.", *, deadline_s=20.0, **callbacks):
        return self.agent.run(
            goal, stop=callbacks.get("stop", self.stop_flag.is_set),
            on_progress=callbacks.get("on_progress", self.on_progress),
            on_ask=callbacks.get("on_ask", self.on_ask), deadline_s=deadline_s)

    def facts(self) -> list[dict]:
        return [json.loads(line) for line in self.record.read_text().splitlines()] if self.record.exists() else []

    def commands(self) -> list[str]:
        return [f["cmd"] for f in self.facts() if "cmd" in f]

    def pids(self) -> list[int]:
        return [f["start"]["pid"] for f in self.facts() if "start" in f]


@pytest.fixture
def harness(tmp_path):
    made = []

    def make(**env):
        h = Harness(tmp_path, **env)
        made.append(h)
        return h

    yield make
    for h in made:
        h.agent.close()


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def wait_dead(pid: int, seconds: float = 3.0) -> bool:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if not alive(pid):
            return True
        time.sleep(0.02)
    return False


# --- a task


def test_a_task_returns_the_last_assistant_text_and_reports_progress(harness):
    h = harness()
    assert h.run() == "The title is Python."  # not "I'll open the page.The title...", not the empty last message
    assert h.progress == [("mcp__chrome__navigate_page", {"url": "https://x.test"})]
    assert h.commands() == ["get_commands", "new_session", "prompt"]
    prompt = next(f for f in h.facts() if f.get("cmd") == "prompt")
    assert "Open x.test and read the title." in prompt["message"]


def test_the_prompt_is_sent_only_after_new_sessions_response(harness):
    """The fake answers new_session after a delay and orphans a prompt that arrived before the answer (as Pi 1.5.2
    does: the prompt never settles). Sent back to back, this task would hang."""
    h = harness(FAKE_PI_NEW_SESSION_DELAY="0.3")
    assert h.run(deadline_s=5) == "The title is Python."
    assert not [f for f in h.facts() if f.get("orphaned")]


def test_the_fake_does_orphan_a_prompt_sent_before_new_sessions_response(tmp_path):
    """Guards the test above: the fake really punishes a back-to-back prompt, so that test can fail."""
    record = tmp_path / "fake.log"
    env = {**os.environ, "FAKE_PI_LOG": str(record), "FAKE_PI_NEW_SESSION_DELAY": "0.3"}
    proc = subprocess.Popen([sys.executable, str(FAKE)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=env)
    try:
        for rec in ({"id": "a", "type": "new_session"}, {"id": "b", "type": "prompt", "message": "hi"}):
            proc.stdin.write(json.dumps(rec).encode() + b"\n")
        proc.stdin.flush()
        time.sleep(1.0)
    finally:
        proc.kill()
        out = proc.stdout.read()
        proc.wait()
    assert b"agent_settled" not in out
    assert {"orphaned": True} in [json.loads(line) for line in record.read_text().splitlines()]


def test_a_progress_callback_that_raises_does_not_end_the_task(harness, caplog):
    h = harness()

    def boom(name, args):
        raise RuntimeError("callback bug")

    with caplog.at_level(logging.WARNING, logger="vaice.browse"):
        assert h.run(on_progress=boom) == "The title is Python."
    assert "progress callback failed" in caplog.text


def test_dialogs_vaice_cannot_answer_are_cancelled_and_notifications_ignored(harness):
    h = harness(FAKE_PI_SCENARIO="ui_noise")
    assert h.run() == "The title is Python."
    answers = {f["id"]: f for f in h.facts() if f.get("cmd") == "extension_ui_response"}
    assert answers["s1"]["cancelled"] is True and answers["i1"]["cancelled"] is True
    assert "n1" not in answers  # notify is fire-and-forget


def test_an_unreadable_line_is_skipped(harness, caplog):
    h = harness(FAKE_PI_SCENARIO="unparsed")
    with caplog.at_level(logging.WARNING, logger="vaice.browse"):
        assert h.run() == "The title is Python."
    assert "unreadable record" in caplog.text


@pytest.mark.parametrize("key", [KEY, "sk-" + "k" * 250])  # the long one spans the 200-character cut
def test_a_failed_model_call_is_an_error_without_the_key(harness, caplog, key):
    h = harness(FAKE_PI_SCENARIO="error", VAICE_PI_API_KEY=key)
    with caplog.at_level(logging.INFO, logger="vaice.browse"), pytest.raises(BrowseError) as caught:
        h.run()
    assert "model failed" in str(caught.value)
    for text in (str(caught.value), caught.value.spoken, caplog.text):
        assert key[:12] not in text  # not even a prefix of it


def test_a_prompt_that_started_no_run_is_an_error(harness):
    h = harness()
    with pytest.raises(BrowseError, match="was not run"):
        h.run("scenario=handled")


def test_tasks_run_one_after_another_in_one_process(harness):
    h = harness()
    assert h.run() == "The title is Python."
    assert h.run() == "The title is Python."
    assert len(h.pids()) == 1
    assert h.commands() == ["get_commands", "new_session", "prompt", "new_session", "prompt"]  # a new session each


# --- confirmations


def test_a_yes_lets_the_step_run(harness):
    h = harness(FAKE_PI_SCENARIO="confirm")
    assert h.run() == "Clicked it."
    assert h.asked == [("mcp__chrome__click", '{"uid":"1_2"}')]
    answer = next(f for f in h.facts() if f.get("cmd") == "extension_ui_response")
    assert answer["id"] == "q1" and answer["confirmed"] is True
    assert "abort" not in h.commands()


def test_a_no_also_aborts_so_the_model_cannot_route_around_it(harness):
    h = harness(FAKE_PI_SCENARIO="confirm")
    h.answer = False
    with pytest.raises(BrowseDeclined) as caught:
        h.run()
    assert caught.value.spoken == browse.DECLINED
    commands = h.commands()
    assert commands[-2:] == ["extension_ui_response", "abort"]
    assert not [f for f in h.facts() if f.get("routed_around")]
    assert h.run("scenario=normal") == "The title is Python."  # the same child, usable again
    assert len(h.pids()) == 1


def test_a_question_callback_that_raises_is_a_no(harness):
    h = harness(FAKE_PI_SCENARIO="confirm")

    def boom(title, message, stop):
        raise RuntimeError("ui bug")

    with pytest.raises(BrowseDeclined):
        h.run(on_ask=boom)
    assert "abort" in h.commands()


def test_a_question_nobody_answers_is_a_no_after_ask_timeout(tmp_path):
    h = Harness(tmp_path, FAKE_PI_SCENARIO="confirm")
    h.agent = PiAgent(h.home, log_path=h.stderr_log, command=[sys.executable, str(FAKE)], ask_timeout_s=0.3,
                      env={"FAKE_PI_LOG": str(h.record), "FAKE_PI_SCENARIO": "confirm"})

    def wait_for_stop(title, message, stop):
        while not stop():  # the contract: on_ask returns when it is told to stop
            time.sleep(0.02)
        return False

    try:
        started = time.monotonic()
        with pytest.raises(BrowseDeclined):
            h.run(on_ask=wait_for_stop)
        assert time.monotonic() - started < 3
        assert "abort" in h.commands()
    finally:
        h.agent.close()


def test_the_deadline_ends_an_open_question_and_approves_nothing(harness):
    h = harness(FAKE_PI_SCENARIO="confirm")
    approve_late = []

    def wait_then_approve(title, message, stop):
        while not stop():  # honors stop, then says yes: too late, the task is out of time
            time.sleep(0.02)
        approve_late.append(True)
        return True

    started = time.monotonic()
    with pytest.raises(BrowseError) as caught:
        h.run(on_ask=wait_then_approve, deadline_s=0.4)
    assert not isinstance(caught.value, BrowseDeclined)
    assert caught.value.spoken == "I stopped the browser task after 0.4 seconds."
    assert time.monotonic() - started < 2  # not the 60 s question timeout
    assert approve_late
    answers = [f for f in h.facts() if f.get("cmd") == "extension_ui_response"]
    assert answers and answers[0]["confirmed"] is False
    assert "abort" in h.commands()


def test_a_yes_after_the_question_timed_out_approves_nothing(tmp_path):
    h = Harness(tmp_path, FAKE_PI_SCENARIO="confirm")
    h.agent = PiAgent(h.home, log_path=h.stderr_log, command=[sys.executable, str(FAKE)], ask_timeout_s=0.1,
                      env={"FAKE_PI_LOG": str(h.record), "FAKE_PI_SCENARIO": "confirm"})

    def wait_then_approve(title, message, stop):
        while not stop():
            time.sleep(0.02)
        return True  # too late: the question expired

    try:
        with pytest.raises(BrowseDeclined):
            h.run(on_ask=wait_then_approve)
        answers = [f for f in h.facts() if f.get("cmd") == "extension_ui_response"]
        assert answers and answers[0]["confirmed"] is False
        assert "abort" in h.commands()
    finally:
        h.agent.close()


def test_a_stop_during_a_question_is_a_stop_not_a_refusal(harness):
    h = harness(FAKE_PI_SCENARIO="confirm")

    def stop_while_asked(title, message, stop):
        h.stop_flag.set()
        return False

    with pytest.raises(BrowseStopped):
        h.run(on_ask=stop_while_asked)
    assert "abort" in h.commands()


# --- stopping


def test_stop_aborts_the_task_and_keeps_the_process(harness):
    h = harness(FAKE_PI_SCENARIO="slow")

    def stop_at_first_tool(name, args):
        h.stop_flag.set()

    started = time.monotonic()
    with pytest.raises(BrowseStopped):
        h.run(on_progress=stop_at_first_tool)
    assert time.monotonic() - started < 3  # abort settles at once; nothing waits for ABORT_WAIT
    assert h.commands()[-1] == "abort"
    h.stop_flag.clear()
    assert h.run("scenario=normal") == "The title is Python."
    assert len(h.pids()) == 1


def test_a_stop_before_anything_starts_nothing(harness):
    h = harness()
    h.stop_flag.set()
    with pytest.raises(BrowseStopped):
        h.run()
    assert h.agent._child is None
    assert h.pids() == []


def test_the_deadline_aborts_with_a_spoken_sentence(harness):
    h = harness(FAKE_PI_SCENARIO="slow")
    with pytest.raises(BrowseError) as caught:
        h.run(deadline_s=0.6)
    assert caught.value.spoken == "I stopped the browser task after 0.6 seconds."
    assert h.commands()[-1] == "abort"
    assert h.run("scenario=normal") == "The title is Python."
    assert len(h.pids()) == 1


@pytest.mark.parametrize("ignore_term", ["0", "1"])
def test_an_abort_that_never_settles_ends_the_process_and_the_next_task_restarts(harness, ignore_term):
    h = harness(FAKE_PI_SCENARIO="abort_ignored", FAKE_PI_IGNORE_TERM=ignore_term)
    stopper = threading.Timer(0.3, h.stop_flag.set)
    stopper.start()
    with pytest.raises(BrowseStopped):
        h.run()
    stopper.join()
    first = h.pids()[0]
    assert wait_dead(first)  # SIGTERM, or SIGKILL when the child ignores it
    h.stop_flag.clear()
    assert h.run("scenario=normal") == "The title is Python."
    assert len(h.pids()) == 2 and alive(h.pids()[1])


def test_end_kills_a_group_member_that_ignores_sigterm_after_pi_exits(tmp_path):
    """Pi exits on SIGTERM, a descendant in its group ignores it and keeps stdout open: end() must still SIGKILL
    the group and return without waiting on the pipe."""
    pid_file = tmp_path / "descendant.pid"
    script = (
        "import os, signal, subprocess, sys, time\n"
        # the descendant writes its pid only once it ignores SIGTERM, so end() cannot race its setup
        "kid = subprocess.Popen([sys.executable, '-c', 'import os, signal, sys, time; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); open(sys.argv[1], \"w\").write(str(os.getpid())); "
        f"time.sleep(60)', {str(pid_file)!r}])\n"
        "print('{}', flush=True)\n"
        "time.sleep(60)\n")
    proc = subprocess.Popen([sys.executable, "-c", script], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            start_new_session=True)
    child = browse._Child(proc)
    deadline = time.monotonic() + 5
    while not (pid_file.exists() and pid_file.read_text()) and time.monotonic() < deadline:
        time.sleep(0.02)
    descendant = int(pid_file.read_text())
    started = time.monotonic()
    child.end()
    assert time.monotonic() - started < browse.TERM_WAIT + browse.KILL_WAIT + 1
    assert wait_dead(descendant)
    assert proc.returncode is not None


def test_a_child_that_exits_mid_task_is_an_error_and_the_next_task_restarts(harness):
    h = harness(FAKE_PI_SCENARIO="exit_midtask")
    with pytest.raises(BrowseError, match="exited"):
        h.run()
    assert h.run("scenario=normal") == "The title is Python."
    assert len(h.pids()) == 2


def test_a_new_session_that_never_answers_ends_the_child(harness, monkeypatch):
    monkeypatch.setattr(browse, "NEW_SESSION_TIMEOUT", 0.3)
    h = harness(FAKE_PI_NEW_SESSION_DELAY="3")
    with pytest.raises(BrowseError, match="did not answer"):
        h.run()
    assert wait_dead(h.pids()[0])
    assert "prompt" not in h.commands()  # never sent without the new session


def test_an_extension_error_aborts_the_task(harness):
    h = harness(FAKE_PI_SCENARIO="ext_error")
    with pytest.raises(BrowseError, match="extension"):
        h.run()
    assert h.commands()[-1] == "abort"


def test_a_child_without_the_gate_is_refused(harness):
    h = harness(FAKE_PI_NO_GATE="1")
    with pytest.raises(BrowseError, match="tool gate") as caught:
        h.run()
    assert caught.value.spoken == browse.CANT_START
    assert wait_dead(h.pids()[0])
    assert "new_session" not in h.commands() and "prompt" not in h.commands()


def test_a_missing_pi_is_a_spoken_error(tmp_path):
    agent = PiAgent(tmp_path / "pi", log_path=tmp_path / "pi.log", command="/nonexistent/pi")
    with pytest.raises(BrowseError) as caught:
        agent.run("x", stop=lambda: False, on_progress=lambda n, a: None, on_ask=lambda t, m, s: True, deadline_s=5)
    assert caught.value.spoken == browse.CANT_START


# --- threads, close


def test_tasks_from_different_threads_share_the_child(harness):
    h = harness()
    for i in range(3):
        assert run_on_thread(h.run, f"fastrtc-worker-{i}") == "The title is Python."
    assert len(h.pids()) == 1


def test_one_task_at_a_time_and_a_waiting_one_honors_stop(harness):
    h = harness(FAKE_PI_SCENARIO="slow")
    first = start_thread(lambda: h.run(deadline_s=10), "task-1")
    deadline = time.monotonic() + 5
    while "prompt" not in h.commands() and time.monotonic() < deadline:
        time.sleep(0.02)
    second_stop = threading.Event()
    second = start_thread(lambda: h.run(stop=second_stop.is_set), "task-2")
    time.sleep(0.3)
    assert second.thread.is_alive()  # waiting for the first
    second_stop.set()
    assert isinstance(second.join(), BrowseStopped)
    h.stop_flag.set()
    assert isinstance(first.join(), BrowseStopped)
    assert h.commands().count("prompt") == 1  # the second never reached the child


def test_close_ends_the_process_is_idempotent_and_unregisters_the_backstop(harness, monkeypatch):
    registered, unregistered = [], []
    monkeypatch.setattr(atexit, "register", lambda fn, *a: registered.append(fn))
    monkeypatch.setattr(atexit, "unregister", lambda fn: unregistered.append(fn))
    h = harness()
    h.run()
    h.run()
    assert len(registered) == 1 and registered[0] == h.agent.close
    pid = h.pids()[0]
    assert alive(pid)
    h.agent.close()
    h.agent.close()
    assert wait_dead(pid)
    assert unregistered and unregistered[0] == h.agent.close
    with pytest.raises(BrowseError, match="closed"):
        h.run()


def test_close_before_any_task_is_fine(harness):
    h = harness()
    h.agent.close()
    h.agent.close()


def test_close_during_a_task_fails_that_task(harness):
    h = harness(FAKE_PI_SCENARIO="slow")
    running = start_thread(lambda: h.run(deadline_s=10), "task")
    deadline = time.monotonic() + 5
    while "prompt" not in h.commands() and time.monotonic() < deadline:
        time.sleep(0.02)
    pid = h.pids()[0]
    h.agent.close()
    assert isinstance(running.join(), BrowseError)
    assert wait_dead(pid)


# --- the child's environment


def test_the_child_runs_isolated_and_logs_to_a_private_file(harness, monkeypatch):
    monkeypatch.setenv("VAICE_BRAIN_API_KEY", "another-secret")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "aws-secret")
    h = harness(VAICE_PI_API_KEY=KEY)
    h.run()
    start = next(f["start"] for f in h.facts() if "start" in f)
    assert start["env"]["PI_CODING_AGENT_DIR"] == str(h.home)
    assert start["env"]["HOME"] == str(h.home / "home")
    assert start["env"]["VAICE_PI_API_KEY"] == KEY  # the one thing passed on purpose
    assert "VAICE_BRAIN_API_KEY" not in start["env_names"] and "AWS_SECRET_ACCESS_KEY" not in start["env_names"]
    assert Path(start["cwd"]).resolve() == (h.home / "work").resolve()
    assert json.loads(start["env"]["VAICE_GATE_POLICY"]) == browse.GATE_POLICY
    assert int(start["env"]["VAICE_GATE_TIMEOUT_MS"]) == 65000  # ask_timeout_s 60 + the grace
    argv = start["argv"]
    assert argv[:2] == ["--mode", "rpc"] and "--no-builtin-tools" in argv and "--no-extensions" not in argv
    assert argv[argv.index("-e") + 1] == str(browse.GATE_FILE)
    assert stat.S_IMODE(h.stderr_log.stat().st_mode) == 0o600
    assert "fake pi stderr line" in h.stderr_log.read_text()


# --- the config directory and the policy


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
