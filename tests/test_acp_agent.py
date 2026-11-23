"""AcpAgent against tests/fixtures/fake_acp_agent.py: a scripted ACP agent built on the SDK's agent side that records
what it receives. The real agents (Pi through pi-acp, others) are checked by hand (see the PR), not here."""
import json
import logging
import os
import stat
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import pytest

pytest.importorskip("acp")  # the optional `browse` extra: these tests skip without it

from threads import run_on_thread, start_thread

from vaice import browse
from vaice.acp_agent import AcpAgent
from vaice.browse import BrowseDeclined, BrowseError, BrowseStopped

FAKE = Path(__file__).parent / "fixtures" / "fake_acp_agent.py"
KEY = "sk-test-secret-key"
MARKER = "vaice-gate abc123"
GOAL = "Open x.test and read the title."

SIMPLE = {"steps": [{"text": "I'll open the page. "}, {"tool": "mcp__chrome__navigate_page",
                                                       "args": {"url": "https://x.test"}},
                    {"text": "The title is Python."}]}


@pytest.fixture(autouse=True)
def quick(monkeypatch):
    """Short waits, so the failure paths take a fraction of a second."""
    monkeypatch.setattr(browse, "START_TIMEOUT", 5.0)
    monkeypatch.setattr(browse, "NEW_SESSION_TIMEOUT", 2.0)
    monkeypatch.setattr(browse, "ABORT_WAIT", 0.5)
    monkeypatch.setattr(browse, "CLOSE_WAIT", 1.0)
    monkeypatch.setattr(browse, "TERM_WAIT", 0.3)
    monkeypatch.setattr(browse, "KILL_WAIT", 2.0)


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


class Harness:
    """An AcpAgent on the fake, its record of what the fake received, and the callbacks' calls."""

    def __init__(self, tmp_path: Path, scenario: dict | None = None, *, scenarios: dict | None = None,
                 env: dict | None = None, **options):
        self.record = tmp_path / "fake.log"
        self.stderr_log = tmp_path / "logs" / "agent.log"
        self.cwd = tmp_path / "work"
        fake_env = {"PATH": os.environ["PATH"], "FAKE_ACP_LOG": str(self.record),
                    "FAKE_ACP_SCENARIO": json.dumps(scenario if scenario is not None else SIMPLE),
                    "FAKE_ACP_SCENARIOS": json.dumps(scenarios or {}), **(env or {})}
        self.agent = AcpAgent([sys.executable, str(FAKE)], env=fake_env, cwd=self.cwd, log_path=self.stderr_log,
                              **options)
        self.progress: list[tuple[str, dict]] = []
        self.texts: list[str] = []
        self.asked: list[tuple[str, str]] = []
        self.answer = True
        self.stop_flag = threading.Event()

    def on_progress(self, name, args):
        self.progress.append((name, args))

    def on_ask(self, title, message, stop):
        self.asked.append((title, message))
        return self.answer

    def run(self, goal=GOAL, *, deadline_s=20.0, **callbacks):
        return self.agent.run(
            goal, stop=callbacks.get("stop", self.stop_flag.is_set),
            on_progress=callbacks.get("on_progress", self.on_progress),
            on_ask=callbacks.get("on_ask", self.on_ask), deadline_s=deadline_s,
            on_text=callbacks.get("on_text", self.texts.append))

    def facts(self) -> list[dict]:
        return [json.loads(line) for line in self.record.read_text().splitlines()] if self.record.exists() else []

    def fact(self, key: str) -> list:
        return [f[key] for f in self.facts() if key in f]

    def eventual_fact(self, key: str, seconds: float = 3.0) -> list:
        """`fact(key)` once the fake has written it: a notification handler runs as its own task in the fake, so its
        record can land after `run()` has already returned (#139)."""
        end = time.monotonic() + seconds
        while not (found := self.fact(key)) and time.monotonic() < end:
            time.sleep(0.02)
        return found

    def pids(self) -> list[int]:
        return [f["start"]["pid"] for f in self.facts() if "start" in f]

    def wire(self) -> list[str]:
        return self.fact("wire")


@pytest.fixture
def harness(tmp_path):
    made = []

    def make(scenario=None, **options):
        h = Harness(tmp_path, scenario, **options)
        made.append(h)
        return h

    yield make
    for h in made:
        h.agent.close()


# --- a task


def test_the_answer_is_the_text_after_the_last_tool_call_and_progress_and_text_are_reported(harness):
    h = harness()
    assert h.run() == "The title is Python."  # not "I'll open the page. The title is Python."
    assert h.progress == [("mcp__chrome__navigate_page", {"url": "https://x.test"})]
    assert h.texts == ["I'll open the page.", "The title is Python."]  # a message per tool call, then the final one
    prompt = h.fact("prompt")[0]
    assert GOAL in prompt and "Bing or DuckDuckGo" in prompt
    assert f"Today is {datetime.now().astimezone():%A}" in prompt  # the agent resolves "this month" from it


def test_the_task_prompt_carries_the_date_and_the_goal():
    prompt = browse.task_prompt("find the news", datetime(2026, 11, 20, 15, 0))
    assert "Today is Friday, November 20, 2026." in prompt and prompt.endswith("Task: find the news")
    # the answer starts with a line vaice speaks (`browse_task.spoken_summary`), and the details follow it
    assert 'begins with "Summary:"' in prompt and "read aloud" in prompt and "details after that line" in prompt
    assert "Bing or DuckDuckGo" in prompt and "never Google" in prompt


def test_the_client_offers_no_file_system_or_terminal_and_new_session_carries_the_cwd_and_mcp_servers(harness):
    from acp.schema import McpServerStdio

    chrome = McpServerStdio(name="chrome", command="npx", args=["-y", browse.CHROME_DEVTOOLS_MCP], env=[])
    h = harness(mcp_servers=[chrome])
    h.run()
    init = h.fact("initialize")[0]
    assert init["client"] == "vaice"
    assert init["caps"]["fs"] == {"readTextFile": False, "writeTextFile": False} and init["caps"]["terminal"] is False
    new = h.fact("new_session")[0]
    assert Path(new["cwd"]).resolve() == h.cwd.resolve()
    assert new["mcp_servers"] == [{"name": "chrome", "command": "npx", "args": ["-y", browse.CHROME_DEVTOOLS_MCP],
                                   "env": []}]
    assert stat.S_IMODE(h.cwd.stat().st_mode) == 0o700


def test_the_agent_process_is_isolated_in_its_own_session_and_logs_to_a_private_file(harness):
    h = harness()
    h.run()
    start = h.fact("start")[0]
    assert start["env"]["PATH"] == os.environ["PATH"] and "FAKE_ACP_LOG" not in start["env"]
    assert os.getpgid(start["pid"]) == start["pid"]  # a session and a group of its own
    assert stat.S_IMODE(h.stderr_log.stat().st_mode) == 0o600
    assert "fake acp agent stderr line" in h.stderr_log.read_text()


def test_one_process_serves_every_task_with_a_fresh_session_each(harness):
    h = harness()
    for _ in range(2):  # each on a fresh thread, like fastrtc's workers
        assert run_on_thread(h.run, "browse-task") == "The title is Python."
    assert len(h.pids()) == 1 and len(h.fact("new_session")) == 2 and len(h.fact("prompt")) == 2
    assert len(h.fact("initialize")) == 1


def test_a_failing_callback_does_not_end_the_task(harness):
    h = harness()

    def boom(*args):
        raise RuntimeError("callback failed")

    assert h.run(on_progress=boom, on_text=boom) == "The title is Python."


@pytest.mark.parametrize(("scenario", "spoken"), [
    ({"steps": [{"text": "partial"}], "stop_reason": "refusal"}, browse.FAILED),
    ({"steps": [{"tool": "mcp__chrome__list_pages"}], "stop_reason": "end_turn"}, browse.NO_ANSWER),
    ({"steps": [], "stop_reason": "end_turn"}, browse.NO_ANSWER)])
def test_a_task_without_an_answer_is_an_error_with_a_sentence_to_say(harness, scenario, spoken):
    h = harness(scenario)
    with pytest.raises(BrowseError) as caught:
        h.run()
    assert caught.value.spoken == spoken


# --- the questions


ASK = {"steps": [{"tool": "mcp__chrome__click", "args": {"uid": "1_7"}},
                 {"permission": {"title": "mcp__chrome__click", "message": "{\"uid\":\"1_7\"}"}},
                 {"text": "Clicked it."}]}


def test_a_yes_selects_the_allow_option_and_the_question_carries_the_agents_title_and_message(harness):
    h = harness(ASK, pi_dialogs=True)
    assert h.run() == "Clicked it."
    assert h.asked == [("mcp__chrome__click", "{\"uid\":\"1_7\"}")]
    assert h.fact("permission") == [{"outcome": "selected", "option": "allow", "message": "{\"uid\":\"1_7\"}"}]


def test_a_question_without_a_message_carries_the_raw_arguments(harness):
    h = harness({"steps": [{"permission": {"title": "x", "raw": {"method": "confirm", "title": "x", "n": 1}}},
                           {"text": "ok"}]})
    h.run()
    assert json.loads(h.asked[0][1]) == {"method": "confirm", "title": "x", "n": 1}


def test_a_long_question_reaches_the_panel_whole(harness):
    script = "() => " + "x" * 5000 + "; submitTheForm()"  # what is approved must be all of what runs
    message = json.dumps({"function": script})
    h = harness({"steps": [{"permission": {"title": "mcp__chrome__evaluate_script", "message": message}},
                           {"permission": {"title": "y", "raw": {"function": script}}},
                           {"text": "ok"}]}, pi_dialogs=True)
    h.run()
    assert h.asked[0][1] == message
    assert json.loads(h.asked[1][1]) == {"function": script}


def test_a_question_carries_a_raw_input_that_is_not_an_object_and_the_requests_content_and_locations(harness):
    h = harness({"steps": [{"permission": {"title": "run", "raw": "rm -rf ~/Documents"}},
                           {"permission": {"title": "edit", "raw": ["a", "b"], "content": ["delete the file"],
                                           "locations": ["/tmp/notes.txt"]}},
                           {"permission": {"title": "nothing", "raw": None}},
                           {"text": "ok"}]}, pi_dialogs=True)
    h.run()
    assert json.loads(h.asked[0][1]) == "rm -rf ~/Documents"
    assert json.loads(h.asked[1][1]) == {"rawInput": ["a", "b"],
                                         "content": [{"type": "content",
                                                      "content": {"type": "text", "text": "delete the file"}}],
                                         "locations": [{"path": "/tmp/notes.txt"}]}
    assert json.loads(h.asked[2][1]) is None


@pytest.mark.parametrize("pi_dialogs", [False, True])
def test_a_message_argument_does_not_hide_the_other_arguments(harness, pi_dialogs):
    # another agent's tool can take a `message` next to what matters (a recipient); so can a Pi-shaped request with
    # more fields than Pi's dialog has, and a pi-acp request only counts as a dialog when the agent is pi-acp
    args = {"method": "confirm", "title": "send", "message": "hi", "to": "someone@example.test"}
    plain = {"method": "confirm", "title": "send", "message": "hi"}
    h = harness({"steps": [{"permission": {"title": "send", "raw": args}},
                           {"permission": {"title": "send", "raw": plain}}, {"text": "ok"}]}, pi_dialogs=pi_dialogs)
    h.run()
    assert json.loads(h.asked[0][1]) == args
    assert h.asked[1][1] == ("hi" if pi_dialogs else json.dumps(plain))


def test_a_no_answers_with_the_reject_option_then_cancels_and_ends_as_declined(harness):
    h = harness(ASK)
    h.answer = False
    with pytest.raises(BrowseDeclined):
        h.run()
    assert h.fact("permission")[0]["option"] == "reject" and h.fact("permission")[0]["outcome"] == "selected"
    assert h.wire() == ["permission_response", "cancel"]  # the answer goes out before the cancel
    assert h.eventual_fact("cancel") == [1]  # the script ends on its own, so the prompt can end before this lands
    assert h.pids() and alive(h.pids()[0])  # the agent is kept: it settled the cancel


def test_a_failing_question_callback_is_a_no(harness):
    h = harness(ASK)

    def broken(title, message, stop):
        raise RuntimeError("callback failed")

    with pytest.raises(BrowseDeclined):
        h.run(on_ask=broken)


def test_a_question_that_waited_too_long_is_a_no_even_when_answered_yes_late(tmp_path):
    h = Harness(tmp_path, ASK, ask_timeout_s=0.3)
    try:
        def late_yes(title, message, stop):
            while not stop():  # honors its stop, which turns true when the question has waited ask_timeout_s
                time.sleep(0.02)
            return True

        with pytest.raises(BrowseDeclined):
            h.run(on_ask=late_yes)
        assert h.fact("permission")[0]["option"] == "reject"
    finally:
        h.agent.close()


def test_the_allow_mode_answers_allow_once_without_asking(harness):
    h = harness(ASK, permissions="allow")
    assert h.run() == "Clicked it."
    assert h.asked == [] and h.fact("permission")[0]["option"] == "allow"


def test_a_request_with_only_always_options_is_allowed_with_the_allow_always_option(harness):
    h = harness({"steps": [{"permission": {"title": "x", "message": "m", "kinds": ["allow_always", "reject_always"]}},
                           {"text": "ok"}]})
    h.run()
    assert h.fact("permission")[0]["option"] == "allow"


def test_a_yes_to_a_request_with_no_allow_option_is_answered_cancelled(harness):
    h = harness({"steps": [{"permission": {"title": "x", "message": "m", "kinds": ["reject_once"]}},
                           {"text": "ok"}]})
    with pytest.raises(BrowseError):  # the fake ends its turn on a cancelled question (an agent may)
        h.run()
    assert h.fact("permission")[0]["outcome"] == "cancelled"


# --- stop, the deadline, a hung or dead agent


def test_stop_during_a_pending_question_answers_it_cancelled_then_cancels_the_session(harness):
    h = harness(ASK)
    asked, release = threading.Event(), threading.Event()

    def waiting(title, message, stop):
        asked.set()
        while not stop() and not release.is_set():
            time.sleep(0.02)
        return True

    task = start_thread(lambda: h.run(on_ask=waiting), "browse-task")
    assert asked.wait(5)
    h.stop_flag.set()
    assert isinstance(task.join(), BrowseStopped)
    assert h.fact("permission") == [{"outcome": "cancelled", "option": None, "message": "{\"uid\":\"1_7\"}"}]
    assert h.wire() == ["permission_response", "cancel"]
    assert alive(h.pids()[0])  # it settled the cancel quickly: kept for the next task


def test_the_deadline_cancels_the_session_and_is_an_error_naming_the_seconds(tmp_path):
    h = Harness(tmp_path, scenarios={"slow": {"steps": [{"text": "working"}, {"sleep": 30}]}, "ok": SIMPLE})
    try:
        h.run("scenario=ok")  # the agent is up, so the deadline falls inside the prompt, not its start-up
        with pytest.raises(BrowseError, match="seconds passed") as caught:
            h.run("scenario=slow", deadline_s=0.4)
        assert "stopped the browser task after" in caught.value.spoken
        assert h.fact("cancel") == [1]
    finally:
        h.agent.close()


def test_stop_before_the_task_starts_never_starts_the_agent(harness):
    h = harness()
    h.stop_flag.set()
    with pytest.raises(BrowseStopped):
        h.run()
    assert h.pids() == []


def test_a_task_waiting_for_the_one_before_it_can_still_be_stopped(harness):
    h = harness({"steps": [{"sleep": 30}]})
    first = start_thread(lambda: h.run(stop=h.stop_flag.is_set), "browse-first")
    while not h.fact("prompt"):
        time.sleep(0.02)
    stop_second = threading.Event()
    second = start_thread(lambda: h.run(stop=stop_second.is_set), "browse-second")
    time.sleep(0.2)
    stop_second.set()
    assert isinstance(second.join(), BrowseStopped)
    h.stop_flag.set()
    assert isinstance(first.join(), BrowseStopped)


def test_an_agent_that_ignores_the_cancel_is_ended_and_the_next_task_starts_a_new_one(tmp_path):
    h = Harness(tmp_path, {"steps": [{"text": "working"}, {"sleep": 30}]}, env={"FAKE_ACP_IGNORE_CANCEL": "1"})
    try:
        started = time.monotonic()
        task = start_thread(lambda: h.run(), "browse-task")
        while not h.fact("prompt"):
            time.sleep(0.02)
        h.stop_flag.set()
        assert isinstance(task.join(), BrowseStopped)
        assert time.monotonic() - started < browse.ABORT_WAIT + browse.TERM_WAIT + 5
        assert wait_dead(h.pids()[0])
        h.stop_flag.clear()
        h.agent._env["FAKE_ACP_SCENARIO"] = json.dumps(SIMPLE)  # the next agent plays the simple scenario
        assert h.run() == "The title is Python."
        assert len(h.pids()) == 2 and alive(h.pids()[1])
    finally:
        h.agent.close()


def test_an_agent_that_exits_mid_task_is_a_browse_error_and_the_next_task_restarts_it(tmp_path):
    h = Harness(tmp_path, scenarios={"die": {"steps": [{"text": "hi"}, {"exit": 3}]}, "ok": SIMPLE})
    try:
        with pytest.raises(BrowseError, match="exited") as caught:
            h.run("scenario=die")
        assert caught.value.spoken == browse.FAILED
        assert h.run("scenario=ok") == "The title is Python."
        assert len(h.pids()) == 2 and wait_dead(h.pids()[0])
    finally:
        h.agent.close()


def test_a_command_that_cannot_be_started_is_a_clean_error(tmp_path):
    agent = AcpAgent([str(tmp_path / "missing")], env={}, cwd=tmp_path / "work", log_path=tmp_path / "log")
    try:
        with pytest.raises(BrowseError, match="could not start") as caught:
            agent.run(GOAL, stop=lambda: False, on_progress=lambda *a: None, on_ask=lambda *a: True, deadline_s=5)
        assert caught.value.spoken == browse.CANT_START
    finally:
        agent.close()


def test_an_agent_that_does_not_answer_initialize_is_ended_and_is_an_error(tmp_path, monkeypatch):
    # long enough for the fake to start and record its pid on a slow CI runner (~0.3 s locally, 2x on CI), far
    # shorter than its 30 s delay
    monkeypatch.setattr(browse, "START_TIMEOUT", 3.0)
    h = Harness(tmp_path, env={"FAKE_ACP_INIT_DELAY": "30"})
    try:
        with pytest.raises(BrowseError, match="did not answer") as caught:
            h.run()
        assert caught.value.spoken == browse.CANT_START
        assert wait_dead(h.pids()[0])
    finally:
        h.agent.close()


def test_a_json_rpc_error_from_the_agent_is_an_error_without_secrets(harness, caplog):
    h = harness({"steps": [{"fail": f"upstream said {KEY}"}]}, secrets=[KEY])
    with caplog.at_level(logging.DEBUG), pytest.raises(BrowseError) as caught:
        h.run()
    assert "***" in str(caught.value)  # the agent's own words are kept, scrubbed
    assert KEY not in str(caught.value) and KEY not in caplog.text


def test_close_during_a_task_fails_it_and_ends_the_agent(harness):
    h = harness({"steps": [{"sleep": 30}]})
    task = start_thread(h.run, "browse-task")
    while not h.fact("prompt"):
        time.sleep(0.02)
    h.agent.close()
    assert isinstance(task.join(), BrowseError)
    assert wait_dead(h.pids()[0])


def test_close_is_idempotent_and_a_closed_agent_runs_nothing(harness):
    h = harness()
    h.run()
    h.agent.close()
    h.agent.close()
    with pytest.raises(BrowseError, match="closed"):
        h.run()


# --- the gate's tripwire


def with_marker(*steps, **extra):
    return {"steps": [{"text": MARKER}, *steps], **extra}


def test_the_marker_is_never_part_of_the_answer_or_the_text_reported_and_start_up_text_is_dropped(harness):
    h = harness({"steps": [{"text": "pi-acp 0.0.34 starting. "}, {"text": MARKER}, {"tool": "mcp__chrome__list_pages"},
                           {"text": "Done."}]}, marker=MARKER)
    assert h.run() == "Done."
    assert h.texts == ["Done."]
    assert h.progress == [("mcp__chrome__list_pages", {})]


def test_the_marker_may_arrive_with_surrounding_whitespace(harness):
    h = harness({"steps": [{"text": f" {MARKER}\n"}, {"text": "Done."}]}, marker=MARKER)
    assert h.run() == "Done."


def test_a_tool_call_before_the_marker_ends_the_agent_and_fails_the_task(harness):
    h = harness({"steps": [{"text": "hi"}, {"tool": "mcp__chrome__click", "args": {"uid": "1_7"}}, {"sleep": 30}]},
                marker=MARKER)
    with pytest.raises(BrowseError, match="safety gate") as caught:
        h.run()
    assert caught.value.spoken == browse.FAILED
    assert h.progress == []  # the call was not reported as a step the user could rely on
    assert wait_dead(h.pids()[0])


def test_a_permission_request_before_the_marker_is_answered_cancelled_and_ends_the_agent(harness):
    h = harness({"steps": [{"permission": {"title": "mcp__chrome__click", "message": "m"}}, {"sleep": 30}]},
                marker=MARKER)
    with pytest.raises(BrowseError, match="safety gate"):
        h.run()
    assert h.asked == []
    assert h.fact("permission")[0]["outcome"] == "cancelled"
    assert wait_dead(h.pids()[0])


def test_after_a_tripwire_the_next_task_gets_a_new_agent(tmp_path):
    h = Harness(tmp_path, scenarios={"bad": {"steps": [{"tool": "t"}, {"sleep": 30}]},
                                     "ok": with_marker({"text": "Done."})}, marker=MARKER)
    try:
        with pytest.raises(BrowseError):
            h.run("scenario=bad")
        assert h.run("scenario=ok") == "Done."
        assert len(h.pids()) == 2
    finally:
        h.agent.close()


def test_without_a_marker_nothing_is_demanded(harness):
    h = harness({"steps": [{"tool": "t"}, {"text": "Done."}]})
    assert h.run() == "Done."


# --- session/close


def test_session_close_is_sent_only_when_the_agent_advertises_it(tmp_path):
    advertising = Harness(tmp_path / "a", env={"FAKE_ACP_CLOSE_SESSION": "1"})
    plain = Harness(tmp_path / "b")
    try:
        advertising.run()
        plain.run()
        assert advertising.fact("close_session") == ["s1"]
        assert plain.fact("close_session") == []
    finally:
        advertising.agent.close()
        plain.agent.close()


# --- nothing secret or private is logged


def test_secrets_the_goal_and_the_pages_text_never_reach_a_log_an_error_or_a_repr(harness, caplog):
    goal = "find the secret recipe at private.example.test"
    h = harness({"steps": [{"text": "thinking about the recipe"},
                           {"tool": f"Run curl -H 'Authorization: {KEY}'", "args": {}},  # a free-text title
                           {"tool": "mcp__chrome__fill", "args": {"value": "typed-by-the-agent"}},
                           {"permission": {"title": "mcp__chrome__fill",
                                           "message": "{\"value\":\"typed-by-the-agent\"}"}},
                           {"text": "Done."}]}, secrets=[KEY], env={"VAICE_PI_API_KEY": KEY})
    outcomes = []
    with caplog.at_level(logging.DEBUG):
        outcomes.append(h.run(goal))
        h.answer = False
        for _ in range(2):
            try:
                h.run(goal)
            except BrowseError as exc:
                outcomes.append(exc)
        h.agent.close()
    seen = caplog.text + repr(h.agent) + " ".join(map(str, outcomes))
    for private in (KEY, goal, "private.example.test", "typed-by-the-agent", "thinking about the recipe"):
        assert private not in seen, private


# --- the browser profile


def test_the_agent_sweeps_the_profile_before_every_new_process_and_after_close(tmp_path, monkeypatch):
    swept = []
    monkeypatch.setattr(browse, "end_profile_processes", lambda profile: swept.append(profile) or [])
    profile = tmp_path / "chrome-profile"
    h = Harness(tmp_path, scenarios={"die": {"steps": [{"text": "hi"}, {"exit": 3}]}, "ok": SIMPLE},
                chrome_profile=profile)
    try:
        h.run("scenario=ok")
        assert swept == [profile]  # before the first process
        with pytest.raises(BrowseError):
            h.run("scenario=die")
        h.run("scenario=ok")
        assert swept == [profile, profile]  # and again before the replacement
    finally:
        h.agent.close()
    assert swept == [profile, profile, profile]  # and once more after close()


def test_without_a_profile_nothing_is_swept(tmp_path, monkeypatch):
    monkeypatch.setattr(browse, "end_profile_processes", lambda profile: pytest.fail("swept"))
    h = Harness(tmp_path)
    try:
        h.run()
    finally:
        h.agent.close()


def test_an_agent_is_a_browse_agent(tmp_path):
    agent: browse.BrowseAgent = AcpAgent(["x"], env={}, cwd=tmp_path, log_path=tmp_path / "log")
    assert callable(agent.run) and callable(agent.close)
    agent.close()
