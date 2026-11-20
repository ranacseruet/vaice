"""The live browser-agent check's own logic, over a fake agent: no browser, no model, no network."""
import json
import subprocess

import browse_live
import pytest

from vaice import app, browse, config

SECRET = "sk-fake-browse-key-123456"


class FakeAgent:
    """A `BrowseAgent` that answers (or raises) and records how it was used."""

    def __init__(self, outcomes=("An answer about coffee makers.",), profile="/tmp/fake-profile"):
        self.outcomes = list(outcomes)
        self.goals: list[str] = []
        self.runs: list[dict] = []
        self.closed = 0
        self._chrome_profile = profile

    def run(self, goal, *, stop, on_progress, on_ask, deadline_s, on_text=None):
        self.goals.append(goal)
        self.runs.append({"deadline_s": deadline_s, "stop": stop})
        on_progress("navigate_page", {"url": "https://duckduckgo.com/?q=coffee"})
        on_progress("take_snapshot", {})
        assert on_ask("click link", "Click it?", stop) is True
        outcome = self.outcomes.pop(0) if len(self.outcomes) > 1 else self.outcomes[0]
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    def close(self):
        self.closed += 1


@pytest.fixture
def harness(monkeypatch, tmp_path):
    """Patches what `main` reaches for: the app's opener, the profile holders and the sleeps."""
    state = {"agent": FakeAgent(), "holders": []}
    monkeypatch.setattr(app, "_open_browse", lambda: state["agent"])
    monkeypatch.setattr(browse, "_profile_pids", lambda profile, listing: state["holders"])
    monkeypatch.setattr(browse_live, "ps_listing", lambda: "")
    monkeypatch.setattr(browse_live, "LEFTOVER_WAIT", 0)
    monkeypatch.setattr(browse, "pi_version", lambda command="pi": (1, 5, 2))
    monkeypatch.setattr(app, "default_browser_args", app.default_browser_args)  # restored after prepare_browser
    monkeypatch.setattr(config, "BROWSE_HEADLESS", False)
    monkeypatch.setattr(config, "BROWSE", "pi")
    monkeypatch.setattr(config, "BROWSE_MODEL", "fake-model")
    monkeypatch.setattr(config, "BROWSE_API_KEY", SECRET)
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data")
    state["out"] = tmp_path / "out.json"
    return state


def run(harness, *argv):
    return browse_live.main([*argv, "--out", str(harness["out"])])


def test_the_defaults_and_goal_replaces_them():
    args = browse_live.parse_args([])
    assert (args.goal, args.repeats, args.seconds, args.visible, args.label) == (None, 1, None, False, "")
    assert len(browse_live.DEFAULT_GOALS) == 3
    assert browse_live.parse_args(["--goal", "a", "--goal", "b"]).goal == ["a", "b"]
    for bad in (["--repeats", "0"], ["--seconds", "0"]):
        with pytest.raises(SystemExit):
            browse_live.parse_args(bad)


def test_the_default_goals_run_and_goal_replaces_them(harness):
    assert run(harness) == 0
    assert harness["agent"].goals == browse_live.DEFAULT_GOALS
    harness["agent"] = FakeAgent()
    assert run(harness, "--goal", "only this", "--repeats", "2", "--seconds", "30") == 0
    assert harness["agent"].goals == ["only this", "only this"]
    assert {r["deadline_s"] for r in harness["agent"].runs} == {30.0}


def test_the_deadline_defaults_to_the_configured_seconds(harness, monkeypatch):
    monkeypatch.setattr(config, "BROWSE_SECONDS", 42.0)
    run(harness, "--goal", "g")
    assert harness["agent"].runs[0]["deadline_s"] == 42.0


@pytest.mark.parametrize(("visible", "headless"), [(False, True), (True, False)])
def test_chrome_is_headless_unless_visible_and_always_muted(harness, monkeypatch, visible, headless):
    seen = {}

    def opener():
        seen["headless"] = config.BROWSE_HEADLESS
        seen["args"] = app._browse_base()[2]
        return harness["agent"]

    monkeypatch.setattr(app, "_open_browse", opener)
    run(harness, "--goal", "g", *(["--visible"] if visible else []))
    assert seen["headless"] is headless
    assert ("--headless" in seen["args"]) is headless
    assert all(arg in seen["args"] for arg in browse_live.MUTE_ARGS)
    assert any(arg.startswith("--userDataDir=") for arg in seen["args"])


def test_an_answer_is_recorded_and_exits_zero(harness, capsys):
    assert run(harness, "--goal", "coffee makers", "--label", "run A") == 0
    document = json.loads(harness["out"].read_text())
    (record,) = document["runs"]
    assert record["goal"] == "coffee makers" and record["repeat"] == 1
    assert record["outcome"] == "answer" and record["answer"] == "An answer about coffee makers."
    assert record["error"] is None and record["seconds"] >= 0
    assert record["tools"] == ["navigate_page", "take_snapshot"]
    assert record["asks"] == ["click link"]
    assert record["urls"] == ["https://duckduckgo.com/?q=coffee"]
    assert document["label"] == "run A" and document["leftover_processes"] == 0
    assert document["config"]["browse"] == "pi" and document["config"]["model"] == "fake-model"
    assert document["config"]["pi_version"] == "1.5.2" and document["started"]
    assert harness["agent"].closed == 1
    out = capsys.readouterr().out
    assert "answers: 1/1" in out and "An answer about coffee makers." in out


def test_a_browse_error_is_recorded_with_its_spoken_sentence_and_exits_one(harness):
    harness["agent"] = FakeAgent([browse.BrowseError("the task's 5 seconds passed", "I stopped it.")])
    assert run(harness, "--goal", "g") == 1
    (record,) = json.loads(harness["out"].read_text())["runs"]
    assert record["outcome"] == "BrowseError" and record["answer"] is None
    assert record["error"] == "the task's 5 seconds passed" and record["spoken"] == "I stopped it."
    assert harness["agent"].closed == 1


def test_leftover_processes_exit_one(harness):
    harness["holders"] = [101, 102]
    assert run(harness, "--goal", "g") == 1
    assert json.loads(harness["out"].read_text())["leftover_processes"] == 2


def test_an_unreadable_process_table_is_a_failure_not_a_clean_exit(harness, monkeypatch):
    monkeypatch.setattr(browse_live, "ps_listing", lambda: None)
    assert run(harness, "--goal", "g") == 1
    assert json.loads(harness["out"].read_text())["leftover_processes"] is None


def test_ps_listing_is_none_when_ps_fails(monkeypatch):
    def broken(*args, **kwargs):
        raise subprocess.TimeoutExpired("ps", 1)
    monkeypatch.setattr(subprocess, "run", broken)
    assert browse_live.ps_listing() is None


def test_no_agent_is_a_failure_that_points_at_the_startup_warnings(harness):
    harness["agent"] = None
    with pytest.raises(SystemExit) as exc:
        run(harness)
    assert "startup warnings above" in str(exc.value)
    assert not harness["out"].exists()


def test_the_agent_is_closed_even_when_run_raises_unexpectedly(harness):
    harness["agent"] = FakeAgent([RuntimeError("boom")])
    assert run(harness, "--goal", "g", "--goal", "h") == 1
    runs = json.loads(harness["out"].read_text())["runs"]
    assert [r["outcome"] for r in runs] == ["RuntimeError", "RuntimeError"]
    assert harness["agent"].closed == 1


def test_the_agent_is_closed_when_the_summary_step_fails(harness, monkeypatch):
    def boom(*_args):
        raise OSError("disk")

    monkeypatch.setattr(browse_live, "agent_info", boom)
    with pytest.raises(OSError):
        run(harness, "--goal", "g")
    assert harness["agent"].closed == 1


def test_ctrl_c_stops_the_task_and_still_reports(harness, monkeypatch):
    agent, events = harness["agent"], []
    real_event = browse_live.threading.Event

    def make_event():
        events.append(real_event())  # the flag main hands to the handler and to every task's `stop`
        return events[-1]

    def interrupted_run(goal, *, stop, **_kwargs):
        events[0].set()  # what the SIGINT handler does
        assert stop()
        raise browse.BrowseStopped

    monkeypatch.setattr(browse_live.threading, "Event", make_event)
    monkeypatch.setattr(agent, "run", interrupted_run)
    assert run(harness, "--goal", "a", "--goal", "b") == 1
    runs = json.loads(harness["out"].read_text())["runs"]
    assert [r["outcome"] for r in runs] == ["BrowseStopped"]  # the second goal never started
    assert agent.closed == 1


def test_no_api_key_value_reaches_the_output_or_the_results(harness, capsys):
    run(harness, "--goal", "g")
    captured = capsys.readouterr()
    assert SECRET not in captured.out + captured.err
    assert SECRET not in harness["out"].read_text()
