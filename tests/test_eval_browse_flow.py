"""The browser-flow eval's own logic: scoring, the fake agent, the outbox drainer, its data files and (against the fake
realtime server) one whole run. No model, no audio, no browser."""
import json
import threading
import time

import browse_flow
import numpy as np
import pytest
import run as eval_run
from fake_realtime import pcm

from vaice import browse_task
from vaice.browse import BrowseStopped
from vaice.outbox import Outbox
from vaice.skills import SkillResult

AUDIO = (16000, pcm(1, 2, 3).reshape(1, -1))
FIXTURE = {
    "id": "coffee",
    "answer": "I now have the research. Let me compile my findings into a final answer.\n\n---\n\n"
              "## Best coffee makers\n\n### 1. Ninja 12-Cup Programmable Coffee Brewer, $90\n- Braun Pure Flavor is "
              "the value pick from Good Housekeeping\n",
    "keywords": ["Ninja", "Braun"],
}
NINJA_LINE = "### 1. Ninja 12-Cup Programmable Coffee Brewer, $90"


def record(start="Okay, looking into it.", outcome="The browser found a Ninja.", follow="Ninja again.",
           start_calls=(("browse", "{}"),), follow_calls=(), runs=1, ui=()):
    return {
        "start": {"text": start, "calls": [list(c) for c in start_calls], "first_audio_s": 0.8},
        "outcome": None if outcome is None else {"text": outcome, "first_audio_s": 1.5},
        "followup": None if follow is None else {"text": follow, "calls": [list(c) for c in follow_calls],
                                                 "first_audio_s": 0.9},
        "agent_runs": runs, "ui": list(ui),
    }


# -- scoring


def test_a_clean_run_scores_well():
    score = browse_flow.score_run(record(), FIXTURE)
    assert score["start_browse"] and not score["start_invented"] and score["start_extra_chars"] == 22
    assert score["outcome_spoken"] and score["outcome_named"] and not score["outcome_narration"]
    assert not score["chat_report"]
    assert not score["followup_rebrowse"] and not score["followup_status"] and score["followup_named"]
    assert (score["start_first_audio_s"], score["outcome_first_audio_s"], score["followup_first_audio_s"]) == (
        0.8, 1.5, 0.9)


def test_a_long_start_reply_is_invented_even_without_a_name():
    text = "Here are a few top recommendations for you, the best machines for a home kitchen are very popular. " * 2
    assert len(text) > browse_flow.INVENTED_CHARS
    assert browse_flow.score_run(record(start=text), FIXTURE)["start_invented"]
    assert not browse_flow.is_invented("x" * browse_flow.INVENTED_CHARS, [])
    assert browse_flow.is_invented("x" * (browse_flow.INVENTED_CHARS + 1), [])


def test_a_start_reply_that_names_a_finding_is_invented():
    assert browse_flow.score_run(record(start="The Braun is great."), FIXTURE)["start_invented"]


@pytest.mark.parametrize("text", [
    "I'm compiling your findings into a final answer.", "Let me compile that.", "I now have everything.",
    "I have gathered the information.", "Here is the FINAL ANSWER.",
])
def test_the_agents_narration_is_recognised_in_an_outcome(text):
    assert browse_flow.score_run(record(outcome=text), FIXTURE)["outcome_narration"]
    assert not browse_flow.score_run(record(outcome=text), FIXTURE)["outcome_named"]


def test_naming_a_finding_ignores_case():
    assert browse_flow.names_finding("the NINJA won", ["Ninja"])
    assert not browse_flow.names_finding("a coffee maker", ["Ninja"])


def test_no_outcome_is_not_named_and_has_no_narration_verdict():
    score = browse_flow.score_run(record(outcome=None), FIXTURE)
    assert score["outcome_spoken"] is False and score["outcome_named"] is False
    assert score["outcome_narration"] is None and score["outcome_first_audio_s"] is None


def test_no_follow_up_leaves_its_flags_undefined():
    score = browse_flow.score_run(record(follow=None), FIXTURE)
    assert score["followup_rebrowse"] is None and score["followup_status"] is None and score["followup_named"] is None


def test_a_follow_up_that_calls_browse_or_starts_a_second_agent_run_is_a_rebrowse():
    assert browse_flow.score_run(record(follow_calls=[("browse", "{}")]), FIXTURE)["followup_rebrowse"]
    assert browse_flow.score_run(record(runs=2), FIXTURE)["followup_rebrowse"]
    assert not browse_flow.score_run(record(follow_calls=[("browser_status", "{}")]), FIXTURE)["followup_rebrowse"]
    assert browse_flow.score_run(record(follow_calls=[("browser_status", "{}")]), FIXTURE)["followup_status"]


def test_the_report_itself_in_the_chat_counts_but_a_summary_does_not():
    summary = [{"role": "assistant", "content": "Sure. The Ninja and the Braun were the picks."}]
    report = [{"role": "user", "content": "x"},
              {"role": "assistant", "content": f"The browser task finished:\n\n**{NINJA_LINE}**\n- more"}]
    assert not browse_flow.score_run(record(ui=summary), FIXTURE)["chat_report"]
    assert browse_flow.score_run(record(ui=report), FIXTURE)["chat_report"]
    # the user's own words never count
    assert not browse_flow.chat_has_report([{"role": "user", "content": NINJA_LINE}], FIXTURE)


def test_report_lines_are_the_keyword_lines_of_the_body_only():
    lines = browse_flow.report_lines(FIXTURE)
    assert lines == ["1. ninja 12-cup programmable coffee brewer, $90",
                     "- braun pure flavor is the value pick from good housekeeping"]


def test_tally_counts_k_of_n_where_defined_and_medians():
    scores = [browse_flow.score_run(record(), FIXTURE), browse_flow.score_run(record(start="Braun.", follow=None),
                                                                              FIXTURE)]
    tally = browse_flow.tally(scores)
    assert tally["runs"] == 2
    assert tally["flags"]["start_browse"] == [2, 2] and tally["flags"]["start_invented"] == [1, 2]
    assert tally["flags"]["followup_named"] == [1, 1]  # undefined for the run without a follow-up
    assert tally["medians"]["start_extra_chars"] == (22 + 6) / 2
    assert browse_flow.tally([])["medians"]["start_extra_chars"] is None


# -- the fake agent


def run_agent(agent, stop=lambda: False, **kwargs):
    seen = []
    answer = agent.run("a goal", stop=stop, on_progress=lambda tool, args: seen.append(tool),
                       on_ask=lambda *_: True, deadline_s=30, **kwargs)
    return answer, seen


def test_the_fake_agent_reports_tools_returns_the_answer_and_counts_runs():
    agent = browse_flow.FakeBrowseAgent("the answer", seconds=0.2)
    answer, seen = run_agent(agent)
    assert answer == "the answer" and seen == [tool for tool, _ in browse_flow.STEPS]
    assert agent.finished_at is not None
    run_agent(agent)
    assert agent.runs == 2 and agent.goals == ["a goal", "a goal"]


def test_the_fake_agent_honours_stop():
    agent = browse_flow.FakeBrowseAgent("the answer", seconds=30)
    stop = threading.Event()
    threading.Timer(0.1, stop.set).start()
    started = time.monotonic()
    with pytest.raises(BrowseStopped):
        run_agent(agent, stop.is_set)
    assert time.monotonic() - started < 5 and agent.stopped == 1 and agent.finished_at is None


def test_a_held_fake_task_runs_until_released_and_stop_still_ends_it():
    release = threading.Event()
    agent = browse_flow.FakeBrowseAgent("the answer", seconds=0.05, release=release)
    result = []
    thread = threading.Thread(target=lambda: result.append(run_agent(agent)[0]))
    thread.start()
    time.sleep(0.4)
    assert thread.is_alive() and agent.finished_at is None  # past its tools, still held
    release.set()
    thread.join(5)
    assert result == ["the answer"]

    held = browse_flow.FakeBrowseAgent("x", seconds=0.05, release=threading.Event())
    with pytest.raises(BrowseStopped):
        run_agent(held, lambda: True)


# -- the outbox drainer


class Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


def frame(seconds):
    return 24000, np.zeros(int(24000 * seconds), dtype=np.float32)


def test_the_drainer_serves_frames_and_markers_in_order_paced_by_the_audio_served():
    clock, said = Clock(), []
    outbox = Outbox(clock=clock)
    drainer = browse_flow.Drainer(outbox, clock=clock)
    drainer.step()  # the first poll of an outbox counts as stale and clears: nothing is queued yet
    outbox.put(frame(1.0))
    outbox.put_marker(lambda: said.append("first"))
    outbox.put(frame(1.0))
    outbox.put_marker(lambda: said.append("second"))

    drainer.step()
    assert len(drainer.frames) == 1 and said == []  # one frame; the audio served ends 1 s ahead
    clock.now += 0.3
    drainer.step()
    assert len(drainer.frames) == 1 and said == []  # still more than the lead ahead
    clock.now += 0.4  # 0.7 s in: the first second is nearly played
    drainer.step()
    assert said == ["first"] and len(drainer.frames) == 2  # the marker ran, then the next frame was served
    clock.now += 1.0
    drainer.step()
    assert said == ["first", "second"] and len(drainer.frames) == 2 and len(outbox) == 0
    assert [seconds for _, seconds in drainer.frames] == [1.0, 1.0]


def test_the_drainer_serves_nothing_while_a_reply_is_in_flight_but_keeps_the_outbox_fresh():
    clock = Clock()
    outbox = Outbox(clock=clock)
    drainer = browse_flow.Drainer(outbox, clock=clock)
    drainer.step()
    outbox.put(frame(0.5))
    drainer.replying.set()
    clock.now += 20  # not stale: every step touches the outbox
    drainer.step()
    clock.now += 20
    drainer.step()
    assert drainer.frames == [] and len(outbox) == 1
    drainer.replying.clear()
    drainer.step()
    assert len(drainer.frames) == 1


def test_reply_audio_counts_as_playing_and_speech_start_clears_the_outbox():
    clock = Clock()
    outbox = Outbox(clock=clock)
    drainer = browse_flow.Drainer(outbox, clock=clock)
    drainer.step()
    drainer.note_playing(2.0)
    outbox.put(frame(0.5))
    drainer.step()
    assert drainer.frames == [] and drainer.playing_for() == 2.0  # an outcome waits behind the reply's own audio
    dropped = []
    outbox.put_marker(lambda: None, on_drop=lambda: dropped.append(1))
    drainer.speech_start()
    assert len(outbox) == 0 and dropped == [1] and drainer.playing_for() == 0.0


def test_the_drainer_thread_serves_and_stops_cleanly():
    outbox = Outbox()
    drainer = browse_flow.Drainer(outbox)
    said = threading.Event()
    drainer.start()
    time.sleep(0.1)  # it has polled, so what is queued next is not stale
    outbox.put(frame(0.1))
    outbox.put_marker(said.set)
    assert said.wait(5)
    drainer.stop()
    assert not drainer.is_alive() and len(drainer.frames) == 1
    browse_flow.Drainer(Outbox()).stop()  # never started: nothing to end


def test_the_drainer_is_paced_by_the_apps_own_lead():
    from vaice import interruptible

    assert browse_flow.LEAD_SECONDS == interruptible.LEAD_SECONDS
    assert browse_flow.IDLE_WAIT_SECONDS == interruptible.IDLE_WAIT_SECONDS


# -- the data files


def test_every_case_names_a_fixture_and_a_valid_follow_up():
    cases, fixtures = browse_flow.load_cases()
    ids = [case["id"] for case in cases]
    assert len(ids) == len(set(ids)) and len(cases) >= 8
    for case in cases:
        assert set(case) == {"id", "say", "fixture", "followup"}, case["id"]
        assert case["fixture"] in fixtures and case["say"].strip(), case["id"]
        assert set(case["followup"]) == {"say", "when"} and case["followup"]["say"].strip()
        assert case["followup"]["when"] in ("running", "finished")
    assert {case["followup"]["when"] for case in cases} == {"running", "finished"}
    assert any(case["followup"]["say"] == "Have you completed compiling into the final answer?" for case in cases)
    assert {case["fixture"] for case in cases} == set(fixtures)  # every recorded answer is used


def test_every_fixture_is_a_recorded_answer_with_keywords_from_its_report():
    _, fixtures = browse_flow.load_cases()
    cases, _ = browse_flow.load_cases()
    asked = " ".join(case["say"] + " " + case["followup"]["say"] for case in cases).casefold()
    assert len(fixtures) == 6
    for fixture in fixtures.values():
        assert set(fixture) == {"id", "goal", "answer", "keywords"}
        head, separator, body = fixture["answer"].partition("---")
        assert separator and browse_flow.is_narration(head), fixture["id"]  # starts with the agent's self-narration
        assert 2 <= len(fixture["keywords"]) <= 4
        for keyword in fixture["keywords"]:
            assert keyword.casefold() in body.casefold(), (fixture["id"], keyword)
            assert keyword.casefold() not in head.casefold() and keyword.casefold() not in asked
        assert browse_flow.report_lines(fixture), fixture["id"]


# -- the options


def test_the_defaults():
    args = browse_flow.parse_args([])
    assert (args.repeats, args.task_seconds, args.vision, args.cases, args.prompt, args.label, args.out) == (
        3, 8.0, False, None, None, "", None)
    assert args.url == eval_run.config.LLAMACPP_URL


def test_the_options():
    args = browse_flow.parse_args(["--repeats", "1", "--cases", "a,b", "--task-seconds", "2.5", "--vision",
                                   "--prompt", "p", "--label", "L", "--out", "x.json", "--url", "http://h:1"])
    assert (args.repeats, args.cases, args.task_seconds, args.vision, args.prompt, args.label, str(args.out),
            args.url) == (1, "a,b", 2.5, True, "p", "L", "x.json", "http://h:1")


@pytest.mark.parametrize("bad", [["--repeats", "0"], ["--task-seconds", "0"], ["--task-seconds", "-1"]])
def test_bad_options_are_refused(bad):
    with pytest.raises(SystemExit):
        browse_flow.parse_args(bad)


# -- the stub skills with the assistant's own browser control


def test_stub_skills_take_the_callers_browser_control_in_place_of_the_stubs():
    control = browse_task.browse_skills(busy=None, status=lambda: SkillResult("a real status"),
                                        stop=lambda: SkillResult(browse_task.NO_TASK))
    stubs = eval_run.stub_skills(browse=True, browse_control=control)
    assert stubs.run("browser_status", "{}").speech == "a real status"
    assert stubs.run("stop_browsing", "{}").speech == browse_task.NO_TASK
    assert eval_run.stub_skills(browse=True).run("browser_status", "{}").speech == eval_run.BROWSER_STATUS_STUB
    assert "browse" not in eval_run.stub_skills(browse_control=control).names  # without `browse` nothing is offered


# -- a whole run against the fake realtime server


def a_reply(text):
    """A reply session's script: the words, 0.1 s of audio, done."""
    return [("text", text), ("audio", np.ones(2400, dtype="<i2")), ("done",)]


def a_call(name, arguments="{}"):
    """A reply session's script that calls one skill."""
    return [("call", "item-1", name, "call-1", arguments), ("done",)]


@pytest.fixture
def quick(monkeypatch):
    monkeypatch.setattr(browse_flow, "SETTLE_SECONDS", 0)

    def boom(*_args, **_kwargs):
        raise AssertionError("a stub skill ran something on this machine")

    monkeypatch.setattr("subprocess.run", boom)


CASE = {"id": "coffee", "say": "Research coffee makers.", "fixture": "coffee",
        "followup": {"say": "What did you find?", "when": "finished"}}


def test_a_whole_finished_run_records_the_start_the_outcome_the_chat_and_the_follow_up(realtime, quick):
    server = realtime(reply=[
        a_call("browse", '{"goal": "research coffee makers"}'),  # the request: a call ...
        a_reply("I'll look into it."),                          # ... and its words after the result
        a_reply("The Ninja is the pick."),                      # the outcome, spoken by the task
        a_reply("I found the Ninja."),                          # the follow-up
    ])
    result = browse_flow.run_flow(server.url, "prompt", CASE, FIXTURE, AUDIO, AUDIO, task_seconds=0.5)

    assert result["errors"] == []
    start, outcome, follow = result["start"], result["outcome"], result["followup"]
    assert [call[0] for call in start["calls"]] == ["browse"] and start["text"] == "I'll look into it."
    assert start["first_audio_s"] is not None and start["audio_s"] > 0
    assert outcome["text"] == "The Ninja is the pick." and outcome["first_audio_s"] is not None
    assert follow["text"] == "I found the Ninja." and follow["calls"] == [] and not follow["task_running"]
    assert result["agent_runs"] == 1 and result["agent_goals"] == ["research coffee makers"]
    # the chat shows the request's answer with the spoken outcome after it, not the report
    assistant = [m["content"] for m in result["ui"] if m["role"] == "assistant"]
    assert assistant == ["I'll look into it. The Ninja is the pick.", "I found the Ninja."]
    assert [a["turn"] for a in result["appended"]] == [0]
    score = result["score"]
    assert score["start_browse"] and not score["start_invented"] and score["outcome_named"]
    assert not score["chat_report"] and not score["followup_rebrowse"] and score["followup_named"]
    assert not any(t.name == browse_flow.BROWSE_THREAD for t in threading.enumerate())


def test_a_running_follow_up_asks_for_the_status_while_the_task_is_held_and_the_outcome_follows(realtime, quick):
    server = realtime(reply=[
        a_call("browse", '{"goal": "research coffee makers"}'),
        a_reply("I'll look into it."),
        a_call("browser_status"),
        a_reply("It is searching."),
        a_reply("The Ninja is the pick."),
    ])
    case = {**CASE, "followup": {"say": "How is the browser task going?", "when": "running"}}
    result = browse_flow.run_flow(server.url, "prompt", case, FIXTURE, AUDIO, AUDIO, task_seconds=0.5)

    assert result["errors"] == []
    follow = result["followup"]
    assert follow["task_running"] is True  # held open by the harness for the whole follow-up
    assert [call[0] for call in follow["calls"]] == ["browser_status"]
    outcome_event = next(e for e in result["turns"][1]["tools"] if e["name"] == "browser_status")
    assert "research coffee makers" in outcome_event["output"]  # the assistant's own status, not the stub's text
    assert result["outcome"]["text"] == "The Ninja is the pick."  # spoken after the follow-up's answer
    assert result["score"]["followup_status"] and not result["score"]["followup_rebrowse"]
    assert result["agent_runs"] == 1


def test_a_start_reply_that_never_calls_browse_has_no_task_and_no_outcome(realtime, quick, monkeypatch):
    monkeypatch.setattr(browse_flow, "GIVE_UP_AFTER", 0.1)
    server = realtime(reply=[a_reply("Here are a few top recommendations: the Breville Barista Pro, the Technivorm "
                                "Moccamaster and the Bonavita Connoisseur, all very popular with home baristas."),
                             a_reply("It was a guess.")])
    result = browse_flow.run_flow(server.url, "prompt", CASE, FIXTURE, AUDIO, AUDIO, task_seconds=0.5)

    assert result["outcome"] is None and result["agent_runs"] == 0
    score = result["score"]
    assert not score["start_browse"] and score["start_invented"] and score["outcome_spoken"] is False


def test_a_failed_reply_is_recorded_and_the_run_still_ends_cleanly(realtime, quick):
    server = realtime(reply=[[("error", "the model is gone")]])
    result = browse_flow.run_flow(server.url, "prompt", CASE, FIXTURE, AUDIO, AUDIO, task_seconds=0.5)

    assert result["errors"] and result["errors"][0].startswith("start: ")
    assert result["followup"] is None and result["score"]["start_browse"] is False
    assert not any(t.name == browse_flow.BROWSE_THREAD for t in threading.enumerate())


def test_the_summary_prints_every_case_and_the_overall_row(capsys):
    runs = []
    for case_id in ("one", "two"):
        run = {"case": case_id, "repeat": 1, "errors": [], "score": browse_flow.score_run(record(), FIXTURE)}
        runs.append(run)
    summary = browse_flow.print_summary(runs, ["one", "two"])
    out = capsys.readouterr().out
    assert "2 runs, 0 with errors" in out and "one" in out and "two" in out and "all" in out
    assert summary["overall"]["flags"]["start_browse"] == [2, 2]
    assert json.dumps(summary)  # JSON-serializable as written to the results file
