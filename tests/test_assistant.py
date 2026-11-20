"""The assistant: one speech-to-speech session per reply with vaice's skills as its tools, the confirmation requests
only the Run button settles, and the browser task, against a fake /v1/realtime server (tests/fake_realtime.py) and
fake skills and browser agent (nothing runs on this machine). The session itself is tested in test_llamacpp.py."""
import json
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
from fake_realtime import pcm
from threads import next_on_fresh_thread

from vaice import browse_task, lifecycle
from vaice.browse import BrowseDeclined, BrowseError, BrowseStopped
from vaice.confirm import Confirmations
from vaice.conversation import WAITING_FOR_CONFIRMATION, Turn
from vaice.memory import REFUSED_AFTER_TOOL, Memory, memory_skills
from vaice.outbox import Marker, Outbox
from vaice.responders import assistant, llamacpp
from vaice.responders.assistant import APPROVED, NOTES_HEADER, REPORT, SAY, WAITING, AssistantResponder
from vaice.skills import Skill, SkillError, SkillResult, Skills

AUDIO = (48000, np.zeros((1, 480), dtype=np.int16))
SPEECH = [("text", "All done."), ("audio", pcm(1, 2, 3)), ("done",)]  # a reply that only speaks
QUIT = ("quit_app", '{"name": "Safari"}')


@pytest.fixture(autouse=True)
def quick_polls(monkeypatch):
    monkeypatch.setattr(assistant, "POLL_SECONDS", 0.02)
    monkeypatch.setattr(llamacpp, "POLL_SECONDS", 0.02)


def calls_then_speech(*calls, speech=None):
    """The server's scripts for a reply in which the model makes `calls` (name, arguments) in one round and then
    speaks; every later session (the outcome of a click, a browser task) gets the last script again."""
    round_one = [("call", f"item_{i}", name, f"call_{i}", arguments) for i, (name, arguments) in enumerate(calls)]
    return [[*round_one, ("done",)], speech or SPEECH]


def drive(responder, turn=None, history=()):
    """All frames of one reply, resuming the generator on a fresh thread each time, like fastrtc.
    Raises the error that ended the reply."""
    gen = responder.respond(AUDIO, history, turn or Turn())
    frames = []
    while True:
        item = next_on_fresh_thread(gen, f"fastrtc-{len(frames)}")
        if isinstance(item, StopIteration):
            return frames
        if isinstance(item, BaseException):
            raise item
        frames.append(item)


def wait_until(condition, seconds=3.0):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if condition():
            return
        time.sleep(0.01)
    raise AssertionError("condition not reached")


def threads_named(name):
    return [t for t in threading.enumerate() if t.name == name]


def sessions(server):
    """The conversation sessions (not the transcription ones) that were opened, in order."""
    return [c for c in server.connections if c.session and c.session.get("type") != "transcription"]


def outputs(conn):
    """What the session sent back as the results of its tool calls, in order."""
    return [e["item"]["output"] for e in conn.events
            if e["type"] == "conversation.item.create" and e["item"]["type"] == "function_call_output"]


def spoken_requests(server):
    """The texts the assistant had a text-input session say (the outcome of a click, a browser task), in order."""
    return [event["item"]["content"][0]["text"] for conn in sessions(server) for event in conn.events
            if event["type"] == "conversation.item.create" and event["item"].get("role") == "user"]


def serve(outbox):
    """What the handler's idle emit() does, without pacing: the frames served and the markers run, in order."""
    frames = []
    while (item := outbox.poll(True)) is not None:
        if isinstance(item, Marker):
            item.callback()
        else:
            frames.append(item)
    return frames


def tool_events(turn):
    return [(e["name"], e["output"]) for e in turn.tool_events]


def speech_over():
    return not threads_named("vaice-assistant-speech")


def make_skills(ran):
    """disk_space (read), set_volume (action, always fails) and quit_app (confirm); `ran` collects the runs."""

    def disk(_args):
        ran.append(("disk_space", {}))
        return SkillResult("160 gigabytes free of 926 gigabytes", "raw: /dev/disk3s1 160G")

    def quit_run(args):
        ran.append(("quit_app", args))
        return SkillResult(f"quit {args['name']}")

    def question(args):
        if args["name"] == "Ghost":
            raise SkillError("Ghost is not running")
        return f"Shall I quit {args['name']}?"

    def broken(_args):
        raise SkillError("the volume could not be set")

    nothing = {"type": "object", "properties": {}}
    name = {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}
    return Skills([Skill("disk_space", "Report free disk space.", nothing, disk),
                   Skill("set_volume", "Set the volume.", nothing, broken, kind="action"),
                   Skill("quit_app", "Quit an app.", name, quit_run, kind="confirm", ask=question)])


def assistant_over(server, skills, outbox=True, **options):
    """An `AssistantResponder` on the fake server; `resp.outbox` is one primed like a draining connection."""
    resp = AssistantResponder(server.url, 5.0, "Be brief.", 1024, skills=skills, **options)
    resp.outbox = None
    if outbox:
        resp.outbox = Outbox()
        resp.outbox.touch()
        resp.attach_outbox(resp.outbox)
    return resp


# -- a read skill: it runs on the session thread and its output goes back to the session


def test_a_read_skill_call_runs_and_its_output_goes_back_to_the_session_and_onto_the_turn(realtime):
    ran = []
    server = realtime(reply=calls_then_speech(("disk_space", "{}")))
    resp = assistant_over(server, make_skills(ran))
    turn = Turn()

    frames = drive(resp, turn)

    assert ran == [("disk_space", {})]
    assert len(frames) == 1  # the model's answer
    (conn,) = sessions(server)
    assert [t["name"] for t in conn.session["tools"]] == ["disk_space", "set_volume", "quit_app"]  # flat format
    assert outputs(conn) == ["160 gigabytes free of 926 gigabytes"]
    assert tool_events(turn) == [("disk_space", "raw: /dev/disk3s1 160G")]  # the panel keeps the detail
    assert turn.assistant_text == "All done."


def test_a_failed_or_unknown_skill_is_a_result_the_model_reads_not_an_error(realtime):
    server = realtime(reply=calls_then_speech(("set_volume", "{}"), ("nonesuch", "{}")))
    resp = assistant_over(server, make_skills([]))

    drive(resp)

    first, second = outputs(sessions(server)[0])
    assert first == "the set volume skill failed: the volume could not be set"
    assert "nonesuch" in second


# -- a confirm skill: a request at once, nothing runs until the Run button


def test_a_confirm_call_creates_a_pending_request_at_once_and_runs_nothing(realtime):
    ran = []
    server = realtime(reply=calls_then_speech(QUIT))
    resp = assistant_over(server, make_skills(ran))
    turn = Turn()

    drive(resp, turn)

    pending = resp.pending()
    assert pending is not None and (pending.name, pending.arguments) == QUIT
    assert pending.question == "Shall I quit Safari?"
    assert ran == []  # nothing ran
    assert outputs(sessions(server)[0]) == [WAITING]  # the model is told that only the Run button runs it
    assert "Run" in WAITING and "Safari" not in WAITING  # the question text goes to the panel only
    assert tool_events(turn) == [("quit_app", WAITING_FOR_CONFIRMATION)]


def test_a_confirm_call_that_cannot_be_asked_about_is_an_ordinary_failed_call(realtime):
    ran = []
    server = realtime(reply=calls_then_speech(("quit_app", '{"name": "Ghost"}')))
    resp = assistant_over(server, make_skills(ran))

    drive(resp)

    assert resp.pending() is None and ran == []
    assert outputs(sessions(server)[0]) == ["the quit app skill failed: Ghost is not running"]


def test_run_runs_the_stored_call_and_the_outcome_is_spoken_through_the_outbox_and_recorded(realtime):
    ran = []
    server = realtime(reply=calls_then_speech(QUIT))
    resp = assistant_over(server, make_skills(ran))
    turn = Turn()
    drive(resp, turn)
    asked_text = turn.assistant_text

    assert resp.approve(resp.pending().id) == "quit Safari"

    assert ran == [("quit_app", {"name": "Safari"})]  # the stored call, not anything the model says now
    assert resp.pending() is None
    wait_until(speech_over)
    # a session of its own whose input is a text message: no tools, no audio
    assert spoken_requests(server) == [APPROVED.format(text="quit Safari")]
    speaker = sessions(server)[1]
    assert "tools" not in speaker.session and speaker.audio == b""
    assert ("quit_app", "quit Safari") in tool_events(turn)  # recorded on the asking turn
    assert turn.assistant_text == asked_text  # the words are recorded once their audio was served, not before
    assert len(serve(resp.outbox)) == 1
    assert turn.assistant_text == f"{asked_text} All done.".strip()


def test_a_clear_after_the_outcome_was_queued_drops_it_and_records_nothing(realtime):
    server = realtime(reply=calls_then_speech(QUIT))
    resp = assistant_over(server, make_skills([]))
    turn = Turn()
    drive(resp, turn)
    before = turn.assistant_text

    resp.approve(resp.pending().id)
    wait_until(speech_over)
    resp.outbox.clear()  # the user spoke

    assert serve(resp.outbox) == [] and turn.assistant_text == before


def test_a_stale_or_second_click_runs_nothing(realtime):
    ran = []
    server = realtime(reply=calls_then_speech(QUIT))
    resp = assistant_over(server, make_skills(ran))
    drive(resp)
    pending = resp.pending()

    assert resp.approve("an-old-panels-id") is None and ran == []
    assert resp.approve(pending.id) == "quit Safari"
    assert resp.approve(pending.id) is None  # settled once
    assert len(ran) == 1


def test_cancel_records_cancelled_runs_nothing_and_speaks_nothing(realtime):
    ran = []
    server = realtime(reply=calls_then_speech(QUIT))
    resp = assistant_over(server, make_skills(ran))
    turn = Turn()
    drive(resp, turn)

    resp.cancel(resp.pending().id)

    assert resp.pending() is None and ran == []
    assert ("quit_app", "cancelled") in tool_events(turn)
    assert speech_over() and len(sessions(server)) == 1  # no session was opened to speak
    assert len(resp.outbox) == 0


def test_new_conversation_drops_a_waiting_request_and_what_was_queued_to_be_said(realtime):
    ran = []
    server = realtime(reply=calls_then_speech(QUIT))
    resp = assistant_over(server, make_skills(ran))
    drive(resp)
    pending = resp.pending()
    resp.outbox.put_marker(lambda: None)

    resp.clear_pending()

    assert resp.pending() is None and len(resp.outbox) == 0
    assert resp.approve(pending.id) is None and ran == []  # the old panel's button settles nothing


def test_a_run_without_an_outbox_still_runs_and_records_but_speaks_nothing(realtime):
    ran = []
    server = realtime(reply=calls_then_speech(QUIT))
    resp = assistant_over(server, make_skills(ran), outbox=False)
    turn = Turn()
    drive(resp, turn)

    assert resp.approve(resp.pending().id) == "quit Safari"

    assert ran and ("quit_app", "quit Safari") in tool_events(turn)
    assert speech_over() and len(sessions(server)) == 1


def test_new_conversation_stops_a_run_a_click_started_and_a_stopped_run_is_not_reported(realtime):
    started, release = threading.Event(), threading.Event()

    def quit_run(args, stop):
        started.set()
        release.wait(5)
        return SkillResult("the command was stopped" if stop() else "quit Safari")

    skills = Skills([Skill("quit_app", "Quit an app.", {"type": "object", "properties": {"name": {"type": "string"}}},
                           lambda args: quit_run(args, lambda: False), kind="confirm",
                           ask=lambda args: "Shall I quit?", run_stoppable=quit_run)])
    server = realtime(reply=calls_then_speech(QUIT))
    resp = assistant_over(server, skills)
    drive(resp)
    clicked = threading.Thread(target=resp.approve, args=(resp.pending().id,), daemon=True)
    clicked.start()
    assert started.wait(3)

    resp.clear_pending()  # "New conversation" while the run is in flight
    release.set()
    clicked.join(5)

    assert not clicked.is_alive() and speech_over() and len(sessions(server)) == 1


# -- remember: refused after another skill in the same reply, and the notes in the prompt


def memory_skills_over(memory, ran):
    return Skills([*make_skills(ran).skills, *memory_skills(memory)])


NOTE = '{"note": "The user parks on level three."}'


def test_remember_is_stored_when_it_comes_first_and_refused_after_another_skill_in_the_same_reply(realtime):
    memory = Memory(None)
    server = realtime(reply=calls_then_speech(("remember", NOTE), ("disk_space", "{}"),
                                              ("remember", '{"note": "The user lives in Lisbon."}')))
    resp = assistant_over(server, memory_skills_over(memory, []), memory=memory)
    turn = Turn()

    drive(resp, turn)

    assert memory.notes() == ["The user parks on level three."]  # the first one, before any other skill
    first, _, third = outputs(sessions(server)[0])
    assert first.startswith("remembered") and third == REFUSED_AFTER_TOOL
    assert ("remember", REFUSED_AFTER_TOOL) in tool_events(turn)


@pytest.mark.parametrize("first", [("set_volume", "{}"), ("quit_app", '{"name": "Ghost"}'), ("nonesuch", "{}")])
def test_a_failed_unaskable_or_unknown_call_counts_as_a_skill_that_ran(realtime, first):
    memory = Memory(None)
    server = realtime(reply=calls_then_speech(first, ("remember", NOTE)))
    resp = assistant_over(server, memory_skills_over(memory, []), memory=memory)

    drive(resp)

    assert memory.notes() == []
    assert outputs(sessions(server)[0])[1] == REFUSED_AFTER_TOOL


def test_the_notes_are_in_the_prompt_behind_a_header_that_calls_them_facts_and_only_with_memory_on(realtime):
    memory = Memory(None)
    memory.add("The user parks on level three.")
    server = realtime(reply=SPEECH)

    drive(assistant_over(server, make_skills([]), memory=memory))
    drive(assistant_over(server, make_skills([])))  # no memory: no block at all

    with_notes, without = (c.session["instructions"] for c in sessions(server))
    assert NOTES_HEADER in with_notes and "- The user parks on level three." in with_notes
    assert NOTES_HEADER not in without


def test_a_note_remembered_in_one_reply_is_in_the_next_replys_prompt(realtime):
    memory = Memory(None)
    server = realtime(reply=calls_then_speech(("remember", NOTE)))
    resp = assistant_over(server, memory_skills_over(memory, []), memory=memory)

    drive(resp)
    drive(resp)

    first, second = (c.session["instructions"] for c in sessions(server))
    assert "level three" not in first and "level three" in second


def test_forget_everything_asks_and_only_the_run_button_erases_while_new_conversation_keeps_the_notes(realtime):
    memory = Memory(None)
    memory.add("The user parks on level three.")
    server = realtime(reply=calls_then_speech(("forget_everything", "{}")))
    resp = assistant_over(server, memory_skills_over(memory, []), memory=memory)
    drive(resp)
    assert resp.pending() is not None and memory.notes()  # asked, nothing erased

    resp.clear_pending()
    assert memory.notes() == ["The user parks on level three."]

    server.reply_scripts[:] = calls_then_speech(("forget_everything", "{}"))  # asked again
    server._replies_played = 0
    drive(resp)
    resp.approve(resp.pending().id)
    assert memory.notes() == []
    wait_until(speech_over)


def test_the_default_skills_have_the_memory_skills_only_with_memory_on():
    on = AssistantResponder("http://h", memory=Memory(None))
    off = AssistantResponder("http://h")

    assert {"remember", "forget", "forget_everything"} <= set(on.skills.names)
    assert not {"remember", "forget", "forget_everything"} & set(off.skills.names)
    assert "run_shell" not in off.skills.names and "browse" not in off.skills.names


# -- the browser task

GOAL = '{"goal": "open python.org and tell me the page title"}'
CLICK_STEP = ("mcp__chrome__click", '{"uid": "1_7"}')
STEP_QUESTION = "The browser task wants to click something on python.org. The details are on the screen."
BROWSE_ANSWER = "The page title is Welcome to Python.org."


class FakeAgent:
    """A BrowseAgent: `script(agent, goal, stop, on_progress, on_ask, deadline_s)` plays the task."""

    def __init__(self, script):
        self.script = script
        self.goals = []
        self.stops = []
        self.entered = threading.Event()
        self.approved = None

    def run(self, goal, *, stop, on_progress, on_ask, deadline_s, on_text=None):
        self.goals.append(goal)
        self.stops.append(stop)
        self.deadline = deadline_s
        self.entered.set()
        return self.script(self, goal, stop, on_progress, on_ask, deadline_s)

    def close(self):
        pass


def answers(agent, goal, stop, on_progress, on_ask, deadline_s):
    on_progress("mcp__chrome__navigate_page", {"url": "https://www.python.org/"})
    on_progress("mcp__chrome__take_snapshot", {})
    return BROWSE_ANSWER


def asks_then_answers(agent, goal, stop, on_progress, on_ask, deadline_s):
    on_progress("mcp__chrome__navigate_page", {"url": "https://www.python.org/"})
    agent.approved = on_ask(CLICK_STEP[0], CLICK_STEP[1], stop)
    if stop():
        raise BrowseStopped
    if not agent.approved:
        raise BrowseDeclined
    return BROWSE_ANSWER


def holds(release):
    """A task that works until `release` is set, and ends in a stop like the real agent when it is stopped."""

    def script(agent, goal, stop, on_progress, on_ask, deadline_s):
        while not release.is_set():
            if stop():
                raise BrowseStopped
            time.sleep(0.005)
        return BROWSE_ANSWER

    return script


def browse_over():
    return not threads_named("vaice-assistant-browse")


BROWSERS = []


@pytest.fixture(autouse=True)
def end_browser_tasks():
    """A test may leave a task waiting for an answer: stop them all, so the next test starts with none."""
    yield
    while BROWSERS:
        BROWSERS.pop().stop_task()
    wait_until(browse_over)


def browsing(realtime, script, outbox=True, **options):
    """A responder with the browse skills and a fake agent, driven through the reply in which the model calls
    `browse`; later sessions (the outcome, a heads-up) speak `SPEECH`."""
    server = realtime(reply=calls_then_speech(("browse", GOAL)))
    agent = FakeAgent(script)
    resp = assistant_over(server, None, outbox=outbox, browse=agent, **options)
    BROWSERS.append(resp)
    s = SimpleNamespace(resp=resp, server=server, agent=agent, turn=Turn(), outbox=resp.outbox)
    s.frames = drive(resp, s.turn)
    return s


def ask_again(s, *calls):
    """Another reply on the same responder, in which the model makes `calls`."""
    s.server.reply_scripts[:] = calls_then_speech(*calls)
    s.server._replies_played = 0
    drive(s.resp)
    return outputs(sessions(s.server)[-1])


def test_browse_starts_the_task_in_the_background_and_the_reply_ends_while_it_runs(realtime):
    release = threading.Event()

    s = browsing(realtime, holds(release))

    assert outputs(sessions(s.server)[0]) == [browse_task.WORKING]  # the model is told it works in the background
    assert s.agent.entered.wait(5) and s.agent.goals == ["open python.org and tell me the page title"]
    assert threads_named("vaice-assistant-browse") and s.resp.browsing()
    assert s.agent.deadline == 180.0
    assert spoken_requests(s.server) == []  # quiet while it works
    release.set()
    wait_until(browse_over)
    assert not s.resp.browsing()


def test_the_outcome_is_spoken_through_the_outbox_and_recorded_on_the_asking_turn(realtime):
    s = browsing(realtime, answers)

    wait_until(browse_over)

    assert ("browse", BROWSE_ANSWER) in tool_events(s.turn)
    assert spoken_requests(s.server) == [REPORT.format(text=f"the browser task finished: {BROWSE_ANSWER}")]
    spoken = s.turn.assistant_text
    serve(s.outbox)  # the handler serves the frames, then the marker that records the words
    assert s.turn.assistant_text == f"{spoken} All done.".strip()


def test_the_tools_a_task_uses_are_kept_for_the_status_and_never_spoken(realtime):
    release = threading.Event()

    def script(agent, goal, stop, on_progress, on_ask, deadline_s):
        on_progress("mcp__chrome__navigate_page", {"url": "https://news.ycombinator.com/"})
        on_progress("mcp__chrome__take_snapshot", {})
        return holds(release)(agent, goal, stop, on_progress, on_ask, deadline_s)

    s = browsing(realtime, script)
    wait_until(lambda: "2 tool calls" in s.resp._browser_status().detail)

    assert "now on news.ycombinator.com" in s.resp._browser_status().speech
    assert spoken_requests(s.server) == []
    release.set()


def test_a_step_that_needs_the_button_is_a_button_only_request_with_a_spoken_heads_up(realtime):
    s = browsing(realtime, asks_then_answers)

    wait_until(lambda: s.resp.pending() is not None)
    pending = s.resp.pending()

    assert pending.waiter is not None  # a question of the running task: nothing is run, the click is its answer
    assert (pending.name, pending.arguments) == CLICK_STEP  # the panel shows the tool and its arguments
    assert pending.question == STEP_QUESTION and "uid" not in pending.question
    wait_until(lambda: spoken_requests(s.server))
    assert spoken_requests(s.server) == [SAY.format(text=f"{STEP_QUESTION} Press Run to allow it, or Cancel.")]
    assert s.resp.browsing() and s.agent.approved is None


def test_run_is_the_yes_to_a_step_and_cancel_the_no_and_neither_runs_a_skill(realtime):
    s = browsing(realtime, asks_then_answers)
    wait_until(lambda: s.resp.pending() is not None)
    pending = s.resp.pending()

    assert s.resp.approve(pending.id) == "go ahead"
    wait_until(browse_over)

    assert s.agent.approved is True and s.resp.pending() is None
    assert s.resp.approve(pending.id) is None  # settled once

    t = browsing(realtime, asks_then_answers)
    wait_until(lambda: t.resp.pending() is not None)
    t.resp.cancel(t.resp.pending().id)
    wait_until(browse_over)
    assert t.agent.approved is False and ("browse", "declined") in tool_events(t.turn)
    assert SAY.format(text="Okay, I stopped there.") in spoken_requests(t.server)  # said once, in fixed words


def test_the_stop_button_and_new_conversation_end_the_task_quietly(realtime):
    s = browsing(realtime, holds(threading.Event()))
    assert s.agent.entered.wait(5) and s.resp.browsing()

    assert s.resp.stop_task() is True  # the Stop button

    wait_until(browse_over)
    assert s.agent.stops[0]() and not s.resp.browsing()
    assert ("browse", "stopped") in tool_events(s.turn)
    assert spoken_requests(s.server) == []  # a stop is quiet
    assert s.resp.stop_task() is False  # nothing left to stop

    t = browsing(realtime, holds(threading.Event()))
    assert t.agent.entered.wait(5)
    t.resp.clear_pending()  # "New conversation"
    wait_until(browse_over)
    assert t.agent.stops[0]() and not t.resp.browsing()
    assert spoken_requests(t.server) == []


def test_the_app_ending_stops_the_task_quietly(realtime):
    s = browsing(realtime, holds(threading.Event()))
    assert s.agent.entered.wait(5)

    lifecycle.stopping.set()

    wait_until(browse_over)
    assert spoken_requests(s.server) == []


def test_a_new_utterance_does_not_stop_the_task(realtime):
    release = threading.Event()
    s = browsing(realtime, holds(release))
    assert s.agent.entered.wait(5)

    ask_again(s, ("disk_space", "{}"))  # the user speaks while it works: another reply

    assert s.resp.browsing() and not s.agent.stops[0]()
    release.set()
    wait_until(browse_over)
    assert ("browse", BROWSE_ANSWER) in tool_events(s.turn)  # still reported on the turn that asked


def test_a_second_browse_while_one_runs_is_refused_and_starts_nothing(realtime):
    s = browsing(realtime, holds(threading.Event()))
    assert s.agent.entered.wait(5)

    (refused,) = ask_again(s, ("browse", '{"goal": "another"}'))

    assert refused.endswith(browse_task.BUSY)

    assert s.agent.goals == ["open python.org and tell me the page title"] and s.resp.browsing()


def test_the_status_and_stop_skills_are_the_models_way_to_ask_how_it_goes_and_to_end_it(realtime):
    s = browsing(realtime, holds(threading.Event()))
    assert s.agent.entered.wait(5)

    status, stopped = ask_again(s, ("browser_status", "{}"), ("stop_browsing", "{}"))

    assert status.startswith("the browser task is running: open python.org")
    assert stopped == browse_task.STOPPED
    wait_until(browse_over)


def test_the_browse_skills_are_offered_only_with_an_agent():
    names = {"browse", "browser_status", "stop_browsing"}

    assert not names & set(AssistantResponder("http://h").skills.names)
    assert names <= set(AssistantResponder("http://h", browse=FakeAgent(None)).skills.names)


def test_a_failure_and_a_crash_are_recorded_and_spoken_once_in_fixed_words(realtime):
    def failing(agent, goal, stop, on_progress, on_ask, deadline_s):
        raise BrowseError("the browser agent exited", "Sorry, the browser stopped working.")

    s = browsing(realtime, failing)
    wait_until(browse_over)
    assert spoken_requests(s.server) == [SAY.format(text="Sorry, the browser stopped working.")]
    assert ("browse", "failed: the browser agent exited") in tool_events(s.turn)

    def crashing(agent, goal, stop, on_progress, on_ask, deadline_s):
        raise RuntimeError("secret-key-123")

    t = browsing(realtime, crashing)
    wait_until(browse_over)
    assert spoken_requests(t.server) == [SAY.format(text=browse_task.FAILED)]
    assert ("browse", "failed: RuntimeError") in tool_events(t.turn)  # the message (could hold a key) is not kept
    assert "secret-key-123" not in json.dumps(t.turn.tool_events)


def test_without_an_outbox_the_task_still_runs_and_records_and_a_step_is_answerable_from_the_screen(realtime):
    s = browsing(realtime, asks_then_answers, outbox=False)
    wait_until(lambda: s.resp.pending() is not None)

    assert s.resp.approve(s.resp.pending().id) == "go ahead"
    wait_until(browse_over)

    assert ("browse", BROWSE_ANSWER) in tool_events(s.turn)
    assert spoken_requests(s.server) == []  # nothing to say it through


def test_a_click_cannot_answer_a_step_twice(realtime):
    confirmations = Confirmations()
    s = browsing(realtime, asks_then_answers, confirmations=confirmations)
    wait_until(lambda: confirmations.current() is not None)
    pending = confirmations.current()

    results = [s.resp.approve(pending.id), s.resp.approve(pending.id)]

    assert results.count("go ahead") == 1 and results.count(None) == 1
    wait_until(browse_over)


# -- "New conversation" while a reply is still preparing a call, and one utterance at a time


def test_new_conversation_while_a_confirm_call_is_prepared_leaves_no_request(realtime):
    preparing, release, ran = threading.Event(), threading.Event(), []

    def question(args):  # `quit_app` asks `lsappinfo` here; the user starts over meanwhile
        preparing.set()
        release.wait(5)
        return "Shall I quit Safari?"

    name = {"type": "object", "properties": {"name": {"type": "string"}}}
    skills = Skills([Skill("quit_app", "Quit an app.", name, lambda args: ran.append(args) or SkillResult("quit"),
                           kind="confirm", ask=question)])
    server = realtime(reply=calls_then_speech(QUIT))
    resp = assistant_over(server, skills)
    turn = Turn()
    reply = threading.Thread(target=drive, args=(resp, turn), daemon=True)
    reply.start()
    assert preparing.wait(3)

    resp.clear_pending()  # "New conversation"
    release.set()
    reply.join(5)

    assert resp.pending() is None and ran == []
    assert outputs(sessions(server)[0]) == [assistant.DROPPED]
    assert tool_events(turn) == [("quit_app", assistant.DROPPED)]


def test_a_browser_task_or_step_from_before_new_conversation_never_starts_or_asks(realtime):
    server = realtime(reply=SPEECH)
    agent = FakeAgent(answers)
    resp = assistant_over(server, None, browse=agent)
    BROWSERS.append(resp)
    began = resp._confirmations.epoch

    resp.clear_pending()
    resp._start_browse(Turn(), "browse", GOAL, began)

    assert not resp.browsing() and not agent.entered.wait(0.2)
    hooks = assistant._BrowseHooks(resp._confirmations, lambda request: None, browse_task.BrowseStatus("goal"),
                                   lambda: False, began)
    assert hooks.ask(*CLICK_STEP, lambda: False) is False and resp.pending() is None


def test_outcomes_are_spoken_one_at_a_time_and_a_clear_while_waiting_still_cancels(realtime, monkeypatch):
    active, most, generations = [0], [0], []
    first_in, release = threading.Event(), threading.Event()
    lock = threading.Lock()

    def fake_speak(url, timeout, max_tokens, outbox, turn, request, stop, generation):
        with lock:
            active[0] += 1
            most[0] = max(most[0], active[0])
            generations.append((request, generation))
        if request == "first":
            first_in.set()
            release.wait(5)
        with lock:
            active[0] -= 1

    monkeypatch.setattr(assistant, "speak", fake_speak)
    resp = assistant_over(realtime(reply=SPEECH), None)
    before = resp.outbox.generation
    first = threading.Thread(target=resp._say, args=(Turn(), "first"), daemon=True)
    first.start()
    assert first_in.wait(3)
    second = threading.Thread(target=resp._say, args=(Turn(), "second"), daemon=True)
    second.start()
    time.sleep(0.1)  # the second waits for the first

    resp.outbox.clear()  # the user spoke: what waits to be said is dropped
    release.set()
    first.join(5)
    second.join(5)

    assert most[0] == 1
    assert generations == [("first", before), ("second", before)]  # stale: its frames are refused
    assert resp.outbox.generation != before
