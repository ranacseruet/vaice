"""The assistant path: transcribe, brain, skills, /tts — against fakes (no server, no machine commands)."""
import json
import logging
import subprocess
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
from threads import next_on_fresh_thread, start_thread

from vaice import lifecycle, shell
from vaice.audio import from_pcm16
from vaice.brain import BrainReply, ToolCall
from vaice.confirm import CONFIRM_SECONDS, Confirmations, Pending
from vaice.conversation import Conversation, Turn
from vaice.responders import assistant
from vaice.responders.assistant import AssistantResponder, decide
from vaice.responders.llamacpp import _Stopped
from vaice.skills import Skill, SkillError, SkillResult, Skills

AUDIO = (48000, np.zeros((1, 480), dtype=np.int16))
CONTENT_TYPE = "audio/pcm;rate=24000;encoding=signed-int;bits=16"
FIRST = "The first sentence is long enough."
SECOND = "The second sentence follows it."


@pytest.fixture(autouse=True)
def quick_polls(monkeypatch):
    monkeypatch.setattr(assistant, "POLL_SECONDS", 0.02)


@pytest.fixture(autouse=True)
def heard(monkeypatch):
    """What the fake ASR returns; a test sets heard.text, or heard.error to make the transcription fail."""

    class Heard:
        text = "How much disk space is free?"
        error: BaseException | None = None

    def transcribe(url, timeout, audio):
        if Heard.error is not None:
            raise Heard.error
        return Heard.text

    monkeypatch.setattr(assistant, "_transcribe", transcribe)
    return Heard


class CountingTurn(Turn):
    """A Turn that remembers every transcript it was given (the field alone can't show a second write)."""

    def __init__(self):
        super().__init__()
        self.user_texts = []

    def set_user_text(self, text):
        self.user_texts.append(text)
        super().set_user_text(text)


def pcm_bytes(count, start=0):
    return np.arange(start, start + count, dtype="<i2").tobytes()


class FakeResponse:
    """A streaming /tts response: `chunks` of PCM bytes; with a `gate`, every chunk after the first waits for it."""

    def __init__(self, chunks, status=200, content_type=CONTENT_TYPE, gate=None):
        self.status_code = status
        self.ok = status < 400
        self.text = "the tts body"
        self.headers = {} if content_type is None else {"Content-Type": content_type}
        self._chunks = chunks
        self._gate = gate
        self.closed = False

    def iter_content(self, chunk_size):
        for index, chunk in enumerate(self._chunks):
            if index and self._gate is not None:
                assert self._gate.wait(5)
            yield chunk

    def close(self):
        self.closed = True


class FakeTTS:
    """Stands in for `requests.post`: every sentence gets `make(text)`'s response; the bodies are recorded."""

    def __init__(self, make=None):
        self.bodies = []
        self.responses = []
        self._make = make or (lambda text: FakeResponse([pcm_bytes(2400)]))

    def __call__(self, url, json=None, stream=None, timeout=None):
        assert url.endswith("/tts") and stream is True
        self.bodies.append(json)
        self.responses.append(self._make(json["input"]))
        return self.responses[-1]


@pytest.fixture
def tts(monkeypatch):
    fake = FakeTTS()
    monkeypatch.setattr(assistant.requests, "post", fake)
    return fake


def answer(text):
    return BrainReply(text, [], {"role": "assistant", "content": text})


def ask(*calls):
    """A brain reply that asks for skills: calls are (name, arguments)."""
    tool_calls = [ToolCall(f"call_{i}", name, arguments) for i, (name, arguments) in enumerate(calls)]
    message = {"role": "assistant", "content": "", "tool_calls": [
        {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": c.arguments}} for c in tool_calls]}
    return BrainReply("", tool_calls, message)


class FakeBrain:
    """Plays `replies` (a BrainReply or an exception) in order; remembers what each call was given."""

    def __init__(self, *replies):
        self._replies = list(replies)
        self.calls = []  # (messages as they were then, tools)

    def complete(self, messages, tools):
        self.calls.append((list(messages), tools))
        reply = self._replies.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        return reply


def disk_skills(log=None):
    def run(args):
        if log is not None:
            log.append(args)
        return SkillResult("160 gigabytes free of 926 gigabytes", "raw: /dev/disk3s1 160G")

    return Skills([Skill("disk_space", "Report free disk space.", {"type": "object", "properties": {}}, run)])


def responder(*replies, skills=None, voice="female"):
    brain = FakeBrain(*replies)
    return AssistantResponder("http://server", 5.0, "Be brief.", voice, skills or disk_skills(), brain), brain


def drive(resp, history=(), turn=None, audio=AUDIO):
    """All frames of one reply, resuming the generator on a fresh thread each time, like fastrtc.
    Raises the error that ended the reply."""
    gen = resp.respond(audio, history, turn or Turn())
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


# -- the reply


def test_a_plain_answer_is_spoken_and_recorded(tts):
    resp, _ = responder(answer("Paris is the capital of France."))
    turn = CountingTurn()

    frames = drive(resp, [("user", "Hi."), ("assistant", "Hello!")], turn)

    assert [rate for rate, _ in frames] == [24000]
    np.testing.assert_array_equal(frames[0][1], from_pcm16(pcm_bytes(2400)))
    assert frames[0][1].dtype == np.float32
    assert turn.assistant_text == "Paris is the capital of France."
    assert turn.user_texts == ["How much disk space is free?"]
    assert turn.tool_events == []
    assert [b["input"] for b in tts.bodies] == ["Paris is the capital of France."]


def test_the_brain_gets_the_prompt_the_clock_the_history_the_question_and_the_skills(tts):
    resp, brain = responder(answer("Fine, thanks for asking."))
    drive(resp, [("user", "Hi."), ("assistant", "Hello!")])

    messages, tools = brain.calls[0]
    assert messages[0]["role"] == "system"
    assert messages[0]["content"].startswith("Be brief.\nYou know the current date and time: ")
    assert messages[1:] == [{"role": "user", "content": "Hi."}, {"role": "assistant", "content": "Hello!"},
                            {"role": "user", "content": "How much disk space is free?"}]
    assert [t["function"]["name"] for t in tools] == ["disk_space"]


def test_a_skill_round_runs_the_skill_and_the_brain_answers_from_its_result(tts):
    ran = []
    resp, brain = responder(ask(("disk_space", "{}")), answer("You have 160 gigabytes free."),
                            skills=disk_skills(ran))
    turn = CountingTurn()

    drive(resp, turn=turn)

    assert ran == [{}]
    second_messages, second_tools = brain.calls[1]
    assert second_messages[-2] == {"role": "assistant", "content": "", "tool_calls": [
        {"id": "call_0", "type": "function", "function": {"name": "disk_space", "arguments": "{}"}}]}
    assert second_messages[-1] == {"role": "tool", "tool_call_id": "call_0",
                                   "content": "160 gigabytes free of 926 gigabytes"}
    assert second_tools is not None  # a second look at the tools is allowed
    assert turn.assistant_text == "You have 160 gigabytes free."
    assert turn.user_texts == ["How much disk space is free?"]


def test_each_skill_is_reported_on_the_turn_with_its_detail(tts):
    skills = Skills([
        Skill("disk_space", "d", {"type": "object", "properties": {}}, lambda _: SkillResult("160 free", "raw rows")),
        Skill("system_info", "s", {"type": "object", "properties": {}}, lambda _: SkillResult("macOS 15")),
    ])
    resp, _ = responder(ask(("disk_space", "{}"), ("system_info", "")), answer("Done with both of them."),
                        skills=skills)
    turn = Turn()

    drive(resp, turn=turn)

    assert [(e["name"], e["arguments"], e["output"]) for e in turn.tool_events] == [
        ("disk_space", "{}", "raw rows"), ("system_info", "", "macOS 15")]  # detail, else speech


def test_the_last_brain_call_is_made_without_tools_so_it_has_to_answer(tts):
    resp, brain = responder(ask(("disk_space", "{}")), ask(("disk_space", "{}")), answer("About 160 gigabytes."))
    turn = Turn()

    drive(resp, turn=turn)

    assert [tools is not None for _, tools in brain.calls] == [True, True, False]
    assert len(turn.tool_events) == 2
    assert turn.assistant_text == "About 160 gigabytes."


def test_a_skill_call_in_the_forced_round_is_ignored_and_the_reply_says_it_could_not_work_it_out(tts, caplog):
    resp, _ = responder(ask(("disk_space", "{}")), ask(("disk_space", "{}")), ask(("disk_space", "{}")))
    turn = Turn()

    with caplog.at_level(logging.WARNING, logger="vaice.responders.assistant"):
        drive(resp, turn=turn)

    assert len(turn.tool_events) == 2  # the third request was not run
    assert turn.assistant_text == "Sorry, I couldn't work that out."
    assert "ignored a skill call in the final round" in caplog.text


def test_the_skill_is_logged(tts, caplog):
    resp, _ = responder(ask(("disk_space", '{"x": 1}')), answer("Plenty is free."))

    with caplog.at_level(logging.INFO, logger="vaice.responders.assistant"):
        drive(resp)

    assert 'assistant skill disk_space({"x": 1})' in caplog.text


def test_the_timings_are_logged_in_one_line(tts, caplog):
    resp, _ = responder(ask(("disk_space", "{}")), answer("Plenty is free."))

    with caplog.at_level(logging.INFO, logger="vaice.responders.assistant"):
        drive(resp)

    lines = [r.message for r in caplog.records if r.message.startswith("assistant timings")]
    assert len(lines) == 1
    assert lines[0].startswith("assistant timings: transcript ")
    assert ", brain " in lines[0] and "s+" in lines[0] and lines[0].endswith("s") and ", first audio " in lines[0]


def test_the_voice_is_the_configured_one_and_the_request_asks_for_streamed_pcm16(tts):
    resp, _ = responder(answer("Paris is the capital of France."), voice="male")
    drive(resp)

    assert tts.bodies == [{"input": "Paris is the capital of France.", "voice": "male",
                           "response_format": "pcm16", "stream": True}]


def test_a_reply_is_spoken_one_sentence_per_tts_request_and_cleaned_for_speech(tts):
    resp, _ = responder(answer(f"**{FIRST}** Saved in /Users/x/Downloads/report.pdf, {SECOND.lower()}"))
    turn = Turn()

    drive(resp, turn=turn)

    assert [b["input"] for b in tts.bodies] == [
        FIRST, "Saved in report.pdf, the second sentence follows it."]
    assert turn.assistant_text == f"{FIRST} Saved in report.pdf, the second sentence follows it."


def test_the_recorded_text_grows_with_the_sentences_whose_audio_was_handed_out(tts):
    resp, _ = responder(answer(f"{FIRST} {SECOND}"))
    turn = Turn()
    gen = resp.respond(AUDIO, [], turn)

    next_on_fresh_thread(gen)
    assert turn.assistant_text == FIRST
    next_on_fresh_thread(gen)
    assert turn.assistant_text == f"{FIRST} {SECOND}"
    assert isinstance(next_on_fresh_thread(gen), StopIteration)


def test_an_empty_answer_is_replaced_by_an_apology(tts):
    resp, _ = responder(answer(""))
    turn = Turn()
    drive(resp, turn=turn)
    assert turn.assistant_text == "Sorry, I couldn't work that out."


# -- the transcript


def test_an_empty_transcript_gets_a_sorry_and_no_brain_call(tts, heard):
    heard.text = ""
    resp, brain = responder()
    turn = CountingTurn()

    drive(resp, turn=turn)

    assert brain.calls == []
    assert turn.assistant_text == "Sorry, I didn't catch that."
    assert turn.user_texts == [""]


def test_a_failed_transcription_fails_the_reply_after_reporting_an_empty_transcript(tts, heard):
    heard.error = RuntimeError("asr down")
    resp, brain = responder()
    turn = CountingTurn()

    with pytest.raises(RuntimeError, match="asr down"):
        drive(resp, turn=turn)

    assert turn.user_texts == [""]
    assert brain.calls == []


def test_a_stopped_transcription_still_reports_the_transcript_once(tts, heard):
    heard.error = _Stopped()
    resp, _ = responder()
    turn = CountingTurn()

    assert drive(resp, turn=turn) == []

    assert turn.user_texts == [""]


def test_a_brain_failure_is_raised_by_the_generator_and_the_transcript_was_reported_once(tts):
    resp, _ = responder(RuntimeError("chat completions answered HTTP 500: boom"))
    turn = CountingTurn()

    with pytest.raises(RuntimeError, match="HTTP 500"):
        drive(resp, turn=turn)

    assert turn.user_texts == ["How much disk space is free?"]


def test_the_transcript_is_reported_exactly_once_when_all_goes_well(tts):
    resp, _ = responder(ask(("disk_space", "{}")), answer("Plenty is free."))
    turn = CountingTurn()
    drive(resp, turn=turn)
    assert turn.user_texts == ["How much disk space is free?"]


# -- /tts


def test_a_tts_error_status_fails_the_reply_with_the_status_and_the_body(monkeypatch):
    fake = FakeTTS(lambda text: FakeResponse([], status=503))
    monkeypatch.setattr(assistant.requests, "post", fake)
    resp, _ = responder(answer("Paris is the capital of France."))

    with pytest.raises(RuntimeError, match="/tts answered HTTP 503: the tts body"):
        drive(resp)

    assert fake.responses[0].closed


def test_the_sample_rate_comes_from_the_response_header(monkeypatch):
    fake = FakeTTS(lambda text: FakeResponse([pcm_bytes(100)], content_type="audio/pcm;rate=16000;bits=16"))
    monkeypatch.setattr(assistant.requests, "post", fake)
    resp, _ = responder(answer("Paris is the capital of France."))

    assert [rate for rate, _ in drive(resp)] == [16000]


@pytest.mark.parametrize("content_type", [None, "audio/pcm", "application/json"])
def test_a_tts_response_without_a_sample_rate_fails_clearly(monkeypatch, content_type):
    fake = FakeTTS(lambda text: FakeResponse([pcm_bytes(100)], content_type=content_type))
    monkeypatch.setattr(assistant.requests, "post", fake)
    resp, _ = responder(answer("Paris is the capital of France."))

    with pytest.raises(RuntimeError, match="sent no sample rate"):
        drive(resp)

    assert fake.responses[0].closed


def test_a_chunk_that_ends_between_the_bytes_of_a_sample_is_reassembled(monkeypatch):
    data = pcm_bytes(10, start=100)
    pieces = [data[:3], data[3:4], data[4:11], data[11:]]  # odd, then a single byte, then odd, then the rest
    fake = FakeTTS(lambda text: FakeResponse(pieces))
    monkeypatch.setattr(assistant.requests, "post", fake)
    resp, _ = responder(answer("Paris is the capital of France."))

    frames = drive(resp)

    np.testing.assert_array_equal(np.concatenate([f for _, f in frames]), from_pcm16(data))
    assert all(len(f) for _, f in frames)  # no empty frame is handed out


def test_a_stream_that_ends_mid_sample_drops_the_dangling_byte_and_says_so(monkeypatch, caplog):
    data = pcm_bytes(4)
    fake = FakeTTS(lambda text: FakeResponse([data + b"\x07"]))
    monkeypatch.setattr(assistant.requests, "post", fake)
    resp, _ = responder(answer("Paris is the capital of France."))

    with caplog.at_level(logging.WARNING, logger="vaice.responders.assistant"):
        frames = drive(resp)

    np.testing.assert_array_equal(frames[0][1], from_pcm16(data))
    assert "ended in the middle of a sample" in caplog.text


def test_a_multi_chunk_reply_resumed_on_a_fresh_thread_each_time_plays_in_order(monkeypatch):
    fake = FakeTTS(lambda text: FakeResponse([pcm_bytes(50, 0), pcm_bytes(50, 50), pcm_bytes(50, 100)]))
    monkeypatch.setattr(assistant.requests, "post", fake)
    resp, _ = responder(answer(f"{FIRST} {SECOND}"))
    turn = Turn()

    frames = drive(resp, turn=turn)

    assert len(frames) == 6  # three chunks per sentence
    np.testing.assert_array_equal(np.concatenate([f for _, f in frames[:3]]), from_pcm16(pcm_bytes(150)))
    assert turn.assistant_text == f"{FIRST} {SECOND}"


# -- stopping


def test_a_barge_in_stops_the_generator_and_the_record_keeps_only_the_sentences_that_started(tts):
    resp, _ = responder(answer(f"{FIRST} {SECOND}"))
    turn = Turn()
    gen = resp.respond(AUDIO, [], turn)

    next_on_fresh_thread(gen)  # the first sentence's audio: handed out
    turn.closed.set()  # the next turn started

    assert isinstance(next_on_fresh_thread(gen), StopIteration)  # the second sentence's audio is dropped
    assert turn.assistant_text == FIRST


def test_a_barge_in_before_any_audio_ends_the_generator_with_nothing_recorded(tts):
    release = threading.Event()

    class Slow(FakeBrain):
        def complete(self, messages, tools):
            assert release.wait(5)
            return super().complete(messages, tools)

    brain = Slow(answer("Paris is the capital of France."))
    resp = AssistantResponder("http://server", 5.0, "Be brief.", "female", disk_skills(), brain)
    turn = Turn()
    gen = resp.respond(AUDIO, [], turn)
    running = start_thread(lambda: next(gen, None), "fastrtc-barge")

    turn.closed.set()  # the poll loop notices while the reply thread is still waiting for the brain
    assert running.join() is None
    release.set()
    wait_until(lambda: not threads_named("vaice-assistant"))
    assert turn.assistant_text == ""


def test_the_generator_ends_promptly_when_the_app_stops_even_though_the_worker_is_blocked(tts, caplog):
    release = threading.Event()

    entered = threading.Event()

    class Blocked(FakeBrain):
        def complete(self, messages, tools):
            entered.set()
            assert release.wait(5)
            return answer("late")

    resp = AssistantResponder("http://server", 5.0, "Be brief.", "female", disk_skills(), Blocked())
    gen = resp.respond(AUDIO, [], Turn())
    running = start_thread(lambda: next(gen, None), "fastrtc-stop")

    assert entered.wait(5)  # the worker is inside the brain call
    lifecycle.stopping.set()
    try:
        with caplog.at_level(logging.WARNING, logger="vaice.responders.assistant"):
            assert running.join(timeout=2) is None
        assert assistant.ABANDONED in caplog.text
    finally:
        release.set()
    wait_until(lambda: not threads_named("vaice-assistant"))  # the daemon thread winds down once unblocked


def test_closing_the_generator_stops_the_worker_and_closes_the_streaming_response(monkeypatch):
    gate = threading.Event()
    fake = FakeTTS(lambda text: FakeResponse([pcm_bytes(50), pcm_bytes(50), pcm_bytes(50)], gate=gate))
    monkeypatch.setattr(assistant.requests, "post", fake)
    resp, _ = responder(answer(f"{FIRST} {SECOND}"))
    gen = resp.respond(AUDIO, [], Turn())

    next_on_fresh_thread(gen)  # the first chunk; the worker now waits at the gate for the second
    gen.close()  # fastrtc dropped the reply
    gate.set()

    wait_until(lambda: fake.responses[0].closed)
    wait_until(lambda: not threads_named("vaice-assistant"))
    assert len(fake.responses) == 1  # the second sentence was never requested


def test_a_reply_that_fails_after_it_was_abandoned_is_logged(tts, caplog):
    release = threading.Event()

    class Failing(FakeBrain):
        def complete(self, messages, tools):
            assert release.wait(5)
            raise RuntimeError("brain died")

    resp = AssistantResponder("http://server", 5.0, "Be brief.", "female", disk_skills(), Failing())
    turn = Turn()
    gen = resp.respond(AUDIO, [], turn)
    running = start_thread(lambda: next(gen, None), "fastrtc-fail")
    turn.closed.set()
    assert running.join() is None

    with caplog.at_level(logging.WARNING, logger="vaice.responders.assistant"):
        release.set()
        wait_until(lambda: not threads_named("vaice-assistant"))

    assert "failed after it was abandoned: RuntimeError: brain died" in caplog.text


# -- decide, directly


def test_decide_returns_the_answer_the_calls_and_the_timings(tts):
    brain = FakeBrain(ask(("disk_space", "{}")), answer("About 160 gigabytes."))
    seen = []

    decision = decide(brain, disk_skills(), [{"role": "user", "content": "space?"}], lambda: False, seen.append,
                      turn=Turn())

    assert decision.answer == "About 160 gigabytes."
    assert [(c.name, c.arguments, c.result.speech) for c in decision.calls] == [
        ("disk_space", "{}", "160 gigabytes free of 926 gigabytes")]
    assert seen == decision.calls
    assert len(decision.brain_seconds) == 2


def test_decide_gives_a_failed_skill_to_the_brain_as_plain_words():
    brain = FakeBrain(ask(("nope", "{}")), answer("I could not check that."))
    messages = [{"role": "user", "content": "x"}]

    decide(brain, disk_skills(), messages, lambda: False, turn=Turn())

    assert messages[-1] == {"role": "tool", "tool_call_id": "call_0", "content": "there is no skill called nope"}


def test_decide_backfills_the_content_of_the_assistant_message():
    call = ToolCall("c1", "disk_space", "{}")
    reply = BrainReply("", [call], {"role": "assistant", "content": None, "tool_calls": []})
    messages = [{"role": "user", "content": "x"}]

    decide(FakeBrain(reply, answer("ok then, fine.")), disk_skills(), messages, lambda: False, turn=Turn())

    assert messages[1]["content"] == ""


TOOL_CALL_MARKUP = ("<tool_call>\n<function=open_web>\n<parameter=query>\nwhy is the sky blue\n</parameter>\n"
                    "</function>\n</tool_call>")


@pytest.mark.parametrize(("written", "spoken"), [
    (f"Let me check. {TOOL_CALL_MARKUP}", "Let me check."),
    (f"{TOOL_CALL_MARKUP} Because of scattering.", "Because of scattering."),
    ("Sure. <tool_call>\n<function=open_web>\n<parameter=query>\nwhy is the sky bl", "Sure."),  # never closed
    ("<function=open_web><parameter=query>x</parameter></function> Done.", "Done."),
    ("<answer>The sky is blue.</answer><|im_end|>", "The sky is blue."),  # stray delimiters, as in chat mode
    ("Two is less than three, so 2 < 3.", "Two is less than three, so 2 < 3."),  # ordinary text is untouched
])
def test_decide_never_lets_tool_call_markup_into_the_answer(caplog, written, spoken):
    with caplog.at_level(logging.WARNING, logger="vaice.responders.assistant"):
        decision = decide(FakeBrain(answer(written)), disk_skills(), [{"role": "user", "content": "x"}],
                          lambda: False, turn=Turn())

    assert decision.answer == spoken
    assert ("carried model markup" in caplog.text) == (written.strip() != spoken)


def test_an_answer_that_is_only_markup_becomes_the_no_answer_sentence(caplog):
    with caplog.at_level(logging.WARNING, logger="vaice.responders.assistant"):
        decision = decide(FakeBrain(answer(TOOL_CALL_MARKUP)), disk_skills(), [{"role": "user", "content": "x"}],
                          lambda: False, turn=Turn())

    assert decision.answer == assistant.NO_ANSWER
    assert "carried model markup" in caplog.text


def test_the_forced_final_round_cannot_speak_its_tool_call_markup():
    brain = FakeBrain(ask(("disk_space", "{}")), ask(("disk_space", "{}")), answer(f"Checking. {TOOL_CALL_MARKUP}"))

    decision = decide(brain, disk_skills(), [{"role": "user", "content": "x"}], lambda: False, turn=Turn())

    assert decision.answer == "Checking."


def test_an_approved_skills_answer_is_cleaned_and_falls_back_to_its_result():
    for written, spoken in [(f"Safari is closed. {TOOL_CALL_MARKUP}", "Safari is closed."),
                            (TOOL_CALL_MARKUP, "quit Safari")]:
        s = confirming(answer(written))
        request = Pending("quit_app", QUIT[1], QUESTION, "call_0", [], Turn())

        decision = assistant._approved(request, s.resp.skills, s.brain, Turn(), lambda: False)

        assert decision.answer == spoken


def test_decide_stops_before_a_brain_call_and_between_skills():
    with pytest.raises(_Stopped):
        decide(FakeBrain(), disk_skills(), [], lambda: True, turn=Turn())

    stops = iter([False, False, True])  # before round 1, before the first skill, before the second
    brain = FakeBrain(ask(("disk_space", "{}"), ("disk_space", "{}")))
    ran = []
    with pytest.raises(_Stopped):
        decide(brain, disk_skills(ran), [], lambda: next(stops), turn=Turn())
    assert ran == [{}]


def test_the_default_responder_offers_the_default_skills():
    resp = AssistantResponder("http://server")
    assert resp.name == "assistant"
    assert resp.skills.names == ["disk_space", "battery_status", "system_info", "top_processes", "find_files",
                                 "open_app", "quit_app", "set_volume", "volume_up", "volume_down", "mute_sound"]


def test_the_recorded_text_never_includes_what_was_not_handed_out_even_when_the_worker_ran_ahead(tts):
    resp, _ = responder(answer(f"{FIRST} {SECOND}"))
    turn = Turn()
    gen = resp.respond(AUDIO, [], turn)
    next_on_fresh_thread(gen)
    wait_until(lambda: len(tts.bodies) == 2)  # the second sentence's audio is already queued

    assert turn.assistant_text == FIRST
    gen.close()


def test_a_slow_brain_does_not_end_the_reply(tts):
    class Slow(FakeBrain):
        def complete(self, messages, tools):
            time.sleep(0.1)  # several polls of the generator
            return super().complete(messages, tools)

    resp = AssistantResponder("http://server", 5.0, "Be brief.", "female", disk_skills(),
                              Slow(answer("Paris is the capital of France.")))
    turn = Turn()

    assert len(drive(resp, turn=turn)) == 1
    assert turn.assistant_text == "Paris is the capital of France."


def test_an_app_stop_during_the_transcription_ends_the_reply_before_the_brain(tts, monkeypatch, caplog):
    def transcribe(url, timeout, audio):
        lifecycle.stopping.set()
        return "How much disk space is free?"

    monkeypatch.setattr(assistant, "_transcribe", transcribe)
    resp, brain = responder(answer("never asked"))
    turn = CountingTurn()

    with caplog.at_level(logging.WARNING, logger="vaice.responders.assistant"):
        assert drive(resp, turn=turn) == []

    assert brain.calls == []
    assert turn.user_texts == ["How much disk space is free?"]  # the transcript that was heard is still reported
    assert assistant.ABANDONED in caplog.text


def test_an_app_stop_that_the_reply_thread_notices_first_is_still_logged_by_the_generator(tts, monkeypatch, caplog):
    monkeypatch.setattr(assistant, "POLL_SECONDS", 5.0)  # the generator sits in its wait when the stop comes

    class Stopping(FakeBrain):
        def complete(self, messages, tools):
            time.sleep(0.1)  # long enough for the generator to be waiting
            lifecycle.stopping.set()
            return answer("never spoken")

    resp = AssistantResponder("http://server", 5.0, "Be brief.", "female", disk_skills(), Stopping())
    gen = resp.respond(AUDIO, [], Turn())

    with caplog.at_level(logging.WARNING, logger="vaice.responders.assistant"):
        assert start_thread(lambda: next(gen, None), "fastrtc-app-stop").join() is None

    assert assistant.ABANDONED in caplog.text


# --- earlier skill results in the prompt


def test_the_assistant_asks_for_tool_history_and_the_chat_responder_does_not():
    from vaice.responders.llamacpp import LlamaCppResponder

    assert AssistantResponder.tool_history is True
    assert not getattr(LlamaCppResponder, "tool_history", False)


def test_a_tool_entry_becomes_a_tool_exchange_before_the_turns_answer():
    history = [("user", "How long has it been on?"), ("tool", "system_info: up 3 days, 48 gigabytes of memory"),
               ("assistant", "Three days.")]

    messages = assistant.build_messages("Be brief.", history, "And memory?")

    assert messages[1:] == [
        {"role": "user", "content": "How long has it been on?"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "h1", "type": "function", "function": {"name": "system_info", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "h1", "content": "up 3 days, 48 gigabytes of memory"},
        {"role": "assistant", "content": "Three days."},
        {"role": "user", "content": "And memory?"},
    ]


def test_several_tool_entries_of_one_turn_are_one_assistant_message_with_several_calls():
    history = [("user", "q"), ("tool", "disk_space: 160 free"), ("tool", "battery_status: 80%"),
               ("assistant", "a")]

    messages = assistant.build_messages("p", history, "next")[2:-2]

    assert [m["role"] for m in messages] == ["assistant", "tool", "tool"]
    assert [c["function"]["name"] for c in messages[0]["tool_calls"]] == ["disk_space", "battery_status"]
    assert [m["tool_call_id"] for m in messages[1:]] == [c["id"] for c in messages[0]["tool_calls"]]
    assert [m["content"] for m in messages[1:]] == ["160 free", "80%"]


def test_tool_call_ids_are_unique_across_turns_and_output_may_hold_colons():
    history = [("user", "q1"), ("tool", "disk_space: a: b"), ("assistant", "a1"),
               ("user", "q2"), ("tool", "disk_space: c"), ("assistant", "a2")]

    messages = assistant.build_messages("p", history, "q3")
    ids = [c["id"] for m in messages for c in m.get("tool_calls", [])]

    assert len(ids) == len(set(ids)) == 2
    assert [m["content"] for m in messages if m["role"] == "tool"] == ["a: b", "c"]
    assert [m["role"] for m in messages] == [
        "system", "user", "assistant", "tool", "assistant", "user", "assistant", "tool", "assistant", "user"]


# --- confirmation of a skill that cannot be undone

QUIT = ("quit_app", '{"name": "Safari"}')
QUESTION = "Shall I quit Safari?"


class Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


def skills_with_quit(ran, disk_ran):
    """disk_space (a read skill) and quit_app (confirm): `ran` collects what quit_app was run with."""

    def quit_run(args):
        ran.append(args)
        return SkillResult("quit Safari")

    def question(args):
        if args["name"] == "Ghost":
            raise SkillError("Ghost is not running")
        return f"Shall I quit {args['name']}?"

    def disk(_args):
        disk_ran.append(1)
        return SkillResult("160 gigabytes free of 926 gigabytes")

    name = {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}
    return Skills([Skill("disk_space", "Report free disk space.", {"type": "object", "properties": {}}, disk),
                   Skill("quit_app", "Quit an app.", name, quit_run, kind="confirm", ask=question)])


def confirming(*replies, clock=None):
    ran, disk_ran, brain = [], [], FakeBrain(*replies)
    confirmations = Confirmations(clock or Clock())
    resp = AssistantResponder("http://server", 5.0, "Be brief.", "female", skills_with_quit(ran, disk_ran), brain,
                              confirmations)
    return SimpleNamespace(resp=resp, brain=brain, confirmations=confirmations, ran=ran, disk_ran=disk_ran)


def asked(*more, clock=None, request=QUIT):
    """A reply in which the brain asks for `request`, driven to its end; `more` are the brain's later replies."""
    s = confirming(ask(request), *more, clock=clock)
    s.turn = Turn()
    drive(s.resp, turn=s.turn)
    return s


def tool_events(turn):
    return [(e["name"], e["output"]) for e in turn.tool_events]


def test_a_confirm_skill_does_not_run_its_question_is_spoken_and_only_then_armed(tts):
    s = asked()

    assert s.ran == []
    assert [b["input"] for b in tts.bodies] == [QUESTION]
    assert s.turn.assistant_text == QUESTION
    assert len(s.brain.calls) == 1  # the loop ended at the question
    assert tool_events(s.turn) == [("quit_app", "waiting for confirmation")]
    assert s.turn.tool_events[0]["arguments"] == QUIT[1]
    pending = s.confirmations.current()
    assert (pending.name, pending.arguments, pending.question, pending.call_id) == (
        "quit_app", QUIT[1], QUESTION, "call_0")
    assert pending.turn is s.turn
    assert pending.armed_at == 100.0


def test_the_request_is_on_the_screen_while_its_question_is_spoken_but_cannot_be_answered_yet(monkeypatch):
    gate = threading.Event()
    monkeypatch.setattr(assistant.requests, "post",
                        FakeTTS(lambda text: FakeResponse([pcm_bytes(50), pcm_bytes(50)], gate=gate)))
    s = confirming(ask(QUIT))
    gen = s.resp.respond(AUDIO, [], Turn())

    next_on_fresh_thread(gen)  # the first chunk of the question is out; the worker waits at the gate

    pending = s.resp.pending()
    assert pending is not None and pending.armed_at is None
    gate.set()
    while not isinstance(next_on_fresh_thread(gen), StopIteration):
        pass
    wait_until(lambda: s.resp.pending().armed_at is not None)


def test_a_question_cut_off_before_its_audio_was_all_out_is_dropped(monkeypatch):
    gate = threading.Event()
    monkeypatch.setattr(assistant.requests, "post",
                        FakeTTS(lambda text: FakeResponse([pcm_bytes(50), pcm_bytes(50)], gate=gate)))
    s = confirming(ask(QUIT))
    gen = s.resp.respond(AUDIO, [], Turn())

    next_on_fresh_thread(gen)
    assert s.resp.pending() is not None
    gen.close()  # the user talked over it
    gate.set()

    wait_until(lambda: s.resp.pending() is None)
    wait_until(lambda: not threads_named("vaice-assistant"))
    assert s.ran == []


def test_a_question_whose_audio_failed_is_dropped_too(monkeypatch):
    monkeypatch.setattr(assistant.requests, "post", FakeTTS(lambda text: FakeResponse([], status=503)))
    s = confirming(ask(QUIT))

    with pytest.raises(RuntimeError, match="HTTP 503"):
        drive(s.resp)

    assert s.resp.pending() is None


def test_calls_after_the_confirm_skill_are_dropped_and_calls_before_it_are_answered(tts, caplog):
    s = confirming(ask(("disk_space", "{}"), QUIT, ("disk_space", "{}")))
    turn = Turn()

    with caplog.at_level(logging.WARNING, logger="vaice.responders.assistant"):
        drive(s.resp, turn=turn)

    assert s.disk_ran == [1]  # the call before ran, the one after did not
    assert "dropped 1 skill call(s) requested after quit_app" in caplog.text
    messages = s.resp.pending().messages
    assert [c["id"] for c in messages[2]["tool_calls"]] == ["call_0", "call_1"]
    assert messages[3] == {"role": "tool", "tool_call_id": "call_0", "content": "160 gigabytes free of 926 gigabytes"}
    assert len(messages) == 4  # a valid chat: the result of the confirm call is all that is missing
    assert tool_events(turn) == [("disk_space", "160 gigabytes free of 926 gigabytes"),
                                 ("quit_app", "waiting for confirmation")]


def test_a_confirm_skill_that_cannot_be_asked_about_is_an_ordinary_failed_call(tts):
    s = confirming(ask(("quit_app", '{"name": "Ghost"}')), answer("Ghost is not running, so nothing to quit."))
    turn = Turn()

    drive(s.resp, turn=turn)

    assert s.resp.pending() is None
    assert s.brain.calls[1][0][-1] == {"role": "tool", "tool_call_id": "call_0",
                                       "content": "the quit app skill failed: Ghost is not running"}
    assert tool_events(turn) == [("quit_app", "the quit app skill failed: Ghost is not running")]
    assert turn.assistant_text == "Ghost is not running, so nothing to quit."


def test_decide_stops_at_the_confirm_skill_with_its_question_as_the_answer():
    s = confirming(ask(QUIT))
    messages = [{"role": "user", "content": "quit Safari"}]
    turn = Turn()

    decision = decide(s.brain, s.resp.skills, messages, lambda: False, turn=turn)

    assert decision.answer == QUESTION and decision.calls == []
    assert decision.pending.turn is turn and decision.pending.messages is messages
    assert messages[-1]["tool_calls"][0]["function"] == {"name": "quit_app", "arguments": QUIT[1]}
    assert s.ran == []


def test_yes_runs_the_stored_call_makes_one_tool_less_brain_call_and_is_not_a_question(tts, heard):
    s = asked(answer("Safari is closed now."))
    heard.text = "Yes, please!"
    second = CountingTurn()

    drive(s.resp, turn=second)

    assert s.ran == [{"name": "Safari"}]  # the stored arguments
    assert len(s.brain.calls) == 2
    messages, tools = s.brain.calls[1]
    assert tools is None  # the brain only has to say how it went
    assert messages[-1] == {"role": "tool", "tool_call_id": "call_0", "content": "quit Safari"}
    assert messages[-2]["tool_calls"][0]["id"] == "call_0"
    assert not any(m.get("content") == "Yes, please!" for m in messages)  # "yes" never reaches the brain
    assert [b["input"] for b in tts.bodies] == [QUESTION, "Safari is closed now."]
    assert second.assistant_text == "Safari is closed now."
    assert second.user_texts == ["Yes, please!"]
    assert tool_events(second) == [("quit_app", "quit Safari")]  # on this turn
    assert tool_events(s.turn) == [("quit_app", "waiting for confirmation")]
    assert s.resp.pending() is None


def test_a_stop_before_the_approved_skill_runs_runs_nothing():
    s = confirming()
    request = Pending("quit_app", QUIT[1], QUESTION, "call_0", [], Turn())

    with pytest.raises(_Stopped):
        assistant._approved(request, s.resp.skills, s.brain, Turn(), lambda: True)

    assert s.ran == [] and s.brain.calls == []


def test_an_empty_answer_after_the_skill_ran_speaks_its_result(tts, heard):
    s = asked(answer(""))
    heard.text = "yes"
    turn = Turn()

    drive(s.resp, turn=turn)

    assert turn.assistant_text == "quit Safari"


def test_no_cancels_and_says_so_without_asking_the_brain(tts, heard):
    s = asked()
    heard.text = "No, thanks."
    turn = Turn()

    drive(s.resp, turn=turn)

    assert s.ran == []
    assert len(s.brain.calls) == 1
    assert turn.assistant_text == "Okay, I won't."
    assert tool_events(turn) == [("quit_app", "cancelled")]
    assert tool_events(s.turn) == [("quit_app", "waiting for confirmation")]
    assert s.resp.pending() is None


@pytest.mark.parametrize("utterance", ["What's the weather like?", "yes, and open Notes too", "Maybe later",
                                       "I said no"])
def test_anything_else_drops_the_request_and_is_an_ordinary_question(tts, heard, utterance):
    s = asked(answer("It is sunny today."))
    heard.text = utterance
    turn = Turn()

    drive(s.resp, turn=turn)

    assert s.ran == []
    assert s.brain.calls[1][0][-1] == {"role": "user", "content": utterance}  # asked as a new question
    assert turn.assistant_text == "It is sunny today."
    assert tool_events(s.turn) == [("quit_app", "waiting for confirmation"), ("quit_app", "not confirmed")]
    assert s.resp.pending() is None


def test_a_yes_cannot_approve_a_request_whose_question_was_not_fully_out(tts, heard):
    s = confirming(answer("Okay, sure thing."))
    request = Pending("quit_app", QUIT[1], QUESTION, "call_0", [], Turn())
    s.confirmations.ask(request)  # asked, never armed
    heard.text = "yes"

    drive(s.resp)

    assert s.ran == []
    assert s.brain.calls[0][0][-1] == {"role": "user", "content": "yes"}
    assert tool_events(request.turn) == [("quit_app", "not confirmed")]


def test_a_request_expires_a_minute_after_its_question_was_spoken(tts, heard):
    clock = Clock()
    s = asked(answer("I am not sure what you mean."), clock=clock)
    clock.now += CONFIRM_SECONDS + 1
    heard.text = "yes"

    drive(s.resp)

    assert s.ran == []
    assert s.brain.calls[1][0][-1] == {"role": "user", "content": "yes"}


def test_a_request_is_still_answerable_just_inside_the_minute(tts, heard):
    clock = Clock()
    s = asked(answer("Safari is closed now."), clock=clock)
    clock.now += CONFIRM_SECONDS - 1
    heard.text = "yes"

    drive(s.resp)

    assert s.ran == [{"name": "Safari"}]


def test_a_plain_question_with_nothing_waiting_leaves_no_request(tts):
    s = confirming(answer("About 160 gigabytes."))

    drive(s.resp)

    assert s.resp.pending() is None and len(s.brain.calls) == 1


def test_a_click_on_the_current_request_runs_it_and_records_the_result_on_the_asking_turn(tts):
    s = asked()
    pending = s.resp.pending()

    assert s.resp.approve(pending.id) == "quit Safari"

    assert s.ran == [{"name": "Safari"}]
    assert tool_events(s.turn) == [("quit_app", "waiting for confirmation"), ("quit_app", "quit Safari")]
    assert s.resp.pending() is None
    assert s.resp.approve(pending.id) is None  # once only
    assert s.ran == [{"name": "Safari"}]
    assert len(s.brain.calls) == 1  # nothing is spoken or asked of the brain on a click


def test_a_click_on_a_stale_panel_changes_nothing(tts):
    s = asked()
    pending = s.resp.pending()

    assert s.resp.approve("0" * 32) is None
    s.resp.cancel("0" * 32)

    assert s.resp.pending() is pending and s.ran == []
    assert tool_events(s.turn) == [("quit_app", "waiting for confirmation")]


def test_an_old_panels_id_cannot_approve_a_newer_request(tts, heard):
    s = asked(ask(("quit_app", '{"name": "Mail"}')))
    old = s.resp.pending().id
    heard.text = "Quit Mail."
    drive(s.resp)  # a second request replaces the first (which was "not confirmed")
    assert s.resp.pending().id != old

    assert s.resp.approve(old) is None

    assert s.ran == []
    assert s.resp.pending() is not None


def test_cancel_by_click_drops_the_request_and_records_it(tts):
    s = asked()

    s.resp.cancel(s.resp.pending().id)

    assert s.resp.pending() is None and s.ran == []
    assert tool_events(s.turn) == [("quit_app", "waiting for confirmation"), ("quit_app", "cancelled")]


def test_a_voice_yes_after_a_click_is_an_ordinary_question_and_the_skill_ran_once(tts, heard):
    s = asked(answer("Hello to you too."))
    s.resp.approve(s.resp.pending().id)
    heard.text = "yes"

    drive(s.resp)

    assert s.ran == [{"name": "Safari"}]
    assert s.brain.calls[1][0][-1] == {"role": "user", "content": "yes"}


def test_new_conversation_clears_a_waiting_request(tts):
    s = asked()

    s.resp.clear_pending()

    assert s.resp.pending() is None
    assert s.resp.approve("anything") is None


def test_a_follow_up_after_a_click_sees_what_the_action_did(tts):
    s = confirming(ask(QUIT))
    conversation = Conversation()
    first = conversation.start_turn()
    drive(s.resp, turn=first)
    first.finish()  # as the app does when the reply ran to its end
    s.resp.approve(s.resp.pending().id)

    history = conversation.history(before=conversation.start_turn(), tools=True)

    assert history == [("user", "How much disk space is free?"),
                       ("tool", "quit_app: waiting for confirmation"), ("tool", "quit_app: quit Safari"),
                       ("assistant", QUESTION)]
    messages = assistant.build_messages("Be brief.", history, "Now start it again.")
    assert [m["content"] for m in messages if m["role"] == "tool"] == ["waiting for confirmation", "quit Safari"]


def test_a_follow_up_after_a_spoken_yes_sees_the_result_on_the_yes_turn(tts, heard):
    s = confirming(ask(QUIT), answer("Safari is closed now."))
    conversation = Conversation()
    first = conversation.start_turn()
    drive(s.resp, turn=first)
    first.finish()
    heard.text = "yes"
    second = conversation.start_turn()
    drive(s.resp, turn=second)
    second.finish()

    history = conversation.history(before=conversation.start_turn(), tools=True)

    assert history[-3:] == [("user", "yes"), ("tool", "quit_app: quit Safari"),
                            ("assistant", "Safari is closed now.")]


# --- shell commands: two approval levels (see shell.py)

PWD = ("run_shell", json.dumps({"command": "pwd"}))  # harmless by form: a spoken yes may approve it
PIPE = ("run_shell", json.dumps({"command": "echo hi | tr a-z A-Z"}))  # needs zsh: only the button
SLEEP = ("run_shell", json.dumps({"command": "sleep 30"}))


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    return tmp_path


@pytest.fixture
def popen_log(monkeypatch):
    """The argv lists subprocess.Popen was given (the real Popen still runs them)."""
    started = []
    real = subprocess.Popen
    monkeypatch.setattr(shell.subprocess, "Popen", lambda argv, **kw: started.append(argv) or real(argv, **kw))
    return started


def shelling(*replies):
    brain = FakeBrain(*replies)
    confirmations = Confirmations(Clock())
    resp = AssistantResponder("http://server", 5.0, "Be brief.", "female", Skills([shell.shell_skill()]), brain,
                              confirmations)
    return SimpleNamespace(resp=resp, brain=brain, confirmations=confirmations)


def shell_asked(request, *more):
    s = shelling(ask(request), *more)
    s.turn = Turn()
    drive(s.resp, turn=s.turn)
    return s


def test_a_harmless_command_is_asked_by_voice_and_a_spoken_yes_runs_its_argv_without_a_shell(
        tts, heard, home, popen_log):
    s = shell_asked(PWD, answer("You are in your home folder."))
    pending = s.resp.pending()
    assert pending.voice and pending.question == "Shall I run the command pwd? The full command is on the screen."
    assert popen_log == []  # nothing runs when the model asks
    heard.text = "yes"
    turn = Turn()

    drive(s.resp, turn=turn)

    assert popen_log == [["pwd"]]
    messages, tools = s.brain.calls[1]
    assert tools is None
    assert messages[-1] == {"role": "tool", "tool_call_id": "call_0",
                            "content": f"the command finished. Output: {home.resolve()}"}
    assert turn.assistant_text == "You are in your home folder."
    assert s.resp.pending() is None


def test_a_command_that_needs_the_button_is_not_approved_by_a_spoken_yes_and_stays_pending(
        tts, heard, home, popen_log):
    s = shell_asked(PIPE)
    pending = s.resp.pending()
    assert not pending.voice and pending.question == shell.ASK_CLICK
    heard.text = "Yes, go ahead!"
    turn = Turn()

    drive(s.resp, turn=turn)

    assert popen_log == []
    assert len(s.brain.calls) == 1  # the yes was not a question for the brain either
    spoken = [b["input"] for b in tts.bodies]  # one request per sentence
    assert " ".join(spoken[:-1]) == shell.ASK_CLICK and spoken[-1] == shell.NEEDS_BUTTON
    assert turn.assistant_text == shell.NEEDS_BUTTON
    assert s.resp.pending() is pending and pending.armed_at is not None  # still there, still answerable
    assert tool_events(s.turn) == [("run_shell", "waiting for confirmation")]
    assert turn.tool_events == []


def test_a_click_then_approves_the_command_that_survived_the_yes(tts, heard, home, popen_log):
    s = shell_asked(PIPE)
    heard.text = "yes"
    drive(s.resp)
    pending = s.resp.pending()

    assert s.resp.approve(pending.id) == "the command finished. Output: HI"

    assert popen_log == [["/bin/zsh", "-c", "echo hi | tr a-z A-Z"]]
    events = tool_events(s.turn)
    assert events[0] == ("run_shell", "waiting for confirmation")
    assert events[1][1] == "$ echo hi | tr a-z A-Z\n(the command finished)\nHI"
    assert s.resp.pending() is None and len(s.brain.calls) == 1


@pytest.mark.parametrize("utterance", ["No.", "cancel that"])
def test_a_no_still_cancels_a_command_that_needs_the_button(tts, heard, home, popen_log, utterance):
    s = shell_asked(PIPE)
    heard.text = utterance
    turn = Turn()

    drive(s.resp, turn=turn)

    assert popen_log == [] and turn.assistant_text == "Okay, I won't."
    assert tool_events(turn) == [("run_shell", "cancelled")]
    assert s.resp.pending() is None


def test_anything_else_still_drops_a_command_that_needs_the_button(tts, heard, home, popen_log):
    s = shell_asked(PIPE, answer("It is sunny."))
    heard.text = "What's the weather like?"

    drive(s.resp)

    assert popen_log == [] and s.resp.pending() is None
    assert tool_events(s.turn) == [("run_shell", "waiting for confirmation"), ("run_shell", "not confirmed")]


def test_a_yes_before_the_question_was_fully_spoken_drops_a_command_that_needs_the_button(tts, heard, home):
    s = shelling(answer("Okay."))
    request = Pending("run_shell", PIPE[1], shell.ASK_CLICK, "call_0", [], Turn(), voice=False)
    s.confirmations.ask(request)  # asked, never armed
    heard.text = "yes"

    drive(s.resp)

    assert s.resp.pending() is None and tool_events(request.turn) == [("run_shell", "not confirmed")]


def test_a_click_that_wins_the_race_with_a_yes_leaves_a_question_not_an_approval():
    brain = FakeBrain(answer("Okay, that is done."))
    request = Pending("run_shell", PWD[1], "Shall I run the command pwd?", "call_0", [], Turn(), armed_at=1.0)

    class Raced(Confirmations):
        def current(self):
            return request

        def take(self, pending_id=None):
            return None  # the button took it between the peek and the take

    turn = Turn()
    decision = assistant._decision("Be brief.", [], "yes", Skills([shell.shell_skill()]), brain, Raced(), turn,
                                   lambda: False)

    assert decision.answer == "Okay, that is done."  # an ordinary question: no approval, no event
    assert turn.tool_events == [] and request.turn.tool_events == []


def test_the_pending_request_carries_how_it_may_be_approved(tts, home):
    assert shell_asked(PWD).resp.pending().voice is True
    assert shell_asked(PIPE).resp.pending().voice is False


def test_barge_in_during_a_voice_approved_command_stops_it_and_the_brain_is_not_asked_again(
        tts, heard, home, monkeypatch):
    monkeypatch.setattr(shell, "SAFE_PROGRAMS", frozenset({"sleep"}))  # `sleep 30` is harmless by form here
    s = shell_asked(SLEEP)
    assert s.resp.pending().voice
    heard.text = "yes"
    turn = Turn()
    reply = start_thread(lambda: drive(s.resp, turn=turn), "reply")
    wait_until(lambda: shell._live)  # the command is running

    turn.close()  # the user talked over it

    assert reply.join() == []  # no audio for a reply that was cut off
    wait_until(lambda: not shell._live)
    wait_until(lambda: turn.tool_events)
    assert turn.tool_events[0]["output"].startswith("$ sleep 30\n(the command was stopped)")
    assert len(s.brain.calls) == 1


def test_the_app_stopping_stops_a_voice_approved_command(tts, heard, home, monkeypatch):
    monkeypatch.setattr(shell, "SAFE_PROGRAMS", frozenset({"sleep"}))
    s = shell_asked(SLEEP)
    heard.text = "yes"
    reply = start_thread(lambda: drive(s.resp), "reply")
    wait_until(lambda: shell._live)

    lifecycle.stopping.set()

    reply.join()
    wait_until(lambda: not shell._live)
    assert len(s.brain.calls) == 1


def test_new_conversation_stops_a_command_a_click_started(tts, home):
    s = shell_asked(SLEEP)
    running = start_thread(lambda: s.resp.approve(s.resp.pending().id), "click")
    wait_until(lambda: shell._live)

    s.resp.clear_pending()

    assert running.join() == "the command was stopped"
    assert s.resp._runs == set()
    assert tool_events(s.turn)[-1][1].startswith("$ sleep 30\n(the command was stopped)")


def test_the_app_stopping_stops_a_command_a_click_started(tts, home):
    s = shell_asked(SLEEP)
    running = start_thread(lambda: s.resp.approve(s.resp.pending().id), "click")
    wait_until(lambda: shell._live)

    lifecycle.stopping.set()

    assert running.join() == "the command was stopped"


def test_a_new_conversation_between_the_click_and_the_run_stops_the_run_before_it_starts(tts, home, monkeypatch):
    s = shell_asked(SLEEP)
    pending = s.resp.pending()
    real_take = s.confirmations.take

    def take_then_clear(pending_id=None):
        taken = real_take(pending_id)
        s.resp.clear_pending()  # "New conversation" lands right after the click took the request
        return taken

    monkeypatch.setattr(s.confirmations, "take", take_then_clear)

    assert s.resp.approve(pending.id) == "the command was stopped"


def test_the_default_skills_have_run_shell_only_when_the_setting_is_on(monkeypatch):
    monkeypatch.setattr(assistant.config, "ASSISTANT_SHELL", False)
    assert "run_shell" not in AssistantResponder("http://server").skills.names
    monkeypatch.setattr(assistant.config, "ASSISTANT_SHELL", True)
    assert "run_shell" in AssistantResponder("http://server").skills.names
