"""tests for the llamacpp backend, against a fake /v1/realtime server (tests/fake_realtime.py)"""
import logging
import socket
import subprocess
import sys
import threading
import time

import numpy as np
import pytest
import requests
from fake_realtime import next_on_fresh_thread, pcm

from vaice import lifecycle
from vaice.conversation import Turn
from vaice.responders import llamacpp
from vaice.responders.llamacpp import MAX_OUTPUT_TOKENS, MODE_INSTRUCTION, LlamaCppResponder, check_server

TONE = (8000 * np.sin(2 * np.pi * 440 * np.arange(4800) / 48000)).astype(np.int16)
AUDIO = (48000, np.stack([TONE, np.zeros_like(TONE)]))  # 0.1 s of stereo browser audio


@pytest.fixture(autouse=True)
def quick_polls(monkeypatch):
    monkeypatch.setattr(llamacpp, "POLL_SECONDS", 0.02)


class CountingTurn(Turn):
    """A Turn that remembers every transcript it was given (the field alone can't show a second write)."""

    def __init__(self):
        super().__init__()
        self.user_texts = []

    def set_user_text(self, text):
        self.user_texts.append(text)
        super().set_user_text(text)


def drive(responder, audio=AUDIO, history=(), turn=None):
    """All the frames of one reply, resuming the generator on a fresh thread each time, like fastrtc.
    Raises the error that ended the reply."""
    gen = responder.respond(audio, history, turn or Turn())
    frames = []
    while True:
        item = next_on_fresh_thread(gen, f"fastrtc-{len(frames)}")
        if isinstance(item, StopIteration):
            return frames
        if isinstance(item, BaseException):
            raise item
        frames.append(item)


def wait_for_threads_to_end(name="vaice-llamacpp", seconds=3.0):
    """Until the reply's session thread (exact name; the transcription thread has its own) has exited."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if not [t for t in threading.enumerate() if t.name == name]:
            return
        time.sleep(0.01)
    raise AssertionError(f"threads still running: {[t.name for t in threading.enumerate()]}")


# -- what goes to the server


def test_the_question_is_sent_as_mono_pcm16_at_the_rate_the_server_announces(realtime):
    server = realtime()
    drive(LlamaCppResponder(server.url, "Be brief."))

    sent = np.frombuffer(server.conversation().audio, dtype="<i2")
    assert len(sent) == 1600  # 4800 samples at 48 kHz -> 16 kHz
    # mono = the mean of both channels (left is the tone, right is silent), at 16 kHz
    expected = 4000 * np.sin(2 * np.pi * 440 * np.arange(1600) / 16000)
    np.testing.assert_allclose(sent[100:-100], expected[100:-100], atol=120)  # the filter's edges are skipped

    server2 = realtime(rates=(8000, 24000))  # the rate comes from session.created, never assumed
    drive(LlamaCppResponder(server2.url, ""))
    assert len(server2.conversation().audio) == 800 * 2


def test_the_session_is_configured_before_any_audio(realtime):
    server = realtime()
    drive(LlamaCppResponder(server.url, "Be brief."))

    conn = server.conversation()
    assert conn.types == ["session.update", "input_audio_buffer.append", "response.create"]
    assert conn.session == {
        "instructions": f"{MODE_INSTRUCTION} Be brief.",
        "output_modalities": ["text", "audio"],
        "max_output_tokens": MAX_OUTPUT_TOKENS,
    }


def test_earlier_turns_go_into_the_instructions_as_text(realtime):
    server = realtime()
    history = [("user", "What is the capital of France?"), ("assistant", "Paris,\nof course.")]
    drive(LlamaCppResponder(server.url, "Be brief."), history=history)

    assert server.conversation().session["instructions"] == (
        f"{MODE_INSTRUCTION} Be brief.\n\nThe conversation so far:\n"
        "User: What is the capital of France?\nAssistant: Paris, of course."
    )


def test_the_mode_line_is_there_even_without_a_prompt(realtime):
    server = realtime()
    drive(LlamaCppResponder(server.url, ""))
    assert server.conversation().session["instructions"] == MODE_INSTRUCTION


def test_the_question_is_also_transcribed_in_a_session_of_its_own(realtime):
    server = realtime()
    turn = CountingTurn()
    drive(LlamaCppResponder(server.url, ""), turn=turn)

    assert turn.transcribed.wait(3)
    transcription = server.transcription_session()
    assert transcription.types == ["session.update", "input_audio_buffer.append", "input_audio_buffer.commit"]
    assert transcription.session == {"type": "transcription"}  # no language: the model detects it
    assert transcription.audio == server.conversation().audio
    assert turn.user_texts == ["hello there"]


# -- what comes back


def test_the_replys_audio_comes_out_as_float32_frames_at_the_servers_output_rate(realtime):
    server = realtime(
        reply=[("text", "Hi."), ("audio", pcm(16384, -16384, 0)), ("audio", pcm(-32768)), ("done",)],
        rates=(16000, 22050),
    )
    frames = drive(LlamaCppResponder(server.url, ""))

    assert [rate for rate, _ in frames] == [22050, 22050]
    assert all(samples.dtype == np.float32 for _, samples in frames)
    np.testing.assert_array_equal(frames[0][1], [0.5, -0.5, 0.0])
    np.testing.assert_array_equal(frames[1][1], [-1.0])


def test_a_finished_reply_is_recorded_whole_and_the_socket_is_closed(realtime):
    server = realtime(reply=[("text", "Hello "), ("text", "there."), ("audio", pcm(1, 2)), ("done",)])
    turn = Turn()
    drive(LlamaCppResponder(server.url, ""), turn=turn)

    assert turn.assistant_text == "Hello there."
    assert server.conversation().closed.wait(3)


def test_text_written_ahead_of_the_audio_is_split_by_the_audio_handed_out(realtime):
    # LFM writes its text about twice as fast as it speaks: all of it can be here before the first
    # word is out. Only the share the audio handed out covers counts.
    server = realtime(
        reply=[("text", "one two "), ("text", "three four"), *[("audio", pcm(1, 2)) for _ in range(4)], ("done",)]
    )
    turn = Turn()
    gen = LlamaCppResponder(server.url, "").respond(AUDIO, [], turn)

    assert isinstance(next_on_fresh_thread(gen), tuple)
    wait_for_threads_to_end("vaice-llamacpp")  # the session thread has taken in the whole reply; the rest is queued
    assert isinstance(next_on_fresh_thread(gen), tuple)
    assert turn.assistant_text == "one two"  # 2 of 4 frames, 4 words
    assert isinstance(next_on_fresh_thread(gen), tuple)
    assert turn.assistant_text == "one two three"
    assert isinstance(next_on_fresh_thread(gen), tuple)
    assert isinstance(next_on_fresh_thread(gen), StopIteration)
    assert turn.assistant_text == "one two three four"


def test_a_reply_without_audio_is_a_warning_and_records_no_text(realtime, caplog):
    server = realtime(reply=[("text", "text only"), ("done",)])
    turn = Turn()
    with caplog.at_level(logging.WARNING, logger="vaice.responders.llamacpp"):
        assert drive(LlamaCppResponder(server.url, ""), turn=turn) == []

    assert turn.assistant_text == ""
    assert "no audio" in caplog.text


def test_a_reply_cut_off_by_the_token_limit_is_logged(realtime, caplog):
    server = realtime(reply=[("audio", pcm(1)), ("done", "completed", MAX_OUTPUT_TOKENS)])
    with caplog.at_level(logging.WARNING, logger="vaice.responders.llamacpp"):
        drive(LlamaCppResponder(server.url, ""))
    assert f"{MAX_OUTPUT_TOKENS}-token limit" in caplog.text


def test_a_normal_reply_logs_no_truncation_warning(realtime, caplog):
    server = realtime(reply=[("audio", pcm(1)), ("done", "completed", MAX_OUTPUT_TOKENS - 1)])
    with caplog.at_level(logging.WARNING, logger="vaice.responders.llamacpp"):
        drive(LlamaCppResponder(server.url, ""))
    assert "limit" not in caplog.text


# -- failures


def test_an_error_event_fails_the_reply_with_the_servers_message(realtime):
    server = realtime(reply=[("audio", pcm(1)), ("error", "context is full")])
    with pytest.raises(RuntimeError, match="llama-server error: context is full"):
        drive(LlamaCppResponder(server.url, ""))


def test_a_server_that_hangs_up_mid_reply_fails_the_reply(realtime):
    server = realtime(reply=[("audio", pcm(1)), ("close",)])
    with pytest.raises(RuntimeError, match="closed the connection"):
        drive(LlamaCppResponder(server.url, ""))


def test_a_reply_that_ends_cancelled_fails(realtime):
    server = realtime(reply=[("audio", pcm(1)), ("done", "cancelled")])
    with pytest.raises(RuntimeError, match="status 'cancelled'"):
        drive(LlamaCppResponder(server.url, ""))


def test_a_silent_server_fails_the_reply_after_the_timeout(realtime):
    server = realtime(reply=[("audio", pcm(1)), ("stall",)])
    started = time.monotonic()
    with pytest.raises(RuntimeError, match=r"sent nothing for 0\.3s"):
        drive(LlamaCppResponder(server.url, "", timeout=0.3))
    assert time.monotonic() - started < 3
    assert server.conversation().closed.wait(3)  # and the socket is not left open


def test_a_server_that_is_not_there_fails_the_reply_and_marks_the_question_inaudible():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]  # nothing listens here once the socket is closed
    turn = CountingTurn()
    with pytest.raises(RuntimeError, match=r"cannot open ws://127\.0\.0\.1"):
        drive(LlamaCppResponder(f"http://127.0.0.1:{port}", ""), turn=turn)
    assert turn.transcribed.wait(3)
    assert turn.user_texts == [""]


def test_a_reply_error_does_not_cost_the_question_its_transcript(realtime):
    server = realtime(reply=[("error", "boom")])
    turn = CountingTurn()
    with pytest.raises(RuntimeError, match="boom"):
        drive(LlamaCppResponder(server.url, ""), turn=turn)
    assert turn.transcribed.wait(3)
    assert turn.user_texts == ["hello there"]


# -- the transcript is reported exactly once


def test_a_failed_transcription_is_logged_and_leaves_an_empty_transcript(realtime, caplog):
    server = realtime(transcription=[("error", "no asr")])
    turn = CountingTurn()
    with caplog.at_level(logging.ERROR, logger="vaice.responders.llamacpp"):
        drive(LlamaCppResponder(server.url, ""), turn=turn)

    assert turn.transcribed.wait(3)
    assert turn.user_texts == [""]
    assert "no asr" in caplog.text


def test_a_transcription_that_never_arrives_times_out_into_an_empty_transcript(realtime):
    server = realtime(transcription=[("stall",)])
    turn = CountingTurn()
    drive(LlamaCppResponder(server.url, "", timeout=0.3), turn=turn)

    assert turn.transcribed.wait(3)
    assert turn.user_texts == [""]


def test_a_transcript_that_lands_after_the_turn_was_interrupted_is_still_reported_once(realtime):
    release = threading.Event()
    server = realtime(
        reply=[("audio", pcm(1)), ("audio", pcm(2)), ("stall",)],
        transcription=[("wait", release), ("transcript", "what is two plus two")],
    )
    turn = CountingTurn()
    gen = LlamaCppResponder(server.url, "").respond(AUDIO, [], turn)
    assert isinstance(next_on_fresh_thread(gen), tuple)
    turn.close()  # the next turn started: barge-in
    assert isinstance(next_on_fresh_thread(gen), StopIteration)
    assert not turn.transcribed.is_set()

    release.set()
    assert turn.transcribed.wait(3)
    assert turn.user_texts == ["what is two plus two"]


# -- stopping early


def test_a_turn_closed_mid_reply_stops_the_generator_and_hangs_up(realtime):
    server = realtime(reply=[("text", "a b c d"), ("audio", pcm(1)), ("audio", pcm(2)), ("stall",)])
    turn = Turn()
    gen = LlamaCppResponder(server.url, "").respond(AUDIO, [], turn)
    assert isinstance(next_on_fresh_thread(gen), tuple)
    assert isinstance(next_on_fresh_thread(gen), tuple)

    turn.close()
    started = time.monotonic()
    assert isinstance(next_on_fresh_thread(gen), StopIteration)
    assert time.monotonic() - started < 1
    assert server.conversation().closed.wait(3)  # the server sees the socket close, which cancels the reply
    wait_for_threads_to_end()


def test_a_generator_dropped_mid_reply_hangs_up_even_when_it_is_finalized_late_on_another_thread(realtime):
    server = realtime(reply=[("audio", pcm(1)), ("audio", pcm(2)), ("stall",)])
    gen = LlamaCppResponder(server.url, "").respond(AUDIO, [], Turn())
    assert isinstance(next_on_fresh_thread(gen), tuple)

    finalizer = threading.Thread(target=gen.close, name="finalizer")
    finalizer.start()
    finalizer.join()

    assert server.conversation().closed.wait(3)
    wait_for_threads_to_end()


def test_the_reply_is_abandoned_when_the_app_is_stopping(realtime, caplog):
    server = realtime(reply=[("stall",)])  # never says anything
    gen = LlamaCppResponder(server.url, "").respond(AUDIO, [], Turn())
    result = []
    worker = threading.Thread(target=lambda: result.append(next_on_fresh_thread(gen)))
    worker.start()
    assert server.conversation().finished.wait(3)

    started = time.monotonic()
    with caplog.at_level(logging.WARNING, logger="vaice.responders.llamacpp"):
        lifecycle.request_stop()
        worker.join(3)
    assert not worker.is_alive()
    assert isinstance(result[0], StopIteration)
    assert time.monotonic() - started < 1
    assert "stopping" in caplog.text
    assert server.conversation().closed.wait(3)


STOP_SCRIPT = '''
import asyncio
import sys

import numpy as np

from vaice import lifecycle
from vaice.conversation import Turn
from vaice.responders.llamacpp import LlamaCppResponder

AUDIO = (48000, np.zeros((1, 480), dtype=np.int16))


def pull(gen):
    """What fastrtc does with the reply generator (reply_on_pause.py: next(self.generator))."""
    try:
        return next(gen)
    except StopIteration:
        return None


async def main():
    gen = LlamaCppResponder(sys.argv[1], "").respond(AUDIO, [], Turn())
    task = asyncio.get_running_loop().run_in_executor(None, pull, gen)
    await asyncio.sleep(1.0)
    lifecycle.request_stop()  # Ctrl+C
    await asyncio.wait_for(task, timeout=10)


asyncio.run(main())
print("exited cleanly")
'''


@pytest.fixture
def silent_listener():
    """A server that accepts TCP connections and never answers the websocket upgrade."""
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        yield f"http://127.0.0.1:{listener.getsockname()[1]}"


@pytest.mark.parametrize("server_kind", ["never_upgrades", "never_answers"])
def test_a_server_that_never_answers_does_not_keep_the_process_alive(
    server_kind, realtime, silent_listener, tmp_path
):
    # never_upgrades: connect() hangs for the whole open timeout (120 s); never_answers: the session is up, then silence
    if server_kind == "never_upgrades":
        url = silent_listener
    else:
        url = realtime(reply=[("stall",)], transcription=[("stall",)]).url
    script = tmp_path / "stop_exit.py"
    script.write_text(STOP_SCRIPT)

    try:
        result = subprocess.run([sys.executable, str(script), url], capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        pytest.fail("the process did not exit: the app would hang at exit")

    assert result.returncode == 0, result.stderr[-3000:]
    assert "exited cleanly" in result.stdout


# -- the harness itself


def test_the_fresh_thread_per_next_harness_catches_state_tied_to_one_thread():
    """The harness must fail a thread-local generator because each next() runs on a new thread."""

    def thread_bound():
        local = threading.local()
        local.items = [1, 2, 3]
        while True:
            yield local.items.pop(0)

    assert [next(g := thread_bound()), next(g)] == [1, 2]  # one thread: fine
    gen = thread_bound()
    assert next_on_fresh_thread(gen) == 1
    assert isinstance(next_on_fresh_thread(gen), AttributeError)


# -- the startup check


class Reply:
    def __init__(self, status=200, body=None):
        self.status_code = status
        self.ok = status < 400
        self._body = body
        self.text = str(body)

    def json(self):
        if self._body is None:
            raise ValueError("not json")
        return self._body

    def raise_for_status(self):
        if not self.ok:
            raise requests.HTTPError(str(self.status_code))


def models(*modalities):
    return {"data": [{"id": "LFM3", "architecture": {"output_modalities": list(modalities)}}]}


def serve_checks(monkeypatch, health, listing=None):
    asked = []

    def get(url, timeout):
        asked.append(url)
        if isinstance(health, Exception):
            raise health
        return health if url.endswith("/health") else listing

    monkeypatch.setattr(llamacpp.requests, "get", get)
    return asked


def test_a_server_with_an_audio_model_passes(monkeypatch):
    asked = serve_checks(monkeypatch, Reply(), Reply(body=models("text", "audio")))
    check_server("http://h:1")
    assert asked[:2] == ["http://h:1/health", "http://h:1/v1/models"]


def test_a_model_that_cannot_speak_is_an_error_naming_it(monkeypatch):
    serve_checks(monkeypatch, Reply(), Reply(body=models("text")))
    with pytest.raises(RuntimeError, match=r"LFM3.*does not output audio.*LFM3-Audio-2B-GGUF"):
        check_server("http://h:1")


def test_a_model_list_without_modalities_counts_as_no_audio(monkeypatch):
    serve_checks(monkeypatch, Reply(), Reply(body={"data": [{"id": "old"}]}))
    with pytest.raises(RuntimeError, match="does not output audio"):
        check_server("http://h:1")


def test_an_unreachable_server_is_a_warning_and_startup_can_continue(monkeypatch, caplog):
    serve_checks(monkeypatch, requests.ConnectionError("refused"))
    with caplog.at_level(logging.WARNING, logger="vaice.responders.llamacpp"):
        assert check_server("http://h:1") is None
    assert "not reachable at http://h:1/health" in caplog.text
    assert "replies will fail until it is up" in caplog.text


def test_a_server_still_loading_is_a_note_and_startup_can_continue(monkeypatch, caplog):
    serve_checks(monkeypatch, Reply(503, "Loading model"))
    with caplog.at_level(logging.INFO, logger="vaice.responders.llamacpp"):
        assert check_server("http://h:1") is None
    assert "still loading" in caplog.text


def test_an_unreadable_model_list_is_a_warning(monkeypatch, caplog):
    serve_checks(monkeypatch, Reply(), Reply(body=None))
    with caplog.at_level(logging.WARNING, logger="vaice.responders.llamacpp"):
        assert check_server("http://h:1") is None
    assert "could not read the model list" in caplog.text


def test_an_unhealthy_server_is_a_warning(monkeypatch, caplog):
    serve_checks(monkeypatch, Reply(500, "boom"))
    with caplog.at_level(logging.WARNING, logger="vaice.responders.llamacpp"):
        assert check_server("http://h:1") is None
    assert "returned 500" in caplog.text


def test_a_secure_server_url_means_a_wss_socket():
    assert llamacpp._ws_url("http://h:1") == "ws://h:1/v1/realtime"
    assert llamacpp._ws_url("https://h:1") == "wss://h:1/v1/realtime"
