import argparse
import itertools
import logging
import signal
import threading

import numpy as np
import pytest
from fake_realtime import pcm
from fastrtc import AdditionalOutputs

from vaice import app, lifecycle
from vaice import conversation as conv
from vaice.app import CHATBOT_HEIGHT, UI_TITLE, VIEW_REFRESH_SECONDS, build_stream, make_handler, parse_args
from vaice.conversation import Conversation
from vaice.interruptible import InterruptSafeReplyOnPause
from vaice.responders.llamacpp import LlamaCppResponder

AUDIO = (48000, np.zeros((1, 480), dtype=np.int16))


@pytest.fixture(autouse=True)
def short_waits(monkeypatch):
    monkeypatch.setattr(conv, "TRANSCRIPT_WAIT", 0.05)
    monkeypatch.setattr(app, "UI_TRANSCRIPT_WAIT", 0.05)
    # Each clock reading is a minute later, so a reply's audio has always finished playing by the
    # next turn unless a test says otherwise
    monkeypatch.setattr(conv, "_clock", itertools.count(0, 60).__next__)


class FakeResponder:
    name = "fake"

    def __init__(self):
        self.calls = []

    def respond(self, audio, history, turn):
        self.calls.append((list(history), turn))
        turn.set_user_text("what was asked")
        turn.set_assistant_text("an answer")
        yield (24000, np.zeros(1, dtype=np.float32))


def test_each_reply_logs_which_backend_answered(caplog):
    with caplog.at_level(logging.INFO, logger="vaice.app"):
        outputs = list(make_handler(FakeResponder(), Conversation())(AUDIO))

    assert len(outputs) == 2  # the audio chunk, then the transcript update
    assert "[fake] time to first audio" in caplog.text
    assert "[fake] reply finished" in caplog.text


def test_finished_reply_completes_the_turn_and_sends_the_transcript_to_the_ui():
    conversation = Conversation()
    *chunks, update = make_handler(FakeResponder(), conversation)(AUDIO)

    assert len(chunks) == 1
    assert isinstance(update, AdditionalOutputs)
    assert update.args == (
        [{"role": "user", "content": "what was asked"}, {"role": "assistant", "content": "an answer"}],
    )
    assert conversation._turns[0].complete


def test_replies_get_the_earlier_turns_as_history():
    conversation = Conversation()
    responder = FakeResponder()
    handler = make_handler(responder, conversation)
    list(handler(AUDIO))
    list(handler(AUDIO))

    assert responder.calls[0][0] == []
    assert responder.calls[1][0] == [("user", "what was asked"), ("assistant", "an answer")]
    assert responder.calls[1][1] is conversation._turns[1]


def test_an_interrupted_reply_never_completes_its_turn():
    conversation = Conversation()
    gen = make_handler(FakeResponder(), conversation)(AUDIO)
    next(gen)
    gen.close()

    assert not conversation._turns[0].complete


def resume(gen, name):
    """next(gen) on a fresh thread, like fastrtc does for every chunk. Returns the value, or StopIteration."""
    box = {}

    def run():
        try:
            box["value"] = next(gen)
        except BaseException as exc:
            box["value"] = exc

    t = threading.Thread(target=run, name=name)
    t.start()
    t.join()
    return box["value"]


def finalize(gen):
    """gen.close() from yet another thread, as CPython does when it finally drops the generator."""
    t = threading.Thread(target=gen.close, name="finalizer")
    t.start()
    t.join()


class SentenceResponder:
    """Speaks its replies a sentence at a time, two chunks each, and records a sentence's text only
    after both chunks went out: a reply cut off mid-sentence keeps the sentences before it."""

    name = "sentences"

    def __init__(self, replies, questions):
        self.histories = []
        self._replies = iter(replies)
        self._questions = iter(questions)

    def respond(self, audio, history, turn):
        self.histories.append(list(history))
        turn.set_user_text(next(self._questions))
        spoken = []
        for sentence in next(self._replies):
            if turn.closed.is_set():
                return
            yield (24000, np.zeros(1, dtype=np.float32))
            yield (24000, np.zeros(1, dtype=np.float32))
            spoken.append(sentence)
            turn.set_assistant_text(" ".join(spoken))


def barge_in_setup(tmp_path):
    replies = [["First one.", "Second one.", "Third one."], ["Fine."]]
    responder = SentenceResponder(replies, ["question one", "question two"])
    conversation = Conversation(tmp_path / "conversation.json")
    return make_handler(responder, conversation), conversation, responder.histories


def test_barge_in_keeps_the_partial_reply_and_a_late_finalization_changes_nothing(tmp_path):
    handler, conversation, histories = barge_in_setup(tmp_path)

    # Turn 1: the reply is cut off partway through its second sentence
    first = handler(AUDIO)
    for i in range(3):  # both chunks of "First one.", then the first chunk of "Second one."
        assert isinstance(resume(first, f"fastrtc-{i}"), tuple)

    # The user speaks again: fastrtc starts a new generator and drops the old one without finishing it
    second = handler(AUDIO)
    outputs = []
    while not isinstance(item := resume(second, f"fastrtc-{len(outputs) + 10}"), StopIteration):
        outputs.append(item)

    # The new reply saw the partial answer, marked as interrupted
    assert histories[1] == [("user", "question one"), ("assistant", "First one. (interrupted)")]
    assert isinstance(outputs[-1], AdditionalOutputs)

    expected = [
        {"role": "user", "content": "question one"},
        {"role": "assistant", "content": "First one. (interrupted)"},
        {"role": "user", "content": "question two"},
        {"role": "assistant", "content": "Fine."},
    ]
    assert conversation.messages_for_ui() == expected  # in order

    # Only now does CPython finalize the dropped generator, on another thread
    stored = (conversation.messages_for_ui(), (conversation._path).read_text())
    finalize(first)

    assert (conversation.messages_for_ui(), conversation._path.read_text()) == stored
    assert not conversation._turns[0].complete and conversation._turns[0].interrupted
    assert [t.complete for t in conversation._turns] == [False, True]


def test_two_questions_through_the_app_and_the_llamacpp_backend(tmp_path, realtime):
    server = realtime(
        reply=[("text", "It is Paris."), ("audio", pcm(1, 2)), ("audio", pcm(3)), ("done",)],
        transcription=[("transcript", "What is the capital of France?")],
    )
    conversation = Conversation(tmp_path / "conversation.json")
    handler = make_handler(LlamaCppResponder(server.url, "Be brief."), conversation)

    def ask(n):
        gen = handler(AUDIO)
        outputs = []
        while not isinstance(item := resume(gen, f"fastrtc-{n}-{len(outputs)}"), StopIteration):
            outputs.append(item)
        return outputs

    first = ask(1)
    assert [type(o) for o in first] == [tuple, tuple, AdditionalOutputs]
    assert conversation._turns[0].transcribed.wait(3)
    assert conversation.messages_for_ui() == [
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "It is Paris."},
    ]

    ask(2)
    second = [c for c in server.connections if c.session and c.session.get("type") != "transcription"][1]
    assert second.session["instructions"].endswith(
        "The conversation so far:\nUser: What is the capital of France?\nAssistant: It is Paris."
    )
    assert conversation._turns[1].complete


def test_cli_takes_no_positional_or_flag_arguments(capsys):
    assert vars(parse_args([])) == {}
    with pytest.raises(SystemExit) as exc:
        parse_args(["--unknown"])
    assert exc.value.code == 2
    assert "unrecognized arguments: --unknown" in capsys.readouterr().err


def test_ui_has_the_transcript_a_reset_button_and_loads_the_stored_conversation():
    conversation = Conversation()
    stream = build_stream(FakeResponder(), conversation)
    config = stream.ui.get_config_file()

    (chatbot,) = [c for c in config["components"] if c["type"] == "chatbot"]
    (button,) = [c for c in config["components"] if c["type"] == "button"]
    assert chatbot["props"]["type"] == "messages"
    assert button["props"]["value"] == "New conversation"
    triggers = {
        (d["targets"][0][1], tuple(d["outputs"]))
        for d in config["dependencies"]
        if d["targets"][0][1] in ("load", "click")
    }
    assert triggers == {("load", (chatbot["id"],)), ("click", (chatbot["id"],))}

    conversation.start_turn().set_user_text("hello")
    (click,) = [d for d in config["dependencies"] if d["targets"][0] == (button["id"], "click")]
    assert stream.ui.fns[click["id"]].fn() == []
    assert conversation.messages_for_ui() == []

    # the chatbot's trash icon clears the stored conversation too, not just the view
    (clear,) = [d for d in config["dependencies"] if d["targets"][0] == (chatbot["id"], "clear")]
    conversation.start_turn().set_user_text("hello again")
    stream.ui.fns[clear["id"]].fn()
    assert conversation.messages_for_ui() == []


def one_component(components, type_name, place):
    found = [c for c in components if c["type"] == type_name]
    assert len(found) == 1, f"expected exactly one {type_name} component for {place}, found {len(found)}"
    return found[0]


def test_the_page_identifies_the_llama_server_service():
    stream = build_stream(FakeResponder(), Conversation())
    config = stream.ui.get_config_file()

    title = one_component(config["components"], "html", "the heading")
    subtitle = one_component(config["components"], "markdown", "the service subtitle")
    assert UI_TITLE in title["props"]["value"]
    assert "llama-server" in subtitle["props"]["value"]


def test_the_chatbot_is_taller_and_does_not_render_html_tags():
    stream = build_stream(FakeResponder(), Conversation())

    chatbot = one_component(stream.ui.get_config_file()["components"], "chatbot", "the transcript")
    assert chatbot["props"]["height"] == CHATBOT_HEIGHT
    # Gradio 6.0 flips the default to True; the model can put anything in the transcript
    assert chatbot["props"]["allow_tags"] is False


def test_a_late_transcript_reaches_the_chatbot_after_the_reply_ran_out():
    conversation = Conversation()
    stream = build_stream(FakeResponder(), conversation)
    config = stream.ui.get_config_file()

    # fastrtc has a timer of its own; ours is the one running at our period
    ours = [c for c in config["components"] if c["type"] == "timer" and c["props"]["value"] == VIEW_REFRESH_SECONDS]
    assert len(ours) == 1
    (tick,) = [d for d in config["dependencies"] if d["targets"][0] == (ours[0]["id"], "tick")]
    tick_fn = stream.ui.fns[tick["id"]].fn

    view, seen = tick_fn("")  # the State starts blank, so the first tick shows the empty view once
    assert view == []
    turn = conversation.start_turn()
    view, seen = tick_fn(seen)  # the question was heard; its transcript has not arrived yet
    assert view == [{"role": "user", "content": "…"}]
    turn.set_user_text("and after the reply?")  # the transcript lands late, past the final update
    view, seen = tick_fn(seen)
    assert view[-1] == {"role": "user", "content": "and after the reply?"}  # the stuck "…" is replaced
    assert tick_fn(seen)[0] == {"__type__": "update"}  # unchanged again: silent


def test_what_you_say_after_new_conversation_is_the_next_turn():
    conversation = Conversation()
    stream = build_stream(FakeResponder(), conversation)
    (click,) = [d for d in stream.ui.get_config_file()["dependencies"] if d["targets"][0][1] == "click"]

    conversation.start_turn().set_user_text("said before the reset")
    stream.ui.fns[click["id"]].fn()  # the button

    turn = conversation.start_turn()  # what you say next
    assert conversation.history(before=turn) == []  # the old conversation is gone
    assert conversation.messages_for_ui() == [{"role": "user", "content": "…"}]


def test_the_stream_drops_the_chunk_a_barge_in_leaves_in_flight():
    stream = build_stream(FakeResponder(), Conversation())

    assert isinstance(stream.event_handler, InterruptSafeReplyOnPause)
    assert isinstance(stream.event_handler.copy(), InterruptSafeReplyOnPause)  # fastrtc copies it per connection


def test_the_reset_button_silences_a_playing_reply(tmp_path):
    stream = build_stream(FakeResponder(), Conversation())
    stopped = []

    class PlayingReply:  # a per-connection handler copy, as fastrtc registers it
        def reset(self):
            stopped.append("reset")

        _clear_queue = staticmethod(lambda: stopped.append("queue"))

    stream.handlers["connection-1"] = PlayingReply()  # type: ignore[assignment]
    (button,) = [d for d in stream.ui.get_config_file()["dependencies"] if d["targets"][0][1] == "click"]

    stream.ui.fns[button["id"]].fn()

    assert stopped == ["reset", "queue"]


def test_a_failed_reply_is_finished_not_marked_interrupted():
    class Failing(FakeResponder):
        def respond(self, audio, history, turn):
            turn.set_user_text("what was asked")
            turn.set_assistant_text("Half an")
            yield (24000, np.zeros(1, dtype=np.float32))
            raise RuntimeError("llama-server fell over")

    conversation = Conversation()
    *chunks, update = make_handler(Failing(), conversation)(AUDIO)  # no raise: the error goes to the UI

    assert len(chunks) == 1
    assert isinstance(update, AdditionalOutputs)
    assert update.args == (
        [
            {"role": "user", "content": "what was asked"},
            {"role": "assistant", "content": "Half an (error: RuntimeError: llama-server fell over)"},
        ],
    )
    assert conversation._turns[0].complete and not conversation._turns[0].interrupted
    assert conversation.history(before=conversation.start_turn()) == [
        ("user", "what was asked"),
        ("assistant", "Half an"),  # the error is for the UI, not the model
    ]


def test_a_failed_reply_reports_a_transcript_the_responder_never_gave():
    class Failing(FakeResponder):
        def respond(self, audio, history, turn):
            raise ValueError("transcription exploded")

    conversation = Conversation()
    list(make_handler(Failing(), conversation)(AUDIO))

    (turn,) = conversation._turns
    assert turn.user_text == ""  # inaudible in history, not a hole the next reply waits on
    assert turn.transcribed.is_set()
    assert turn.error == "ValueError: transcription exploded"
    assert conversation.messages_for_ui() == [
        {"role": "user", "content": "(inaudible)"},
        {"role": "assistant", "content": "(error: ValueError: transcription exploded)"},
    ]


def test_a_failed_reply_is_logged_with_its_traceback(monkeypatch, caplog):
    class Failing(FakeResponder):
        def respond(self, audio, history, turn):
            raise RuntimeError("model blew up")

    conversation = Conversation()
    with caplog.at_level(logging.ERROR, logger="vaice.app"):
        list(make_handler(Failing(), conversation)(AUDIO))

    assert "[fake] reply failed" in caplog.text
    assert any(r.exc_info for r in caplog.records)


def test_the_connection_survives_when_the_ui_update_itself_fails(monkeypatch, caplog):
    class Failing(FakeResponder):
        def respond(self, audio, history, turn):
            raise RuntimeError("llama-server fell over")

    conversation = Conversation()

    def boom(self):
        raise RuntimeError("the view exploded")

    monkeypatch.setattr(Conversation, "messages_for_ui", boom)

    with caplog.at_level(logging.ERROR, logger="vaice.app"):
        yields = list(make_handler(Failing(), conversation)(AUDIO))  # no raise: this would kill the connection

    assert yields == []  # no audio and no view to hand over, but the generator ended normally
    assert "[fake] reply failed" in caplog.text
    assert "could not show the failure in the chatbot" in caplog.text


def test_a_failure_that_cannot_be_recorded_still_shows_the_error_and_keeps_the_connection(monkeypatch, caplog):
    class Failing(FakeResponder):
        def respond(self, audio, history, turn):
            raise RuntimeError("llama-server fell over")

    conversation = Conversation()
    calls = []
    real_save = conversation._save

    def save_that_fails_on_the_second_call():  # start_turn's initial save still works; fail()'s does not
        if calls:
            raise OSError("disk full")
        calls.append(1)
        real_save()

    monkeypatch.setattr(conversation, "_save", save_that_fails_on_the_second_call)

    with caplog.at_level(logging.ERROR, logger="vaice.app"):
        *_, update = make_handler(Failing(), conversation)(AUDIO)  # no raise: the error goes to the UI

    assert isinstance(update, AdditionalOutputs)
    assert update.args == (
        [
            {"role": "user", "content": "(inaudible)"},
            {"role": "assistant", "content": "(error: RuntimeError: llama-server fell over)"},
        ],
    )
    assert "could not record the failure on the turn" in caplog.text


def test_a_failure_before_the_reply_starts_is_reported_in_the_ui_and_does_not_escape(monkeypatch, caplog):
    conversation = Conversation()

    def boom():
        raise RuntimeError("cannot start a turn")

    monkeypatch.setattr(conversation, "start_turn", boom)

    with caplog.at_level(logging.ERROR, logger="vaice.app"):
        outputs = list(make_handler(FakeResponder(), conversation)(AUDIO))

    (update,) = outputs
    assert isinstance(update, AdditionalOutputs)
    assert update.args == ([{"role": "assistant", "content": "(error: RuntimeError: cannot start a turn)"}],)
    assert "[fake] reply failed" in caplog.text


def test_a_failure_reading_the_history_is_recorded_on_the_turn(monkeypatch, caplog):
    conversation = Conversation()

    def boom(before):
        raise RuntimeError("history exploded")

    monkeypatch.setattr(conversation, "history", boom)

    with caplog.at_level(logging.ERROR, logger="vaice.app"):
        outputs = list(make_handler(FakeResponder(), conversation)(AUDIO))

    (update,) = outputs
    assert isinstance(update, AdditionalOutputs)
    assert update.args == (
        [
            {"role": "user", "content": "(inaudible)"},
            {"role": "assistant", "content": "(error: RuntimeError: history exploded)"},
        ],
    )
    assert conversation._turns[0].error == "RuntimeError: history exploded"
    assert conversation._turns[0].complete


def test_a_failure_finishing_the_turn_is_reported_in_the_ui(monkeypatch, caplog):
    def boom(self):
        raise RuntimeError("cannot finish the turn")

    monkeypatch.setattr(conv.Turn, "finish", boom)
    conversation = Conversation()

    with caplog.at_level(logging.ERROR, logger="vaice.app"):
        *chunks, update = make_handler(FakeResponder(), conversation)(AUDIO)

    assert len(chunks) == 1  # the audio that was handed out before the failure
    assert isinstance(update, AdditionalOutputs)
    assert update.args == (
        [
            {"role": "user", "content": "what was asked"},
            {"role": "assistant", "content": "an answer (error: RuntimeError: cannot finish the turn)"},
        ],
    )


def test_a_reply_whose_audio_is_still_playing_when_the_next_turn_starts_is_cut_to_what_played(monkeypatch):
    clock = iter([100.0, 102.5])  # first audio at t=100, the next question at t=102.5

    class TenSecondReply(FakeResponder):
        def respond(self, audio, history, turn):
            turn.set_user_text("tell me a story")
            turn.set_assistant_text("one two three four five six seven eight")
            yield (24000, np.zeros(24000 * 10, dtype=np.float32))  # 10 s, handed out at once like fastrtc pulls it

    monkeypatch.setattr(conv, "_clock", lambda: next(clock))
    conversation = Conversation()
    list(make_handler(TenSecondReply(), conversation)(AUDIO))  # the generator finished: all audio handed out
    second = conversation.start_turn()

    # 2.5 of 10 s played -> 2 of 8 words
    assert conversation.history(before=second) == [("user", "tell me a story"), ("assistant", "one two (interrupted)")]


def test_main_installs_the_stop_handler_and_asks_the_app_to_stop_when_the_server_is_gone(monkeypatch, tmp_path):
    """Ctrl+C must reach the responders, and the app must not wait for a stuck model call at exit."""
    previous = signal.getsignal(signal.SIGINT)
    launched = []
    startup = []

    class FakeStream:
        class ui:  # a stand-in for the gradio app

            @staticmethod
            def launch(server_port, pwa):
                launched.append((server_port, pwa))

    monkeypatch.setattr(app, "parse_args", lambda: argparse.Namespace())
    monkeypatch.setattr(app, "check_server", lambda: startup.append("check"))
    monkeypatch.setattr(app, "LlamaCppResponder", lambda: startup.append("construct") or FakeResponder())
    monkeypatch.setattr(app, "build_stream", lambda responder, conversation: FakeStream())
    monkeypatch.setattr(app.config, "CONVERSATION_FILE", tmp_path / "conversation.json")

    try:
        app.main()

        assert signal.getsignal(signal.SIGINT) is not previous  # Ctrl+C is handled
        assert startup == ["check", "construct"]
        assert launched == [(app.config.PORT, True)]  # pwa on, or the browser 404s on /manifest.json
        assert lifecycle.stopping.is_set()  # the server is gone: the responders may give up
    finally:
        signal.signal(signal.SIGINT, previous)
        lifecycle.stopping.clear()
