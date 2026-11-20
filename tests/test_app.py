import argparse
import importlib
import itertools
import json
import logging
import signal
import sys
from types import SimpleNamespace
from typing import ClassVar

import numpy as np
import pytest
import requests
from fake_realtime import pcm
from fastrtc import AdditionalOutputs
from threads import next_on_fresh_thread, run_on_thread

from vaice import app, config, lifecycle, vision
from vaice import conversation as conv
from vaice.app import CHATBOT_HEIGHT, UI_TITLE, VIEW_REFRESH_SECONDS, build_stream, make_handler, parse_args
from vaice.assistant import Assistant
from vaice.confirm import Pending
from vaice.conversation import Conversation
from vaice.interruptible import InterruptSafeReplyOnPause
from vaice.memory import Memory
from vaice.skills import Skills

AUDIO = (48000, np.zeros((1, 480), dtype=np.int16))


@pytest.fixture(autouse=True)
def short_waits(monkeypatch):
    monkeypatch.setattr(conv, "TRANSCRIPT_WAIT", 0.05)
    monkeypatch.setattr(app, "UI_TRANSCRIPT_WAIT", 0.05)
    # Each clock reading is a minute later, so a reply's audio has always finished playing by the
    # next turn unless a test says otherwise
    monkeypatch.setattr(conv, "_clock", itertools.count(0, 60).__next__)


def button_fn(stream, label):
    """The click function of the button with this label."""
    config = stream.ui.get_config_file()
    (button,) = [c for c in config["components"] if c["type"] == "button" and c["props"]["value"] == label]
    (click,) = [d for d in config["dependencies"] if d["targets"][0] == (button["id"], "click")]
    return stream.ui.fns[click["id"]].fn


def tick_fn(stream):
    """The 1 s timer's function (fastrtc has a timer of its own; ours is the one at our period)."""
    config = stream.ui.get_config_file()
    ours = [c for c in config["components"] if c["type"] == "timer" and c["props"]["value"] == VIEW_REFRESH_SECONDS]
    (tick,) = [d for d in config["dependencies"] if d["targets"][0] == (ours[0]["id"], "tick")]
    return stream.ui.fns[tick["id"]].fn


class StubAssistant:
    """What `app` needs of an `Assistant`: respond plus the outbox, panel and Stop hooks."""

    name = "fake"
    memory = None

    def __init__(self):
        self.calls = []
        self.outbox = None
        self.cleared = 0

    def attach_outbox(self, outbox):
        self.outbox = outbox

    def pending(self):
        return None

    def browsing(self):
        return False

    def approve(self, request_id):
        return None

    def cancel(self, request_id):
        return None

    def clear_pending(self):
        self.cleared += 1

    def stop_task(self):
        return False

    def respond(self, audio, history, turn):
        self.calls.append((list(history), turn))
        turn.set_user_text("what was asked")
        turn.set_assistant_text("an answer")
        yield (24000, np.zeros(1, dtype=np.float32))


def test_each_reply_logs_which_backend_answered(caplog):
    with caplog.at_level(logging.INFO, logger="vaice.app"):
        outputs = list(make_handler(StubAssistant(), Conversation())(AUDIO))

    assert len(outputs) == 2  # the audio chunk, then the transcript update
    assert "[fake] time to first audio" in caplog.text
    assert "[fake] reply finished" in caplog.text


def test_finished_reply_completes_the_turn_and_sends_the_transcript_to_the_ui():
    conversation = Conversation()
    *chunks, update = make_handler(StubAssistant(), conversation)(AUDIO)

    assert len(chunks) == 1
    assert isinstance(update, AdditionalOutputs)
    assert update.args == (
        [{"role": "user", "content": "what was asked"}, {"role": "assistant", "content": "an answer"}],
    )
    assert conversation._turns[0].complete


def test_replies_get_the_earlier_turns_as_history():
    conversation = Conversation()
    responder = StubAssistant()
    handler = make_handler(responder, conversation)
    list(handler(AUDIO))
    list(handler(AUDIO))

    assert responder.calls[0][0] == []
    assert responder.calls[1][0] == [("user", "what was asked"), ("assistant", "an answer")]
    assert responder.calls[1][1] is conversation._turns[1]


def test_an_interrupted_reply_never_completes_its_turn():
    conversation = Conversation()
    gen = make_handler(StubAssistant(), conversation)(AUDIO)
    next(gen)
    gen.close()

    assert not conversation._turns[0].complete


class SentenceAssistant:
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
    responder = SentenceAssistant(replies, ["question one", "question two"])
    conversation = Conversation(tmp_path / "conversation.json")
    return make_handler(responder, conversation), conversation, responder.histories


def test_barge_in_keeps_the_partial_reply_and_a_late_finalization_changes_nothing(tmp_path):
    handler, conversation, histories = barge_in_setup(tmp_path)

    # Turn 1: the reply is cut off partway through its second sentence
    first = handler(AUDIO)
    for i in range(3):  # both chunks of "First one.", then the first chunk of "Second one."
        assert isinstance(next_on_fresh_thread(first, f"fastrtc-{i}"), tuple)

    # The user speaks again: fastrtc starts a new generator and drops the old one without finishing it
    second = handler(AUDIO)
    outputs = []
    while not isinstance(item := next_on_fresh_thread(second, f"fastrtc-{len(outputs) + 10}"), StopIteration):
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
    run_on_thread(first.close, "finalizer")

    assert (conversation.messages_for_ui(), conversation._path.read_text()) == stored
    assert not conversation._turns[0].complete and conversation._turns[0].interrupted
    assert [t.complete for t in conversation._turns] == [False, True]


def test_two_questions_through_the_app_and_the_realtime_session(tmp_path, realtime):
    server = realtime(
        reply=[("text", "It is Paris."), ("audio", pcm(1, 2)), ("audio", pcm(3)), ("done",)],
        transcription=[("transcript", "What is the capital of France?")],
    )
    conversation = Conversation(tmp_path / "conversation.json")
    handler = make_handler(Assistant(server.url, prompt="Be brief."), conversation)

    def ask(n):
        gen = handler(AUDIO)
        outputs = []
        while not isinstance(item := next_on_fresh_thread(gen, f"fastrtc-{n}-{len(outputs)}"), StopIteration):
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
        "<|im_end|>\n<|im_start|>user\nWhat is the capital of France?"
        "<|im_end|>\n<|im_start|>assistant\nIt is Paris."
    )
    assert conversation._turns[1].complete


def test_cli_takes_an_optional_up_command_and_nothing_else(capsys):
    assert vars(parse_args([])) == {"command": None}
    assert vars(parse_args(["up"])) == {"command": "up"}
    with pytest.raises(SystemExit) as exc:
        parse_args(["--unknown"])
    assert exc.value.code == 2
    assert "unrecognized arguments: --unknown" in capsys.readouterr().err
    with pytest.raises(SystemExit) as exc:
        parse_args(["down"])
    assert exc.value.code == 2
    assert "invalid choice: 'down'" in capsys.readouterr().err


def test_ui_has_the_transcript_a_reset_button_and_loads_the_stored_conversation():
    conversation = Conversation()
    stream = build_stream(StubAssistant(), conversation)
    config = stream.ui.get_config_file()

    (chatbot,) = [c for c in config["components"] if c["type"] == "chatbot"]
    (new_conversation,) = [c for c in config["components"] if c["type"] == "button"
                           and c["props"]["value"] == "New conversation"]
    (accordion,) = [c for c in config["components"] if c["type"] == "accordion"]
    assert accordion["props"]["label"] == "Tool activity"
    assert accordion["props"]["open"] is False
    (click,) = [d for d in config["dependencies"] if d["targets"][0] == (new_conversation["id"], "click")]
    assert len(click["outputs"]) == 2 and chatbot["id"] in click["outputs"]  # chatbot + tool panel
    (load_output,) = [tuple(d["outputs"]) for d in config["dependencies"] if d["targets"][0][1] == "load"]
    assert load_output == (chatbot["id"],)  # a page load fills only the chatbot; the timer fills the panel

    conversation.start_turn().set_user_text("hello")
    assert button_fn(stream, "New conversation")() == ([], "")  # the reset clears the chatbot and the tool panel
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
    stream = build_stream(StubAssistant(), Conversation())
    config = stream.ui.get_config_file()

    title = one_component(config["components"], "html", "the heading")
    # two markdown components exist now (the tool panel is the other one); the subtitle is fastrtc's
    subtitles = [c for c in config["components"] if c["type"] == "markdown"
                 and "llama-server" in str(c["props"].get("value", ""))]
    assert len(subtitles) == 1
    assert UI_TITLE in title["props"]["value"]


def test_the_chatbot_is_taller_and_does_not_render_html_tags():
    stream = build_stream(StubAssistant(), Conversation())

    chatbot = one_component(stream.ui.get_config_file()["components"], "chatbot", "the transcript")
    assert chatbot["props"]["height"] == CHATBOT_HEIGHT
    # Gradio 6.0 flips the default to True; the model can put anything in the transcript
    assert chatbot["props"]["allow_tags"] is False


def test_a_late_transcript_reaches_the_chatbot_after_the_reply_ran_out():
    conversation = Conversation()
    stream = build_stream(StubAssistant(), conversation)
    tick = tick_fn(stream)

    view, tools, *_, seen = tick("")  # the State starts blank, so the first tick shows the empty view once
    assert view == [] and tools == ""
    turn = conversation.start_turn()
    view, tools, *_, seen = tick(seen)  # the question was heard; its transcript has not arrived yet
    assert view == [{"role": "user", "content": "…"}]
    turn.set_user_text("and after the reply?")  # the transcript lands late, past the final update
    view, tools, *_, seen = tick(seen)
    assert view[-1] == {"role": "user", "content": "and after the reply?"}  # the stuck "…" is replaced
    assert tick(seen)[0] == {"__type__": "update"}  # unchanged again: silent


def test_a_tool_round_fills_the_collapsed_tool_panel():
    conversation = Conversation()
    stream = build_stream(StubAssistant(), conversation)
    tick = tick_fn(stream)

    *_, seen = tick("")
    turn = conversation.start_turn()
    turn.set_user_text("what is this machine?")
    turn.add_tool_event("get_info", "{}", "os: Darwin")
    turn.set_assistant_text("It is a Mac.")
    turn.finish()

    view, tools, *_, seen = tick(seen)
    assert view[-1] == {"role": "assistant", "content": "It is a Mac."}
    assert "**get_info** `{}`" in tools and "os: Darwin" in tools
    assert tick(seen)[0] == {"__type__": "update"}  # unchanged: silent


def test_what_you_say_after_new_conversation_is_the_next_turn():
    conversation = Conversation()
    stream = build_stream(StubAssistant(), conversation)

    conversation.start_turn().set_user_text("said before the reset")
    button_fn(stream, "New conversation")()

    turn = conversation.start_turn()  # what you say next
    assert conversation.history(before=turn) == []  # the old conversation is gone
    assert conversation.messages_for_ui() == [{"role": "user", "content": "…"}]


def test_the_stream_drops_the_chunk_a_barge_in_leaves_in_flight():
    stream = build_stream(StubAssistant(), Conversation())

    assert isinstance(stream.event_handler, InterruptSafeReplyOnPause)
    assert isinstance(stream.event_handler.copy(), InterruptSafeReplyOnPause)  # fastrtc copies it per connection


def test_the_stream_wires_can_interrupt_from_the_config(monkeypatch):
    stream = build_stream(StubAssistant(), Conversation())
    assert stream.event_handler.can_interrupt is config.CAN_INTERRUPT  # default: on

    monkeypatch.setenv("VAICE_CAN_INTERRUPT", "0")
    importlib.reload(config)
    try:
        stream = build_stream(StubAssistant(), Conversation())
        assert stream.event_handler.can_interrupt is False
    finally:
        monkeypatch.undo()
        importlib.reload(config)


def test_the_reset_button_silences_a_playing_reply(tmp_path):
    stream = build_stream(StubAssistant(), Conversation())
    stopped = []

    class PlayingReply:  # a per-connection handler copy, as fastrtc registers it
        def reset(self):
            stopped.append("reset")

        _clear_queue = staticmethod(lambda: stopped.append("queue"))

    stream.handlers["connection-1"] = PlayingReply()  # type: ignore[assignment]

    button_fn(stream, "New conversation")()

    assert stopped == ["reset", "queue"]


def test_a_failed_reply_is_finished_not_marked_interrupted():
    class Failing(StubAssistant):
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
    class Failing(StubAssistant):
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
    class Failing(StubAssistant):
        def respond(self, audio, history, turn):
            raise RuntimeError("model blew up")

    conversation = Conversation()
    with caplog.at_level(logging.ERROR, logger="vaice.app"):
        list(make_handler(Failing(), conversation)(AUDIO))

    assert "[fake] reply failed" in caplog.text
    assert any(r.exc_info for r in caplog.records)


def test_the_connection_survives_when_the_ui_update_itself_fails(monkeypatch, caplog):
    class Failing(StubAssistant):
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
    class Failing(StubAssistant):
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
        outputs = list(make_handler(StubAssistant(), conversation)(AUDIO))

    (update,) = outputs
    assert isinstance(update, AdditionalOutputs)
    assert update.args == ([{"role": "assistant", "content": "(error: RuntimeError: cannot start a turn)"}],)
    assert "[fake] reply failed" in caplog.text


def test_a_failure_reading_the_history_is_recorded_on_the_turn(monkeypatch, caplog):
    conversation = Conversation()

    def boom(before, tools=False):
        raise RuntimeError("history exploded")

    monkeypatch.setattr(conversation, "history", boom)

    with caplog.at_level(logging.ERROR, logger="vaice.app"):
        outputs = list(make_handler(StubAssistant(), conversation)(AUDIO))

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
        *chunks, update = make_handler(StubAssistant(), conversation)(AUDIO)

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

    class TenSecondReply(StubAssistant):
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
    """Ctrl+C must reach the assistant, and the app must not wait for a stuck model call at exit."""
    previous = signal.getsignal(signal.SIGINT)
    launched = []
    startup = []

    class FakeStream:
        class ui:  # a stand-in for the gradio app

            @staticmethod
            def launch(server_port, pwa, inbrowser):
                launched.append((server_port, pwa, inbrowser))

    monkeypatch.setattr(app, "parse_args", lambda: argparse.Namespace(command=None))
    monkeypatch.setattr(app, "check_server", lambda: startup.append("check"))
    monkeypatch.setattr(app, "Assistant", lambda **_: startup.append("construct") or FakeAssistant())
    monkeypatch.setattr(app, "build_stream", lambda responder, conversation: FakeStream())
    monkeypatch.setattr(app.config, "CONVERSATION_FILE", tmp_path / "conversation.json")

    try:
        app.main()

        assert signal.getsignal(signal.SIGINT) is not previous  # Ctrl+C is handled
        assert startup == ["check", "construct"]
        # pwa on, or the browser 404s on /manifest.json; plain vaice does not open the browser
        assert launched == [(app.config.PORT, True, False)]
        assert lifecycle.stopping.is_set()  # the server is gone: the assistant may give up
    finally:
        signal.signal(signal.SIGINT, previous)
        lifecycle.stopping.clear()


def _fake_stream_class(launched=None):
    class FakeStream:
        class ui:  # a stand-in for the gradio app

            @staticmethod
            def launch(server_port, pwa, inbrowser):
                if launched is not None:
                    launched.append((server_port, pwa, inbrowser))

    return FakeStream


def _main_with(monkeypatch, tmp_path, conversation_file="default", command=None):
    launched = []
    monkeypatch.setattr(app, "parse_args", lambda: argparse.Namespace(command=command))
    monkeypatch.setattr(app, "check_server", lambda: None)
    monkeypatch.setattr(app, "build_stream", lambda responder, conversation: _fake_stream_class(launched)())
    monkeypatch.setattr(app.config, "CONVERSATION_FILE",
                        tmp_path / "conversation.json" if conversation_file == "default" else None)
    previous = signal.getsignal(signal.SIGINT)
    try:
        app.main()
    finally:
        signal.signal(signal.SIGINT, previous)
    return launched


def test_main_reports_a_keyboard_interrupt_and_still_requests_stop(monkeypatch, tmp_path, caplog):
    """Ctrl+C while the server is loading: 'stopped' is logged and the assistant is told to give up."""
    monkeypatch.setattr(app, "parse_args", lambda: argparse.Namespace(command=None))

    def interrupt():
        raise KeyboardInterrupt

    monkeypatch.setattr(app, "check_server", interrupt)
    monkeypatch.setattr(app.config, "CONVERSATION_FILE", tmp_path / "conversation.json")
    previous = signal.getsignal(signal.SIGINT)
    try:
        with caplog.at_level(logging.INFO, logger="vaice.app"):
            app.main()  # must not raise
    finally:
        signal.signal(signal.SIGINT, previous)

    assert "stopped" in caplog.text
    assert lifecycle.stopping.is_set()
    lifecycle.stopping.clear()


def test_main_keeps_the_conversation_in_memory_when_vaice_data_dir_is_empty(monkeypatch, tmp_path, caplog):
    """VAICE_DATA_DIR='' disables persistence: no conversation.json is read or written, and the
    startup log says so."""
    with caplog.at_level(logging.INFO, logger="vaice.app"):
        _main_with(monkeypatch, tmp_path, conversation_file=None)

    assert "in memory only" in caplog.text
    assert not (tmp_path / "conversation.json").exists()


class FakeManagedServer:
    def __init__(self, events):
        self.events = events

    def stop(self):
        self.events.append("stop")


def test_plain_main_never_manages_a_server_and_does_not_open_the_browser(monkeypatch, tmp_path):
    monkeypatch.setattr(app, "ensure_server", lambda: pytest.fail("plain vaice must not start a server"))

    assert _main_with(monkeypatch, tmp_path) == [(app.config.PORT, True, False)]


def test_vaice_up_ensures_the_server_before_the_startup_check_and_opens_the_browser(monkeypatch, tmp_path):
    events = []
    launched = []
    monkeypatch.setattr(app, "ensure_server", lambda: events.append("ensure") or FakeManagedServer(events))
    monkeypatch.setattr(app, "parse_args", lambda: argparse.Namespace(command="up"))
    monkeypatch.setattr(app, "check_server", lambda: events.append("check"))
    monkeypatch.setattr(app, "Assistant", FakeAssistant)
    monkeypatch.setattr(app, "build_stream", lambda responder, conversation: _fake_stream_class(launched)())
    monkeypatch.setattr(app.config, "CONVERSATION_FILE", tmp_path / "conversation.json")
    previous = signal.getsignal(signal.SIGINT)
    try:
        app.main()
    finally:
        signal.signal(signal.SIGINT, previous)

    assert launched == [(app.config.PORT, True, True)]
    assert events == ["ensure", "check", "stop"]  # the child is stopped once the UI is over


def test_vaice_up_stops_the_server_it_started_when_a_later_startup_step_raises(monkeypatch, tmp_path):
    events = []
    monkeypatch.setattr(app, "ensure_server", lambda: FakeManagedServer(events))

    def boom():
        raise RuntimeError("the model cannot speak")

    monkeypatch.setattr(app, "parse_args", lambda: argparse.Namespace(command="up"))
    monkeypatch.setattr(app, "check_server", boom)
    previous = signal.getsignal(signal.SIGINT)
    try:
        with pytest.raises(RuntimeError, match="cannot speak"):
            app.main()
    finally:
        signal.signal(signal.SIGINT, previous)

    assert events == ["stop"]


def test_vaice_up_stops_the_server_on_ctrl_c(monkeypatch, tmp_path):
    events = []
    monkeypatch.setattr(app, "ensure_server", lambda: FakeManagedServer(events))

    def interrupt():
        raise KeyboardInterrupt

    monkeypatch.setattr(app, "parse_args", lambda: argparse.Namespace(command="up"))
    monkeypatch.setattr(app, "check_server", interrupt)
    previous = signal.getsignal(signal.SIGINT)
    try:
        app.main()  # must not raise
    finally:
        signal.signal(signal.SIGINT, previous)

    assert events == ["stop"]


def test_vaice_up_does_not_stop_a_server_it_reused(monkeypatch, tmp_path):
    monkeypatch.setattr(app, "ensure_server", lambda: None)  # a running server: nothing to stop

    assert _main_with(monkeypatch, tmp_path, command="up") == [(app.config.PORT, True, True)]


def test_vaice_up_exits_with_the_servers_message_and_starts_nothing_else(monkeypatch, tmp_path):
    def refuse():
        raise SystemExit("vaice: llama-server was not found")

    monkeypatch.setattr(app, "ensure_server", refuse)
    monkeypatch.setattr(app, "parse_args", lambda: argparse.Namespace(command="up"))
    monkeypatch.setattr(app, "check_server", lambda: pytest.fail("startup went on without a server"))
    previous = signal.getsignal(signal.SIGINT)
    try:
        with pytest.raises(SystemExit, match="not found"):
            app.main()
    finally:
        signal.signal(signal.SIGINT, previous)


# -- startup


class FakeAssistant:
    """What `main()` builds in place of `Assistant`, keeping what it was given."""

    name = "assistant"
    skills = SimpleNamespace(names=("disk_space", "battery_status"))  # the attribute main() reads

    def __init__(self, memory=None, vision=None, extra_skills=(), browse=None):
        self.memory = memory
        self.vision = vision  # the look_at_screen skill main() built, if any
        self.extra_skills = extra_skills  # the MCP skills main() started, if any
        self.browse = browse  # the browser agent main() opened, if any


@pytest.fixture(autouse=True)
def fake_responder_class(monkeypatch):
    """`main()` builds a `FakeAssistant`: no test starts a real responder."""
    monkeypatch.setattr(app, "Assistant", FakeAssistant)


def test_main_builds_the_assistant_responder_and_logs_its_skills(monkeypatch, tmp_path, caplog):
    with caplog.at_level(logging.INFO, logger="vaice.app"):
        launched = _main_with(monkeypatch, tmp_path)

    assert launched  # the UI started
    assert "assistant skills: disk_space, battery_status" in caplog.text


def test_main_warns_when_shell_commands_are_on_and_is_quiet_when_they_are_off(monkeypatch, tmp_path, caplog):
    monkeypatch.setattr(app.config, "SHELL", True)
    with caplog.at_level(logging.WARNING, logger="vaice.app"):
        _main_with(monkeypatch, tmp_path)

    assert "VAICE_SHELL is on" in caplog.text
    assert "Run button" in caplog.text and "vaice's own permissions" in caplog.text

    caplog.clear()
    monkeypatch.setattr(app.config, "SHELL", False)
    with caplog.at_level(logging.WARNING, logger="vaice.app"):
        _main_with(monkeypatch, tmp_path)

    assert "VAICE_SHELL" not in caplog.text


FAKE_KEY = "fake-key-do-not-log-9f3a"


def test_removed_settings_that_are_set_are_warned_about_once_by_name_and_never_by_value(monkeypatch, tmp_path, caplog):
    for name in app.config.REMOVED:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("VAICE_BRAIN_URL", "https://brain.example.net/v1")
    monkeypatch.setenv("VAICE_BRAIN_API_KEY", FAKE_KEY)
    monkeypatch.setenv("VAICE_ASSISTANT_AUTO_SECONDS", "7")
    monkeypatch.setenv("VAICE_ASSISTANT_SHELL", "1")
    with caplog.at_level(logging.DEBUG):
        assert _main_with(monkeypatch, tmp_path)

    (warning,) = [r for r in caplog.records if "no longer used" in r.getMessage()]
    assert warning.levelno == logging.WARNING
    text = warning.getMessage()
    for name in ("VAICE_BRAIN_URL", "VAICE_BRAIN_API_KEY", "VAICE_ASSISTANT_AUTO_SECONDS", "VAICE_ASSISTANT_SHELL"):
        assert name in text
    assert "VAICE_BRAIN_MODEL" not in text  # not set
    assert "helper brain was removed" in text and "use run or click" in text and "renamed to VAICE_SHELL" in text
    assert FAKE_KEY not in caplog.text and "brain.example.net" not in caplog.text


def test_no_warning_about_removed_settings_when_none_is_set(monkeypatch, tmp_path, caplog):
    for name in app.config.REMOVED:
        monkeypatch.delenv(name, raising=False)
    with caplog.at_level(logging.DEBUG):
        _main_with(monkeypatch, tmp_path)

    assert "no longer used" not in caplog.text


VISION_KEY = "fake-vision-key-do-not-log-5c1d"


def _vision(monkeypatch, url, model="LFM3-Omni-2B"):
    monkeypatch.setattr(app.config, "VISION_URL", url)
    monkeypatch.setattr(app.config, "VISION_MODEL", model)
    monkeypatch.setattr(app.config, "VISION_API_KEY", VISION_KEY)


def test_without_a_vision_url_there_is_no_screen_skill_and_nothing_about_vision_is_logged(
        monkeypatch, tmp_path, caplog):
    monkeypatch.setattr(app.config, "VISION_URL", "")
    built = []
    monkeypatch.setattr(app, "Assistant", lambda **kwargs: built.append(kwargs) or FakeAssistant(**kwargs))
    with caplog.at_level(logging.DEBUG):
        assert _main_with(monkeypatch, tmp_path)

    assert built[0]["vision"] is None
    assert "vision (look_at_screen)" not in caplog.text and "VAICE_VISION" not in caplog.text


@pytest.mark.parametrize("url", ["http://localhost:8002/v1", "http://127.0.0.1:8002/v1", "http://[::1]:8002/v1"])
def test_a_vision_model_on_this_machine_is_named_without_a_privacy_warning(monkeypatch, tmp_path, caplog, url):
    _vision(monkeypatch, url)
    built = []
    monkeypatch.setattr(app, "Assistant", lambda **kwargs: built.append(kwargs) or FakeAssistant(**kwargs))
    with caplog.at_level(logging.DEBUG):
        assert _main_with(monkeypatch, tmp_path)

    assert built[0]["vision"].name == "look_at_screen"
    assert "vision (look_at_screen): " in caplog.text and ", model LFM3-Omni-2B" in caplog.text
    assert "not on this machine" not in caplog.text
    assert VISION_KEY not in caplog.text


def test_a_remote_vision_model_gets_a_warning_that_a_screenshot_is_sent(monkeypatch, tmp_path, caplog):
    _vision(monkeypatch, "https://vision.example/v1")
    with caplog.at_level(logging.DEBUG):
        assert _main_with(monkeypatch, tmp_path)

    assert "vision (look_at_screen): vision.example, model LFM3-Omni-2B" in caplog.text
    (warning,) = [r for r in caplog.records if "vision model at" in r.getMessage()]
    assert warning.levelno == logging.WARNING
    text = warning.getMessage()
    assert "vision.example" in text and "not on this machine" in text
    assert "screenshot of your main display is sent to it" in text and "deleted from this Mac" in text
    assert VISION_KEY not in caplog.text and "key" not in caplog.text.lower()  # the key is not even mentioned


def test_a_vision_model_without_a_name_warns_but_the_app_starts(monkeypatch, tmp_path, caplog):
    _vision(monkeypatch, "http://localhost:8002/v1", model="")
    with caplog.at_level(logging.INFO):
        assert _main_with(monkeypatch, tmp_path)

    assert "VAICE_VISION_MODEL is empty" in caplog.text and "model (none set)" in caplog.text


def test_the_vision_skill_is_built_over_the_configured_endpoint_with_the_key_and_the_timeout(monkeypatch):
    _vision(monkeypatch, "https://vision.example/v1")
    monkeypatch.setattr(app.config, "VISION_TIMEOUT", 12.5)
    monkeypatch.setattr(app.config, "VISION_MAX_TOKENS", 150)
    sent = []

    def post(url, json=None, headers=None, timeout=None):
        sent.append((url, json, headers, timeout))
        raise requests.ConnectionError("no network in tests")

    monkeypatch.setattr("vaice.vision.requests.post", post)
    skill = app._vision_skill()

    assert skill.name == "look_at_screen"
    monkeypatch.setattr(vision, "capture", lambda timeout: b"jpeg")
    skill_result = Skills([skill]).run("look_at_screen", '{"question": "What is open?"}')

    ((url, body, headers, timeout),) = sent
    assert (url, timeout, body["model"]) == ("https://vision.example/v1/chat/completions", 12.5, "LFM3-Omni-2B")
    assert body["max_tokens"] == 150  # an unbounded answer ran into the timeout on a busy screen
    assert headers == {"Authorization": f"Bearer {VISION_KEY}"}
    assert VISION_KEY not in skill_result.speech and VISION_KEY not in skill_result.detail


class FakeMcp:
    """What `mcp_skills.start` returns: the started servers' skills, what each offers, and a `close()`."""

    def __init__(self, skills=("a skill",), summaries=(("notes", ["search_notes", "add_note"]),)):
        self.skills, self.summaries, self.closed = list(skills), list(summaries), 0

    def close(self):
        self.closed += 1


def _mcp(monkeypatch, tmp_path, mode="assistant", fake=None):
    """`main()` with a config file set; returns (the fake that start() gave back, what start() was called with)."""
    fake = fake or FakeMcp()
    started = []
    monkeypatch.setattr(app.config, "MCP_CONFIG", tmp_path / "mcp.json")
    monkeypatch.setattr(app.config, "MCP_TIMEOUT", 12.5)

    def start(path, timeout):
        started.append((path, timeout))
        return fake

    monkeypatch.setattr(app.mcp_skills, "start", start)
    return fake, started


def test_without_an_mcp_config_nothing_is_started_and_nothing_about_it_is_logged(monkeypatch, tmp_path, caplog):
    monkeypatch.setattr(app.config, "MCP_CONFIG", None)
    monkeypatch.setattr(app.mcp_skills, "start", lambda *_: pytest.fail("MCP started without a config"))
    built = []
    monkeypatch.setattr(app, "Assistant", lambda **kwargs: built.append(kwargs) or FakeAssistant(**kwargs))
    with caplog.at_level(logging.DEBUG):
        assert _main_with(monkeypatch, tmp_path)

    assert built[0]["extra_skills"] == () and "mcp server" not in caplog.text and "VAICE_MCP" not in caplog.text


def test_assistant_mode_starts_the_mcp_servers_gives_their_skills_to_the_responder_and_logs_what_each_offers(
        monkeypatch, tmp_path, caplog):
    _, started = _mcp(monkeypatch, tmp_path, fake=FakeMcp(summaries=[("notes", ["search_notes", "add_note"]),
                                                                        ("idle", [])]))
    built = []
    monkeypatch.setattr(app, "Assistant", lambda **kwargs: built.append(kwargs) or FakeAssistant(**kwargs))
    with caplog.at_level(logging.INFO, logger="vaice.app"):
        assert _main_with(monkeypatch, tmp_path)

    assert started == [(tmp_path / "mcp.json", 12.5)]
    assert built[0]["extra_skills"] == ["a skill"]
    assert "mcp server notes: 2 tools offered (search_notes, add_note)" in caplog.text
    assert "mcp server idle: 0 tools offered (none)" in caplog.text


def test_the_mcp_servers_are_closed_when_the_app_ends_and_when_it_fails_to_start(monkeypatch, tmp_path):
    fake, _ = _mcp(monkeypatch, tmp_path)
    monkeypatch.setattr(app, "Assistant", FakeAssistant)
    assert _main_with(monkeypatch, tmp_path)
    assert fake.closed == 1

    def boom(**_):
        raise SystemExit("the responder could not be built")

    monkeypatch.setattr(app, "Assistant", boom)
    with pytest.raises(SystemExit, match="could not be built"):
        _main_with(monkeypatch, tmp_path)
    assert fake.closed == 2  # started before the failure: still ended


def test_a_missing_mcp_package_stops_assistant_mode_with_the_install_hint(monkeypatch, tmp_path):
    monkeypatch.setattr(app.config, "MCP_CONFIG", tmp_path / "mcp.json")
    monkeypatch.setitem(sys.modules, "mcp", None)

    with pytest.raises(SystemExit, match="uv sync --extra mcp"):
        _main_with(monkeypatch, tmp_path)


# --- tool history reaches only the assistant when it asks for it


# --- the confirmation panel


class ConfirmingAssistant(StubAssistant):
    """A responder that asks for confirmation, as Assistant does."""

    def __init__(self):
        super().__init__()
        self.waiting = None
        self.approved, self.cancelled, self.cleared = [], [], 0

    def pending(self):
        return self.waiting

    def approve(self, pending_id):
        self.approved.append(pending_id)

    def cancel(self, pending_id):
        self.cancelled.append(pending_id)

    def clear_pending(self):
        self.cleared += 1
        self.waiting = None


def waiting_request(arguments='{"name": "Safari"}'):
    return Pending("quit_app", arguments, "Shall I quit Safari?", conv.Turn())


def confirm_page(responder):
    """The stream, its config, the timer's tick function and the buttons' click functions by label."""
    stream = build_stream(responder, Conversation())
    config = stream.ui.get_config_file()
    (timer,) = [c for c in config["components"] if c["type"] == "timer" and c["props"]["value"] == VIEW_REFRESH_SECONDS]
    (tick,) = [d for d in config["dependencies"] if d["targets"][0] == (timer["id"], "tick")]
    clicks = {}
    for button in [c for c in config["components"] if c["type"] == "button"]:
        (click,) = [d for d in config["dependencies"] if d["targets"][0] == (button["id"], "click")]
        clicks[button["props"]["value"]] = (stream.ui.fns[click["id"]].fn, click)
    return SimpleNamespace(stream=stream, config=config, tick=stream.ui.fns[tick["id"]].fn, tick_dep=tick,
                           clicks=clicks)


HIDDEN = {"visible": False, "__type__": "update"}
SHOWN = {"visible": True, "__type__": "update"}


def test_the_page_has_a_hidden_panel_with_run_and_cancel_above_the_tool_activity():
    responder = ConfirmingAssistant()
    page = confirm_page(responder)
    components = page.config["components"]

    assert set(page.clicks) == {"New conversation", "Run", "Cancel", "Stop the browser task"}
    (panel,) = [c for c in components if c["type"] == "column" and c["props"].get("visible") is False]
    (accordion,) = [c for c in components if c["type"] == "accordion"]
    order = [c["id"] for c in components]
    assert order.index(panel["id"]) < order.index(accordion["id"])
    hidden, text, pending_id, run_row, browse_row, _ = page.tick("")[2:]
    assert hidden == run_row == browse_row == HIDDEN and text == "" and pending_id == ""


def test_the_panel_shows_the_question_and_exactly_what_will_run_and_is_drawn_once():
    responder = ConfirmingAssistant()
    page = confirm_page(responder)
    *_, seen = page.tick("")  # the first tick draws the hidden panel
    request = waiting_request()
    responder.waiting = request

    *_, shown, text, pending_id, run_row, _browse_row, seen = page.tick(seen)

    assert shown == SHOWN and run_row == SHOWN  # Run and Cancel
    assert text == 'Shall I quit Safari?\n\n```\nquit_app {"name": "Safari"}\n```'
    assert pending_id == request.id
    assert all(item == {"__type__": "update"} for item in page.tick(seen)[:-1])  # unchanged: nothing re-renders

    responder.waiting = None  # settled somewhere else: the panel goes away
    *_, hidden, text, pending_id, run_row, _browse_row, seen = page.tick(seen)
    assert hidden == run_row == HIDDEN and pending_id == ""


def shell_request(command, question="I would run a command for this request. The command is on the screen."):
    return Pending("run_shell", json.dumps({"command": command}), question, conv.Turn())


def test_a_shell_request_shows_the_command_line_itself_not_the_json():
    view = app.pending_view(shell_request("ls -la ~/Downloads"))

    text = app.confirmation_panel(view)[1]

    assert text == ("I would run a command for this request. The command is on the screen.\n\n"
                    "```\nls -la ~/Downloads\n```")


def test_a_shell_command_is_shown_exactly_and_backticks_cannot_close_its_block():
    text = app.confirmation_panel(app.pending_view(shell_request("echo ```x```\nls")))[1]

    assert text.endswith("````\necho ```x```\nls\n````")  # a newline in the command is shown as it is


@pytest.mark.parametrize("arguments", ["{not json", "[]", '{"command": 5}', "{}"])
def test_a_shell_request_with_odd_arguments_falls_back_to_the_raw_call(arguments):
    request = Pending("run_shell", arguments, "Q?", conv.Turn())

    text = app.confirmation_panel(app.pending_view(request))[1]

    assert f"```\nrun_shell {arguments}\n```" in text


def test_an_ordinary_request_shows_the_panel_and_its_run_and_cancel_row():
    for request in (waiting_request(), shell_request("ls"), shell_request("ls | wc")):
        panel, _, _, run_row = app.confirmation_panel(app.pending_view(request))

        assert panel == SHOWN and run_row == SHOWN


def test_a_request_with_no_view_hides_the_panel_and_its_row():
    assert app.confirmation_panel(None) == (HIDDEN, "", "", HIDDEN)


def test_the_page_has_one_row_of_run_and_cancel():
    page = confirm_page(ConfirmingAssistant())
    components = {c["id"]: c for c in page.config["components"]}

    def rows(node):  # the page's layout tree: the labels of the buttons of every row, in order
        children = node.get("children", [])
        found = []
        if components.get(node["id"], {}).get("type") == "row":
            buttons = [components[c["id"]] for c in children if components[c["id"]]["type"] == "button"]
            found = [[b["props"]["value"] for b in buttons]]
        return found + [row for child in children for row in rows(child)]

    labels = [row for row in rows(page.config["layout"]) if row]

    assert [row for row in labels if "Run" in row] == [["Run", "Cancel"]]
    variants = {c["props"]["value"]: c["props"]["variant"] for c in components.values() if c["type"] == "button"}
    assert variants["Run"] == "primary"


def test_arguments_with_backticks_cannot_close_the_code_block():
    text = app.confirmation_panel(app.pending_view(waiting_request('{"name": "a ``` b"}')))[1]

    assert text.splitlines()[2] == "````" and text.splitlines()[-1] == "````"


def test_run_and_cancel_pass_the_id_the_panel_showed_and_hide_it_and_redraw_on_the_next_tick():
    responder = ConfirmingAssistant()
    page = confirm_page(responder)
    run, run_dep = page.clicks["Run"]
    cancel, _ = page.clicks["Cancel"]
    state_ids = {c["id"] for c in page.config["components"] if c["type"] == "state"}

    assert run("id-from-the-panel") == ({"visible": False, "__type__": "update"}, "", "")
    cancel("another-id")

    assert responder.approved == ["id-from-the-panel"] and responder.cancelled == ["another-id"]
    assert set(run_dep["outputs"]) - state_ids != set()  # the panel itself
    assert len(run_dep["outputs"]) == 3  # the panel, the id the panel holds, and `seen`: the blank forces a redraw


def test_new_conversation_clears_a_waiting_request():
    responder = ConfirmingAssistant()
    responder.waiting = waiting_request()
    page = confirm_page(responder)

    page.clicks["New conversation"][0]()

    assert responder.cleared == 1 and responder.pending() is None


# -- memory: the Memory panel and the startup line


class MemoryAssistant(StubAssistant):
    """What `build_stream` finds on a responder with memory on: a `memory` holding the notes."""

    def __init__(self, memory):
        super().__init__()
        self.memory = memory


def test_the_memory_panel_is_a_collapsed_accordion_under_the_tool_panel_when_memory_is_on():
    stream = build_stream(MemoryAssistant(Memory(None)), Conversation())
    config = stream.ui.get_config_file()

    labels = [c["props"]["label"] for c in config["components"] if c["type"] == "accordion"]
    assert labels == ["Tool activity", "Memory"]  # in the order they appear
    memory_accordion = [c for c in config["components"] if c["type"] == "accordion"][1]
    assert memory_accordion["props"]["open"] is False


def test_without_memory_there_is_no_memory_panel():
    stream = build_stream(StubAssistant(), Conversation())  # VAICE_MEMORY off: `memory` is None
    config = stream.ui.get_config_file()

    assert [c["props"]["label"] for c in config["components"] if c["type"] == "accordion"] == ["Tool activity"]
    assert len(tick_fn(stream)("")) == 8  # chatbot, tool panel, confirm x4, browse row, seen: no memory column


def test_the_timer_fills_the_memory_panel_when_the_notes_change_and_is_silent_otherwise():
    memory = Memory(None)
    stream = build_stream(MemoryAssistant(memory), Conversation())
    tick = tick_fn(stream)
    assert len(tick("")) == 9  # chatbot, tool panel, confirm x4, browse row, memory panel, seen

    *_, shown, seen = tick("")
    assert shown == "Nothing remembered."
    assert tick(seen)[0] == {"__type__": "update"}  # unchanged: nothing re-rendered

    memory.add("The user parks on level three.")
    *_, shown, seen = tick(seen)
    assert shown == "- The user parks on level three\\."
    assert tick(seen)[0] == {"__type__": "update"}

    memory.clear()
    assert tick(seen)[-2] == "Nothing remembered."


def test_new_conversation_leaves_the_notes_alone():
    memory = Memory(None)
    memory.add("The user parks on level three.")
    conversation = Conversation()
    stream = build_stream(MemoryAssistant(memory), conversation)
    config = stream.ui.get_config_file()
    conversation.start_turn().set_user_text("hello")

    assert button_fn(stream, "New conversation")() == ([], "")  # still just the chatbot and the tool panel

    assert conversation.messages_for_ui() == []
    assert memory.notes() == ["The user parks on level three."]  # long-term by design
    (clear,) = [d for d in config["dependencies"] if d["targets"][0][1] == "clear"]
    stream.ui.fns[clear["id"]].fn()  # the chatbot's trash icon
    assert memory.notes() == ["The user parks on level three."]


HOSTILE_NOTES = [
    "The user likes [a link](http://evil.example/steal) a lot.",
    "<img src=x onerror=alert(1)> and <script>alert(1)</script> and <b>bold</b>",
    "**strong** and _emphasis_ and `code` and ~~strike~~",
    "![pixel](http://evil.example/p.png)",
    "bare http://evil.example/x and www.evil.example and <http://evil.example/auto>",
    "# heading and > quote and 1. list and - item and | table | row",
    "back\\slash and \\[escaped\\] and trailing \\",
    "entity &lt;b&gt; &amp; &#60;i&#62;",
]


def test_the_memory_panel_shows_notes_as_plain_text_without_links_html_or_emphasis():
    from vaice.app import memory_markdown

    markdown_it = pytest.importorskip("markdown_it")
    renderer = markdown_it.MarkdownIt("commonmark", {"html": True, "linkify": False})

    text = memory_markdown(HOSTILE_NOTES)

    html = renderer.render(text)
    for tag in ("<a ", "<img", "<script", "<b>", "<strong", "<em>", "<code", "<del", "<h1", "<blockquote", "<table",
                "<ol"):
        assert tag not in html, (tag, html)
    assert html.count("<li>") == len(HOSTILE_NOTES) and "<ul>" in html
    for fragment in ("evil.example", "alert(1)", "strong", "emphasis"):  # the words themselves are all still shown
        assert fragment in html.replace("&amp;", "&")


def test_the_memory_panel_text_round_trips_and_lists_an_empty_memory_plainly():
    import re

    from vaice.app import memory_markdown

    assert memory_markdown([]) == "Nothing remembered."
    lines = memory_markdown(HOSTILE_NOTES).split("\n")
    assert len(lines) == len(HOSTILE_NOTES) and all(line.startswith("- ") for line in lines)
    assert [re.sub(r"\\(.)", r"\1", line[2:]) for line in lines] == HOSTILE_NOTES  # escaping only adds backslashes


def _memory_main(monkeypatch, tmp_path, notes=None, memory_file="default", on=True):
    """main() with the real `_open_memory`; returns the `memory` the responder was given."""
    given = []

    class Built(FakeAssistant):
        def __init__(self, memory=None, vision=None, extra_skills=(), browse=None):
            super().__init__(memory, vision)
            given.append(memory)

    path = tmp_path / "memory.json" if memory_file == "default" else memory_file
    if notes is not None and path is not None:
        path.write_text(json.dumps({"version": 1, "notes": notes}))
    monkeypatch.setattr(app.config, "MEMORY", on)
    monkeypatch.setattr(app.config, "MEMORY_FILE", path)
    monkeypatch.setattr(app, "Assistant", Built)
    assert _main_with(monkeypatch, tmp_path)
    return given[0]


def test_startup_says_how_many_notes_were_loaded_and_from_where_but_never_what_they_say(monkeypatch, tmp_path, caplog):
    secret = "The user's locker code is 4417."
    with caplog.at_level(logging.DEBUG):
        memory = _memory_main(monkeypatch, tmp_path, notes=[secret, "The user parks on level three."])

    assert memory.notes()[0] == secret
    assert f"assistant memory: 2 notes loaded from {tmp_path / 'memory.json'}" in caplog.text
    assert "4417" not in caplog.text and "level three" not in caplog.text


def test_startup_with_no_file_loads_zero_notes(monkeypatch, tmp_path, caplog):
    with caplog.at_level(logging.INFO, logger="vaice.app"):
        _memory_main(monkeypatch, tmp_path)

    assert "assistant memory: 0 notes loaded from" in caplog.text


def test_with_no_data_dir_the_notes_stay_in_memory_and_startup_says_so(monkeypatch, tmp_path, caplog):
    with caplog.at_level(logging.INFO, logger="vaice.app"):
        memory = _memory_main(monkeypatch, tmp_path, memory_file=None)

    assert memory.path is None
    assert "notes the assistant keeps stay in memory only" in caplog.text


def test_with_memory_off_the_responder_gets_no_memory_the_file_is_left_alone_and_startup_says_so(
        monkeypatch, tmp_path, caplog):
    path = tmp_path / "memory.json"
    path.write_text("not even json")  # an off switch must not read it, let alone replace it
    with caplog.at_level(logging.INFO, logger="vaice.app"):
        memory = _memory_main(monkeypatch, tmp_path, memory_file=path, on=False)

    assert memory is None
    assert path.read_text() == "not even json"
    assert "assistant memory is off" in caplog.text and "VAICE_MEMORY" in caplog.text


def test_the_responder_gets_the_outbox_the_handler_serves():
    responder = StubAssistant()

    stream = build_stream(responder, Conversation())

    assert responder.outbox is not None
    assert stream.event_handler.outbox is responder.outbox
    assert stream.event_handler.copy().outbox is responder.outbox  # every connection's copy drains the same outbox


# -- the browse skill (VAICE_BROWSE)

BROWSE_KEY = "sk-fake-browse-key-456"


class FakePiAgent:
    closed = 0
    made: ClassVar[list] = []

    def __init__(self, pi_home, **kwargs):
        self.pi_home, self.kwargs = pi_home, kwargs
        type(self).made.append(self)

    def close(self):
        type(self).closed += 1


@pytest.fixture
def browse_on(monkeypatch, tmp_path):
    """VAICE_BROWSE=pi with an endpoint of its own, a `pi` that is found and prints 1.5.2, and
    PiAgent faked."""
    FakePiAgent.made, FakePiAgent.closed = [], 0
    monkeypatch.setattr(app.config, "BROWSE", "pi")
    monkeypatch.setattr(app.config, "BROWSE_URL", "https://llm.example.com/v1")
    monkeypatch.setattr(app.config, "BROWSE_MODEL", "some-model")
    monkeypatch.setattr(app.config, "BROWSE_API_KEY", BROWSE_KEY)
    monkeypatch.setattr(app.config, "BROWSE_HEADLESS", False)
    monkeypatch.setattr(app.config, "BROWSE_CONFIRM", "ask")
    monkeypatch.setattr(app.shutil, "which", lambda name: "/usr/bin/pi")
    monkeypatch.setattr(app, "pi_version", lambda: (1, 5, 2))
    monkeypatch.setattr(app, "PiAgent", FakePiAgent)
    monkeypatch.setattr(app.config, "DATA_DIR", tmp_path / "data")


def test_browse_is_off_by_default_and_builds_nothing(monkeypatch):
    monkeypatch.setattr(app.config, "BROWSE", "off")
    monkeypatch.setattr(app.shutil, "which", lambda name: pytest.fail("looked for pi"))

    assert app._open_browse() is None


@pytest.mark.parametrize(("setting", "value", "warning"), [
    (app.config, ("BROWSE_URL", ""), "needs VAICE_BROWSE_URL"),
    (app.shutil, ("which", lambda name: None), "not on PATH"),
    (app, ("pi_version", lambda: (1, 5, 1)), "is version 1.5.1, browse needs 1.5.2")])
def test_browse_needs_its_endpoint_the_pi_command_and_a_recent_enough_version(browse_on, monkeypatch, caplog, setting,
                                                                         value, warning):
    monkeypatch.setattr(setting, *value)

    with caplog.at_level(logging.WARNING, logger="vaice.app"):
        assert app._open_browse() is None

    assert warning in caplog.text and "browse is not offered" in caplog.text
    assert FakePiAgent.made == []


def test_an_unreadable_version_warns_and_goes_on(browse_on, monkeypatch, caplog):
    monkeypatch.setattr(app, "pi_version", lambda: None)

    with caplog.at_level(logging.WARNING, logger="vaice.app"):
        agent = app._open_browse()

    assert isinstance(agent, FakePiAgent) and "could not read the version" in caplog.text


def test_browse_writes_pi_home_under_the_data_dir_and_gives_the_key_to_the_child_only(browse_on, tmp_path, caplog):
    with caplog.at_level(logging.INFO, logger="vaice.app"):
        agent = app._open_browse()

    base = tmp_path / "data" / "browse"
    assert agent.pi_home == base / "pi" and agent.kwargs["log_path"] == base / "pi.log"
    assert agent.kwargs["env"] == {"VAICE_PI_API_KEY": BROWSE_KEY}  # the child's environment, nowhere else
    assert agent.kwargs["chrome_profile"] == base / "chrome-profile"
    assert agent.kwargs["ask_timeout_s"] == app.CONFIRM_SECONDS + app.BROWSE_ASK_EXTRA_SECONDS
    mcp = json.loads((base / "pi" / "mcp.json").read_text())["mcpServers"]["chrome"]["args"]
    assert f"--userDataDir={base / 'chrome-profile'}" in mcp and "--headless" not in mcp
    models = json.loads((base / "pi" / "models.json").read_text())["providers"]["vaice"]
    assert models["baseUrl"] == "https://llm.example.com/v1" and models["apiKey"] == "$VAICE_PI_API_KEY"
    for path in (base / "pi").rglob("*"):  # the key is in no file and no log line
        assert not path.is_file() or BROWSE_KEY not in path.read_text()
    assert BROWSE_KEY not in caplog.text and "llm.example.com" in caplog.text and "some-model" in caplog.text
    assert models["models"][0]["id"] == "some-model"


def test_an_empty_browse_model_warns_naming_the_browse_variable(browse_on, monkeypatch, caplog):
    monkeypatch.setattr(app.config, "BROWSE_MODEL", "")

    with caplog.at_level(logging.WARNING, logger="vaice.app"):
        assert isinstance(app._open_browse(), FakePiAgent)

    assert "VAICE_BROWSE_MODEL is empty" in caplog.text


def test_browse_asks_for_page_changing_steps_unless_the_user_turned_that_off(browse_on, monkeypatch, caplog):
    assert app._open_browse().kwargs["gate_policy"] == app.GATE_POLICY

    monkeypatch.setattr(app.config, "BROWSE_CONFIRM", "off")
    with caplog.at_level(logging.WARNING, logger="vaice.app"):
        agent = app._open_browse()

    assert agent.kwargs["gate_policy"] == app.UNGATED_POLICY
    assert "without asking" in caplog.text


def test_headless_adds_the_flag_and_no_key_means_no_environment(browse_on, monkeypatch, tmp_path):
    monkeypatch.setattr(app.config, "BROWSE_HEADLESS", True)
    monkeypatch.setattr(app.config, "BROWSE_API_KEY", "")

    agent = app._open_browse()

    assert agent.kwargs["env"] is None
    mcp = json.loads((tmp_path / "data" / "browse" / "pi" / "mcp.json").read_text())["mcpServers"]["chrome"]["args"]
    assert "--headless" in mcp


def test_without_a_data_dir_the_browser_files_go_to_a_temp_dir(browse_on, monkeypatch, tmp_path):
    monkeypatch.setattr(app.config, "DATA_DIR", None)
    monkeypatch.setattr(app.tempfile, "mkdtemp", lambda prefix: str(tmp_path / "temp"))

    agent = app._open_browse()

    assert agent.pi_home == tmp_path / "temp" / "browse" / "pi" and agent.pi_home.is_dir()


def test_a_failed_write_of_the_pi_config_is_a_warning_and_browse_is_not_offered(browse_on, monkeypatch, caplog):
    def refuse(*args, **kwargs):
        raise OSError("read-only")

    monkeypatch.setattr(app, "write_pi_home", refuse)
    with caplog.at_level(logging.WARNING, logger="vaice.app"):
        assert app._open_browse() is None

    assert "could not write the browser agent's configuration" in caplog.text


def test_assistant_main_hands_the_agent_to_the_responder_and_closes_it_at_the_end(browse_on, monkeypatch, tmp_path):
    seen = []
    original = FakeAssistant.__init__

    def init(self, *args, **kwargs):
        original(self, *args, **kwargs)
        seen.append(self.browse)

    monkeypatch.setattr(FakeAssistant, "__init__", init)

    assert _main_with(monkeypatch, tmp_path)

    assert seen == [FakePiAgent.made[0]] and FakePiAgent.closed == 1


def test_the_agent_is_closed_even_when_startup_fails_after_it(browse_on, monkeypatch, tmp_path):
    monkeypatch.setattr(FakeAssistant, "__init__", lambda self, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")))

    with pytest.raises(RuntimeError, match="boom"):
        _main_with(monkeypatch, tmp_path)

    assert FakePiAgent.closed == 1


def test_the_stop_row_follows_whether_a_browser_task_runs():
    conversation = Conversation(None, window=10)

    def view(known, running):
        return app.refresh_view(conversation, known, pending=lambda: None, browsing=lambda: running)

    off, on = view("", False), view("", True)

    assert off[-2]["visible"] is False and on[-2]["visible"] is True  # the row after the confirmation panel
    assert off[-1] != on[-1]  # the view is part of the dedup key: a change is redrawn
    assert view(on[-1], True)[-2] == app.gr.skip()


class BrowsingAssistant(ConfirmingAssistant):
    """A responder with a browser task (the Stop button's two methods), as Assistant has."""

    def __init__(self):
        super().__init__()
        self.running, self.stopped = False, 0

    def browsing(self):
        return self.running

    def stop_task(self):
        self.stopped += 1
        self.running = False


def test_a_browser_task_shows_a_stop_button_that_ends_it_and_hides_at_once():
    responder = BrowsingAssistant()
    page = confirm_page(responder)
    components = {c["id"]: c for c in page.config["components"]}
    stop_fn, stop_click = page.clicks["Stop the browser task"]

    assert components[stop_click["targets"][0][0]]["props"]["variant"] == "stop"
    first = page.tick("")
    assert first[-2] == HIDDEN  # [..., confirm rows, browse row, seen]: nothing runs yet

    responder.running = True
    shown = page.tick(first[-1])
    assert shown[-2] == SHOWN

    assert stop_fn() == (HIDDEN, "") and responder.stopped == 1  # hidden at once, the blank `seen` redraws
