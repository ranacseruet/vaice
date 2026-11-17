import argparse
import importlib
import itertools
import json
import logging
import signal
from types import SimpleNamespace

import numpy as np
import pytest
import requests
from fake_realtime import pcm
from fastrtc import AdditionalOutputs
from threads import next_on_fresh_thread, run_on_thread

from vaice import app, config, lifecycle
from vaice import conversation as conv
from vaice.app import CHATBOT_HEIGHT, UI_TITLE, VIEW_REFRESH_SECONDS, build_stream, make_handler, parse_args
from vaice.confirm import Pending
from vaice.conversation import Conversation
from vaice.interruptible import InterruptSafeReplyOnPause
from vaice.memory import Memory
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
    stream = build_stream(FakeResponder(), conversation)
    config = stream.ui.get_config_file()

    (chatbot,) = [c for c in config["components"] if c["type"] == "chatbot"]
    (button,) = [c for c in config["components"] if c["type"] == "button"]
    (accordion,) = [c for c in config["components"] if c["type"] == "accordion"]
    assert accordion["props"]["label"] == "Tool activity"
    assert accordion["props"]["open"] is False
    (click_outputs,) = [tuple(d["outputs"]) for d in config["dependencies"] if d["targets"][0][1] == "click"]
    assert len(click_outputs) == 2 and chatbot["id"] in click_outputs  # chatbot + tool panel
    (load_output,) = [tuple(d["outputs"]) for d in config["dependencies"] if d["targets"][0][1] == "load"]
    assert load_output == (chatbot["id"],)  # a page load fills only the chatbot; the timer fills the panel
    (click,) = [d for d in config["dependencies"] if d["targets"][0] == (button["id"], "click")]

    conversation.start_turn().set_user_text("hello")
    assert stream.ui.fns[click["id"]].fn() == ([], "")  # the reset clears the chatbot and the tool panel
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
    # two markdown components exist now (the tool panel is the other one); the subtitle is fastrtc's
    subtitles = [c for c in config["components"] if c["type"] == "markdown"
                 and "llama-server" in str(c["props"].get("value", ""))]
    assert len(subtitles) == 1
    assert UI_TITLE in title["props"]["value"]


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

    view, tools, seen = tick_fn("")  # the State starts blank, so the first tick shows the empty view once
    assert view == [] and tools == ""
    turn = conversation.start_turn()
    view, tools, seen = tick_fn(seen)  # the question was heard; its transcript has not arrived yet
    assert view == [{"role": "user", "content": "…"}]
    turn.set_user_text("and after the reply?")  # the transcript lands late, past the final update
    view, tools, seen = tick_fn(seen)
    assert view[-1] == {"role": "user", "content": "and after the reply?"}  # the stuck "…" is replaced
    assert tick_fn(seen)[0] == {"__type__": "update"}  # unchanged again: silent


def test_a_tool_round_fills_the_collapsed_tool_panel():
    conversation = Conversation()
    stream = build_stream(FakeResponder(), conversation)
    config = stream.ui.get_config_file()
    ours = [c for c in config["components"] if c["type"] == "timer" and c["props"]["value"] == VIEW_REFRESH_SECONDS]
    (tick,) = [d for d in config["dependencies"] if d["targets"][0] == (ours[0]["id"], "tick")]
    tick_fn = stream.ui.fns[tick["id"]].fn

    _, _, seen = tick_fn("")
    turn = conversation.start_turn()
    turn.set_user_text("what is this machine?")
    turn.add_tool_event("get_info", "{}", "os: Darwin")
    turn.set_assistant_text("It is a Mac.")
    turn.finish()

    view, tools, seen = tick_fn(seen)
    assert view[-1] == {"role": "assistant", "content": "It is a Mac."}
    assert "**get_info** `{}`" in tools and "os: Darwin" in tools
    assert tick_fn(seen)[0] == {"__type__": "update"}  # unchanged: silent


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


def test_the_stream_wires_can_interrupt_from_the_config(monkeypatch):
    stream = build_stream(FakeResponder(), Conversation())
    assert stream.event_handler.can_interrupt is config.CAN_INTERRUPT  # default: on

    monkeypatch.setenv("VAICE_CAN_INTERRUPT", "0")
    importlib.reload(config)
    try:
        stream = build_stream(FakeResponder(), Conversation())
        assert stream.event_handler.can_interrupt is False
    finally:
        monkeypatch.undo()
        importlib.reload(config)


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

    def boom(before, tools=False):
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
            def launch(server_port, pwa, inbrowser):
                launched.append((server_port, pwa, inbrowser))

    monkeypatch.setattr(app, "parse_args", lambda: argparse.Namespace(command=None))
    monkeypatch.setattr(app, "check_server", lambda: startup.append("check"))
    monkeypatch.setattr(app, "LlamaCppResponder", lambda **_: startup.append("construct") or FakeResponder())
    monkeypatch.setattr(app, "build_stream", lambda responder, conversation: FakeStream())
    monkeypatch.setattr(app.config, "CONVERSATION_FILE", tmp_path / "conversation.json")

    try:
        app.main()

        assert signal.getsignal(signal.SIGINT) is not previous  # Ctrl+C is handled
        assert startup == ["check", "construct"]
        # pwa on, or the browser 404s on /manifest.json; plain vaice does not open the browser
        assert launched == [(app.config.PORT, True, False)]
        assert lifecycle.stopping.is_set()  # the server is gone: the responders may give up
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


def _main_with(monkeypatch, tmp_path, fetch_tools=None, write_capable_tools=None, tools=True,
               conversation_file="default", command=None):
    launched = []
    if tools is not None:
        monkeypatch.setattr(app.config, "LLAMACPP_TOOLS", tools)
    if fetch_tools is not None:
        monkeypatch.setattr(app, "fetch_tools", fetch_tools)
    if write_capable_tools is not None:
        monkeypatch.setattr(app, "write_capable_tools", write_capable_tools)
    monkeypatch.setattr(app, "parse_args", lambda: argparse.Namespace(command=command))
    monkeypatch.setattr(app, "check_server", lambda: None)
    monkeypatch.setattr(app, "LlamaCppResponder", lambda **_: FakeResponder())
    monkeypatch.setattr(app, "build_stream", lambda responder, conversation: _fake_stream_class(launched)())
    monkeypatch.setattr(app.config, "CONVERSATION_FILE",
                        tmp_path / "conversation.json" if conversation_file == "default" else None)
    previous = signal.getsignal(signal.SIGINT)
    try:
        app.main()
    finally:
        signal.signal(signal.SIGINT, previous)
    return launched


def test_main_continues_without_tools_when_the_servers_tools_are_unavailable(monkeypatch, tmp_path, caplog):
    def boom(url):
        raise requests.ConnectionError("refused")

    with caplog.at_level(logging.WARNING, logger="vaice.app"):
        _main_with(monkeypatch, tmp_path, fetch_tools=boom)

    assert "continuing without" in caplog.text


def test_main_logs_the_advertised_tools(monkeypatch, tmp_path, caplog):
    with caplog.at_level(logging.INFO, logger="vaice.app"):
        _main_with(monkeypatch, tmp_path,
                   fetch_tools=lambda url: [{"name": "get_info"}],
                   write_capable_tools=lambda url: [])

    assert "tools enabled: get_info" in caplog.text


def test_main_warns_when_the_advertised_tools_can_write(monkeypatch, tmp_path, caplog):
    with caplog.at_level(logging.WARNING, logger="vaice.app"):
        _main_with(monkeypatch, tmp_path,
                   fetch_tools=lambda url: [{"name": "exec_shell_command"}],
                   write_capable_tools=lambda url: ["exec_shell_command"])

    assert "can change this machine" in caplog.text


def test_main_warns_when_write_capability_cannot_be_checked(monkeypatch, tmp_path, caplog):
    def boom(url):
        raise requests.ConnectionError("refused")

    with caplog.at_level(logging.WARNING, logger="vaice.app"):
        _main_with(monkeypatch, tmp_path,
                   fetch_tools=lambda url: [{"name": "get_info"}],
                   write_capable_tools=boom)

    assert "could not check whether the advertised tools can modify this machine" in caplog.text


def test_main_warns_when_the_server_advertises_no_tools(monkeypatch, tmp_path, caplog):
    with caplog.at_level(logging.WARNING, logger="vaice.app"):
        _main_with(monkeypatch, tmp_path, fetch_tools=lambda url: [])

    assert "advertises none" in caplog.text


def test_main_reports_a_keyboard_interrupt_and_still_requests_stop(monkeypatch, tmp_path, caplog):
    """Ctrl+C while the server is loading: 'stopped' is logged and the responders are told to give up."""
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
    monkeypatch.setattr(app, "LlamaCppResponder", lambda **_: FakeResponder())
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


# -- assistant mode


class FakeAssistant:
    name = "assistant"
    skills = SimpleNamespace(names=("disk_space", "battery_status"))  # the attribute main() reads

    def __init__(self, memory=None):
        self.memory = memory


def _assistant_main(monkeypatch, tmp_path, tools=None):
    def no_tools(url):
        raise AssertionError("assistant mode must not read llama-server's /tools")

    monkeypatch.setattr(app.config, "MODE", "assistant")
    monkeypatch.setattr(app, "AssistantResponder", FakeAssistant)
    monkeypatch.setattr(app, "fetch_tools", no_tools)
    monkeypatch.setattr(app, "write_capable_tools", no_tools)
    return _main_with(monkeypatch, tmp_path, tools=tools)


def test_assistant_mode_uses_the_assistant_responder_and_skips_the_tools_block(monkeypatch, tmp_path, caplog):
    with caplog.at_level(logging.INFO, logger="vaice.app"):
        launched = _assistant_main(monkeypatch, tmp_path)

    assert launched  # the UI started
    assert "assistant mode: skills disk_space, battery_status" in caplog.text
    assert "ignored in assistant mode" not in caplog.text  # VAICE_LLAMACPP_TOOLS is unset


def test_assistant_mode_warns_when_shell_commands_are_on_and_is_quiet_when_they_are_off(monkeypatch, tmp_path, caplog):
    monkeypatch.setattr(app.config, "ASSISTANT_SHELL", True)
    with caplog.at_level(logging.WARNING, logger="vaice.app"):
        _assistant_main(monkeypatch, tmp_path)

    assert "VAICE_ASSISTANT_SHELL is on" in caplog.text
    assert "vaice's own permissions" in caplog.text

    caplog.clear()
    monkeypatch.setattr(app.config, "ASSISTANT_SHELL", False)
    with caplog.at_level(logging.WARNING, logger="vaice.app"):
        _assistant_main(monkeypatch, tmp_path)

    assert "VAICE_ASSISTANT_SHELL" not in caplog.text


def test_assistant_mode_never_builds_the_chat_responder(monkeypatch, tmp_path):
    monkeypatch.setattr(app, "_chat_responder", lambda: pytest.fail("chat responder built in assistant mode"))
    assert _assistant_main(monkeypatch, tmp_path)


def test_assistant_mode_warns_that_the_server_tools_setting_is_ignored(monkeypatch, tmp_path, caplog):
    with caplog.at_level(logging.WARNING, logger="vaice.app"):
        _assistant_main(monkeypatch, tmp_path, tools=True)

    assert "VAICE_LLAMACPP_TOOLS is ignored in assistant mode" in caplog.text


FAKE_KEY = "fake-key-do-not-log-9f3a"


def _helper(monkeypatch, url, model="lfm2.5-24b-a2b"):
    monkeypatch.setattr(app.config, "BRAIN_URL", url)
    monkeypatch.setattr(app.config, "BRAIN_MODEL", model)
    monkeypatch.setattr(app.config, "BRAIN_API_KEY", FAKE_KEY)


@pytest.mark.parametrize("url", ["http://localhost:1234/v1", "http://127.0.0.1:1234/v1", "http://[::1]:1234/v1"])
def test_a_helper_on_this_machine_is_named_without_a_privacy_warning(monkeypatch, tmp_path, caplog, url):
    _helper(monkeypatch, url)
    with caplog.at_level(logging.DEBUG):
        assert _assistant_main(monkeypatch, tmp_path)

    assert "assistant brain: " in caplog.text and ", model lfm2.5-24b-a2b" in caplog.text
    assert "not on this machine" not in caplog.text
    assert FAKE_KEY not in caplog.text


def test_a_remote_helper_gets_a_warning_that_says_what_is_sent_and_that_audio_stays(monkeypatch, tmp_path, caplog):
    _helper(monkeypatch, "https://openrouter.ai/api/v1")
    with caplog.at_level(logging.DEBUG):
        assert _assistant_main(monkeypatch, tmp_path)

    assert "assistant brain: openrouter.ai, model lfm2.5-24b-a2b" in caplog.text
    (warning,) = [r for r in caplog.records if "not on this machine" in r.getMessage()]
    assert warning.levelno == logging.WARNING
    text = warning.getMessage()
    assert "openrouter.ai" in text
    for sent in ("what you say", "earlier turns", "file names", "process names", "computer's name",
                 "shell command output"):
        assert sent in text
    assert "Audio does not leave" in text and "local llama-server" in text
    assert FAKE_KEY not in caplog.text and "key" not in caplog.text.lower()  # the key is not even mentioned


def test_a_helper_without_a_model_warns_but_the_app_starts(monkeypatch, tmp_path, caplog):
    _helper(monkeypatch, "http://localhost:1234/v1", model="")
    with caplog.at_level(logging.INFO):
        assert _assistant_main(monkeypatch, tmp_path)  # continues

    assert "VAICE_BRAIN_MODEL is empty" in caplog.text
    assert "model (none set)" in caplog.text


def test_without_a_helper_nothing_about_a_brain_is_logged(monkeypatch, tmp_path, caplog):
    monkeypatch.setattr(app.config, "BRAIN_URL", "")
    with caplog.at_level(logging.DEBUG):
        _assistant_main(monkeypatch, tmp_path)

    assert "assistant brain" not in caplog.text and "VAICE_BRAIN" not in caplog.text


def test_chat_mode_ignores_a_helper_url_and_says_so_once(monkeypatch, tmp_path, caplog):
    _helper(monkeypatch, "https://openrouter.ai/api/v1")
    monkeypatch.setattr(app.config, "MODE", "chat")
    with caplog.at_level(logging.DEBUG):
        _main_with(monkeypatch, tmp_path, tools=None)

    assert caplog.text.count("VAICE_BRAIN_URL is ignored in chat mode") == 1
    assert "assistant brain" not in caplog.text and "not on this machine" not in caplog.text
    assert FAKE_KEY not in caplog.text


def test_chat_mode_without_a_helper_url_logs_nothing_about_it(monkeypatch, tmp_path, caplog):
    monkeypatch.setattr(app.config, "BRAIN_URL", "")
    monkeypatch.setattr(app.config, "MODE", "chat")
    with caplog.at_level(logging.DEBUG):
        _main_with(monkeypatch, tmp_path, tools=None)

    assert "VAICE_BRAIN_URL" not in caplog.text


def test_chat_mode_is_unchanged_by_the_assistant_responder_existing(monkeypatch, tmp_path, caplog):
    fetched = []
    monkeypatch.setattr(app, "AssistantResponder", lambda: pytest.fail("assistant built in chat mode"))
    monkeypatch.setattr(app.config, "MODE", "chat")

    with caplog.at_level(logging.INFO, logger="vaice.app"):
        _main_with(monkeypatch, tmp_path, fetch_tools=lambda url: fetched.append(url) or [{"name": "get_info"}],
                   write_capable_tools=lambda url: [])

    assert fetched and "tools enabled: get_info" in caplog.text
    assert "assistant mode" not in caplog.text


def test_the_subtitle_names_the_mode(monkeypatch):
    monkeypatch.setattr(app.config, "MODE", "chat")
    assert app._subtitle() == "llama-server · speech-to-speech"
    monkeypatch.setattr(app.config, "MODE", "assistant")
    assert app._subtitle() == "llama-server · assistant"

    stream = build_stream(FakeResponder(), Conversation())
    subtitles = [c for c in stream.ui.get_config_file()["components"]
                 if c["type"] == "markdown" and "assistant" in str(c["props"].get("value", ""))]
    assert len(subtitles) == 1


# --- tool history reaches only the responders that ask for it


def _conversation_with_a_skill_turn():
    conversation = Conversation()
    first = conversation.start_turn()
    first.set_user_text("how long has it been on?")
    first.add_tool_event("system_info", "{}", "up 3 days")
    first.set_assistant_text("Three days.")
    first.finish()
    return conversation


def test_a_responder_with_tool_history_gets_the_tool_results_in_its_history():
    responder = FakeResponder()
    responder.tool_history = True

    list(make_handler(responder, _conversation_with_a_skill_turn())(AUDIO))

    assert responder.calls[0][0] == [("user", "how long has it been on?"), ("tool", "system_info: up 3 days"),
                                     ("assistant", "Three days.")]


def test_a_responder_without_tool_history_never_gets_tool_entries():
    responder = FakeResponder()

    list(make_handler(responder, _conversation_with_a_skill_turn())(AUDIO))

    assert responder.calls[0][0] == [("user", "how long has it been on?"), ("assistant", "Three days.")]


def test_the_llamacpp_responder_never_receives_tool_entries(monkeypatch):
    seen = []
    responder = LlamaCppResponder("http://127.0.0.1:9", "Be brief.")
    monkeypatch.setattr(responder, "respond", lambda audio, history, turn: iter(seen.append(list(history)) or []))

    list(make_handler(responder, _conversation_with_a_skill_turn())(AUDIO))

    assert seen == [[("user", "how long has it been on?"), ("assistant", "Three days.")]]


# --- the confirmation panel (assistant mode)


class ConfirmingResponder(FakeResponder):
    """A responder that asks for confirmation, as AssistantResponder does."""

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
    return Pending("quit_app", arguments, "Shall I quit Safari?", "call_0", [], conv.Turn())


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


def test_chat_mode_has_no_confirmation_panel():
    page = confirm_page(FakeResponder())

    assert set(page.clicks) == {"New conversation"}
    assert len(page.tick_dep["outputs"]) == 3  # chatbot, tool panel, seen
    assert len(page.tick("")) == 3


def test_assistant_mode_has_a_hidden_panel_with_run_and_cancel_above_the_tool_activity():
    responder = ConfirmingResponder()
    page = confirm_page(responder)
    components = page.config["components"]

    assert set(page.clicks) == {"New conversation", "Run", "Cancel"}
    (panel,) = [c for c in components if c["type"] == "column" and c["props"].get("visible") is False]
    (accordion,) = [c for c in components if c["type"] == "accordion"]
    order = [c["id"] for c in components]
    assert order.index(panel["id"]) < order.index(accordion["id"])
    hidden, text, pending_id, _ = page.tick("")[2:]
    assert hidden == {"visible": False, "__type__": "update"} and text == "" and pending_id == ""


def test_the_panel_shows_the_question_and_exactly_what_will_run_and_is_drawn_once():
    responder = ConfirmingResponder()
    page = confirm_page(responder)
    _, _, _, _, _, seen = page.tick("")  # the first tick draws the hidden panel
    request = waiting_request()
    responder.waiting = request

    *_, shown, text, pending_id, seen = page.tick(seen)

    assert shown == {"visible": True, "__type__": "update"}
    assert text == 'Shall I quit Safari?\n\n```\nquit_app {"name": "Safari"}\n```'
    assert pending_id == request.id
    assert all(item == {"__type__": "update"} for item in page.tick(seen)[:5])  # unchanged: nothing re-renders

    responder.waiting = None  # settled somewhere else: the panel goes away
    *_, hidden, text, pending_id, seen = page.tick(seen)
    assert hidden == {"visible": False, "__type__": "update"} and pending_id == ""


def shell_request(command, voice, question="Shall I run the command ls? The full command is on the screen."):
    return Pending("run_shell", json.dumps({"command": command}), question, "call_0", [], conv.Turn(), voice=voice)


def test_a_shell_request_shows_the_command_line_itself_not_the_json():
    view = app.pending_view(shell_request("ls -la ~/Downloads", voice=True))

    _, text, _ = app.confirmation_panel(view)

    assert text == ("Shall I run the command ls? The full command is on the screen.\n\n"
                    "```\nls -la ~/Downloads\n```")
    assert "Only the Run button" not in text  # a spoken yes may approve this one


def test_a_request_that_needs_the_button_says_so_under_the_question():
    view = app.pending_view(shell_request("echo a | wc -l", voice=False, question="I would run a command."))

    assert view["voice"] is False
    _, text, _ = app.confirmation_panel(view)

    assert text == ("I would run a command.\n\n"
                    "Only the Run button can approve this one; saying yes will not.\n\n"
                    "```\necho a | wc -l\n```")


def test_a_shell_command_is_shown_exactly_and_backticks_cannot_close_its_block():
    _, text, _ = app.confirmation_panel(app.pending_view(shell_request("echo ```x```\nls", voice=False)))

    assert text.endswith("````\necho ```x```\nls\n````")  # a newline in the command is shown as it is


@pytest.mark.parametrize("arguments", ["{not json", "[]", '{"command": 5}', "{}"])
def test_a_shell_request_with_odd_arguments_falls_back_to_the_raw_call(arguments):
    request = Pending("run_shell", arguments, "Q?", "call_0", [], conv.Turn(), voice=False)

    _, text, _ = app.confirmation_panel(app.pending_view(request))

    assert f"```\nrun_shell {arguments}\n```" in text


def test_the_panel_shows_a_view_without_the_voice_key_as_voice_approvable():
    view = {"id": "x", "question": "Shall I quit Safari?", "name": "quit_app", "arguments": "{}"}

    assert "Only the Run button" not in app.confirmation_panel(view)[1]


def test_arguments_with_backticks_cannot_close_the_code_block():
    _, text, _ = app.confirmation_panel(app.pending_view(waiting_request('{"name": "a ``` b"}')))

    assert text.splitlines()[2] == "````" and text.splitlines()[-1] == "````"


def test_run_and_cancel_pass_the_id_the_panel_showed_and_hide_it_and_redraw_on_the_next_tick():
    responder = ConfirmingResponder()
    page = confirm_page(responder)
    run, run_dep = page.clicks["Run"]
    cancel, _ = page.clicks["Cancel"]
    state_ids = {c["id"] for c in page.config["components"] if c["type"] == "state"}

    assert run("id-from-the-panel") == ({"visible": False, "__type__": "update"}, "", "")
    cancel("another-id")

    assert responder.approved == ["id-from-the-panel"] and responder.cancelled == ["another-id"]
    assert set(run_dep["outputs"]) - state_ids != set()  # the panel itself
    assert len(run_dep["outputs"]) == 3  # the panel, the id the panel holds, and `seen`: the blank forces a redraw


def test_new_conversation_clears_a_waiting_request_and_chat_mode_does_not_need_the_method():
    responder = ConfirmingResponder()
    responder.waiting = waiting_request()
    page = confirm_page(responder)

    page.clicks["New conversation"][0]()

    assert responder.cleared == 1 and responder.pending() is None
    confirm_page(FakeResponder()).clicks["New conversation"][0]()  # no clear_pending: no error


# -- assistant mode with memory: the Memory panel, the startup line and the privacy warning


class MemoryResponder(FakeResponder):
    """What `build_stream` finds on an assistant responder with memory on: a `memory` holding the notes."""

    def __init__(self, memory):
        super().__init__()
        self.memory = memory


def _tick_fn(stream):
    config = stream.ui.get_config_file()
    ours = [c for c in config["components"] if c["type"] == "timer" and c["props"]["value"] == VIEW_REFRESH_SECONDS]
    (tick,) = [d for d in config["dependencies"] if d["targets"][0] == (ours[0]["id"], "tick")]
    return config, tick, stream.ui.fns[tick["id"]].fn


def test_the_memory_panel_is_a_collapsed_accordion_under_the_tool_panel_in_assistant_mode_with_memory():
    stream = build_stream(MemoryResponder(Memory(None)), Conversation())
    config = stream.ui.get_config_file()

    labels = [c["props"]["label"] for c in config["components"] if c["type"] == "accordion"]
    assert labels == ["Tool activity", "Memory"]  # in the order they appear
    memory_accordion = [c for c in config["components"] if c["type"] == "accordion"][1]
    assert memory_accordion["props"]["open"] is False


def test_chat_mode_and_assistant_mode_without_memory_have_no_memory_panel():
    class NoMemory(FakeResponder):
        memory = None  # VAICE_ASSISTANT_MEMORY off

    for responder in (FakeResponder(), NoMemory()):
        stream = build_stream(responder, Conversation())
        config = stream.ui.get_config_file()
        assert [c["props"]["label"] for c in config["components"] if c["type"] == "accordion"] == ["Tool activity"]
        _, tick, tick_fn = _tick_fn(stream)
        assert len(tick["outputs"]) == 3  # chatbot, tool panel, seen: no memory column
        assert len(tick_fn("")) == 3


def test_the_timer_fills_the_memory_panel_when_the_notes_change_and_is_silent_otherwise():
    memory = Memory(None)
    stream = build_stream(MemoryResponder(memory), Conversation())
    _, tick, tick_fn = _tick_fn(stream)
    assert len(tick["outputs"]) == 4  # chatbot, tool panel, memory panel, seen

    _, _, shown, seen = tick_fn("")
    assert shown == "Nothing remembered."
    assert tick_fn(seen)[0] == {"__type__": "update"}  # unchanged: nothing re-rendered

    memory.add("The user parks on level three.")
    _, _, shown, seen = tick_fn(seen)
    assert shown == "- The user parks on level three\\."
    assert tick_fn(seen)[0] == {"__type__": "update"}

    memory.clear()
    assert tick_fn(seen)[2] == "Nothing remembered."


def test_the_memory_panel_comes_after_the_confirmation_panel_when_both_exist():
    class Asking(MemoryResponder):
        def pending(self):
            return None

        approve = cancel = clear_pending = staticmethod(lambda *_args: None)

    stream = build_stream(Asking(Memory(None)), Conversation())
    _, tick, tick_fn = _tick_fn(stream)

    result = tick_fn("")
    assert len(tick["outputs"]) == len(result) == 7  # chatbot, tool, confirm x3, memory, seen
    assert result[5] == "Nothing remembered."


def test_new_conversation_leaves_the_notes_alone():
    memory = Memory(None)
    memory.add("The user parks on level three.")
    conversation = Conversation()
    stream = build_stream(MemoryResponder(memory), conversation)
    config = stream.ui.get_config_file()
    (click,) = [d for d in config["dependencies"] if d["targets"][0][1] == "click"]
    conversation.start_turn().set_user_text("hello")

    assert stream.ui.fns[click["id"]].fn() == ([], "")  # still just the chatbot and the tool panel

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
    """main() in assistant mode with the real `_open_memory`; returns the `memory` the responder was given."""
    given = []

    class Responder(FakeAssistant):
        def __init__(self, memory=None):
            super().__init__(memory)
            given.append(memory)

    path = tmp_path / "memory.json" if memory_file == "default" else memory_file
    if notes is not None and path is not None:
        path.write_text(json.dumps({"version": 1, "notes": notes}))
    monkeypatch.setattr(app.config, "ASSISTANT_MEMORY", on)
    monkeypatch.setattr(app.config, "MEMORY_FILE", path)
    monkeypatch.setattr(app.config, "MODE", "assistant")
    monkeypatch.setattr(app, "AssistantResponder", Responder)
    assert _main_with(monkeypatch, tmp_path, tools=None)
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
    assert "assistant memory is off" in caplog.text and "VAICE_ASSISTANT_MEMORY" in caplog.text


def test_chat_mode_does_not_open_the_memory(monkeypatch, tmp_path):
    monkeypatch.setattr(app, "_open_memory", lambda: pytest.fail("chat mode opened the memory"))
    monkeypatch.setattr(app.config, "MODE", "chat")

    assert _main_with(monkeypatch, tmp_path, tools=None)


def test_a_remote_helper_warning_says_the_saved_notes_are_sent_too_only_while_memory_is_on(
        monkeypatch, tmp_path, caplog):
    _helper(monkeypatch, "https://openrouter.ai/api/v1")
    monkeypatch.setattr(app.config, "ASSISTANT_MEMORY", True)
    with caplog.at_level(logging.WARNING):
        assert _assistant_main(monkeypatch, tmp_path)

    (warning,) = [r for r in caplog.records if "not on this machine" in r.getMessage()]
    assert "the notes you asked it to keep about you" in warning.getMessage()
    assert "Audio does not leave" in warning.getMessage()

    caplog.clear()
    monkeypatch.setattr(app.config, "ASSISTANT_MEMORY", False)
    with caplog.at_level(logging.WARNING):
        assert _assistant_main(monkeypatch, tmp_path)

    (warning,) = [r for r in caplog.records if "not on this machine" in r.getMessage()]
    assert "notes" not in warning.getMessage()
