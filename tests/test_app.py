import argparse
import importlib
import itertools
import logging
import signal

import numpy as np
import pytest
import requests
from fake_realtime import pcm
from fastrtc import AdditionalOutputs
from threads import next_on_fresh_thread, run_on_thread

from vaice import app, config, lifecycle
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
