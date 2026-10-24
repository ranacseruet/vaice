import logging
import threading

import numpy as np
import pytest
from fastrtc import AdditionalOutputs

from vaice import app
from vaice import conversation as conv
from vaice.app import build_stream, make_handler, parse_args
from vaice.conversation import Conversation
from vaice.responders.cascade import CascadeResponder

AUDIO = (48000, np.zeros((1, 480), dtype=np.int16))


@pytest.fixture(autouse=True)
def short_waits(monkeypatch):
    monkeypatch.setattr(conv, "TRANSCRIPT_WAIT", 0.05)
    monkeypatch.setattr(app, "UI_TRANSCRIPT_WAIT", 0.05)


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


def barge_in_setup(tmp_path):
    histories = []
    replies = iter([["First one. Second one. Third one."], ["Fine."]])
    questions = iter(["question one", "question two"])

    def stream_reply(_b64, history):
        histories.append(list(history))
        return iter(next(replies))

    def speak(_sentence):
        yield (24000, np.zeros(1, dtype=np.float32))
        yield (24000, np.zeros(1, dtype=np.float32))

    conversation = Conversation(tmp_path / "conversation.json")
    handler = make_handler(CascadeResponder(stream_reply, speak, lambda _b64: next(questions)), conversation)
    return handler, conversation, histories


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


def test_backend_flag_overrides_env_default(monkeypatch):
    monkeypatch.setattr(app.config, "BACKEND", "native")
    assert parse_args([]).backend == "native"
    assert parse_args(["--backend", "cascade"]).backend == "cascade"


def test_invalid_backend_from_env_is_a_usage_error_not_a_traceback(monkeypatch, capsys):
    monkeypatch.setattr(app.config, "BACKEND", "Native")
    with pytest.raises(SystemExit) as exc:
        parse_args([])
    assert exc.value.code == 2
    assert "VAICE_BACKEND" in capsys.readouterr().err


def test_ui_has_the_transcript_a_reset_button_and_loads_the_stored_conversation():
    conversation = Conversation()
    stream = build_stream(FakeResponder(), conversation)
    config = stream.ui.get_config_file()

    (chatbot,) = [c for c in config["components"] if c["type"] == "chatbot"]
    (button,) = [c for c in config["components"] if c["type"] == "button"]
    assert chatbot["props"]["type"] == "messages"
    assert button["props"]["value"] == "New conversation"
    triggers = {(d["targets"][0][1], tuple(d["outputs"])) for d in config["dependencies"] if d["targets"][0][1] in ("load", "click")}
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
