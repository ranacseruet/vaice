import numpy as np
import pytest

from vaice.conversation import Turn
from vaice.responders.cascade import CascadeResponder

AUDIO = (48000, np.zeros((1, 480), dtype=np.int16))


def chunk():
    return (24000, np.zeros(1, dtype=np.float32))


def make(stream_reply, speak=None, transcribe=lambda _: "what was asked"):
    return CascadeResponder(stream_reply, speak or (lambda s: iter([chunk()])), transcribe)


def test_sentences_are_spoken_while_llm_is_still_streaming():
    events = []

    def stream_reply(_question, _history):
        for fragment in ["Hello", " there.", " How are", " you?"]:
            events.append(f"llm:{fragment}")
            yield fragment

    def speak(sentence):
        events.append(f"speak:{sentence}")
        yield chunk()

    chunks = list(CascadeResponder(stream_reply, speak, lambda _: "hi").respond(AUDIO, [], Turn()))

    assert len(chunks) == 2
    assert [e for e in events if e.startswith("speak:")] == ["speak:Hello there.", "speak:How are you?"]
    # first sentence is spoken before the LLM has produced its last fragment
    assert events.index("speak:Hello there.") < events.index("llm: you?")


def test_empty_reply_yields_no_audio():
    assert list(make(lambda *_: iter([]), lambda s: iter([])).respond(AUDIO, [], Turn())) == []


def test_history_reaches_the_llm():
    seen = []

    def stream_reply(question, history):
        seen.append((question, list(history)))
        return iter(["Hi."])

    history = [("user", "hello"), ("assistant", "hi")]
    list(make(stream_reply).respond(AUDIO, history, Turn()))

    assert seen == [("what was asked", history)]  # the reply is to the transcript, not the audio


def test_spoken_text_grows_with_each_fully_yielded_sentence():
    turn = Turn()
    seen = []

    def speak(sentence):
        seen.append(turn.assistant_text)  # what was recorded before this sentence was spoken
        yield chunk()

    list(make(lambda *_: iter(["**One** fish. Two", " fish!"]), speak).respond(AUDIO, [], turn))

    assert seen == ["", "One fish."]
    assert turn.assistant_text == "One fish. Two fish!"  # cleaned for speech, like the audio


def test_a_reply_closed_mid_reply_records_only_the_sentences_fully_yielded():
    turn = Turn()

    def speak(sentence):
        yield chunk()
        yield chunk()

    gen = make(lambda *_: iter(["First one. Second one. Third one."]), speak).respond(AUDIO, [], turn)
    for _ in range(3):  # both chunks of "First one.", then the first chunk of "Second one."
        next(gen)
    assert turn.assistant_text == "First one."
    gen.close()

    assert turn.assistant_text == "First one."


def test_a_reply_stops_at_the_next_sentence_once_the_turn_is_closed():
    spoken = []
    turn = Turn()

    def speak(sentence):
        spoken.append(sentence)
        yield chunk()
        yield chunk()

    gen = make(lambda *_: iter(["First one. ", "Second one. ", "Third one."]), speak).respond(AUDIO, [], turn)
    next(gen), next(gen)  # both chunks of "First one."
    next(gen)  # records "First one." and hands out the first chunk of "Second one."
    assert turn.assistant_text == "First one."

    turn.close()  # the user barged in or reset the conversation
    list(gen)

    assert spoken == ["First one.", "Second one."]  # "Third one." was never spoken or recorded
    assert turn.assistant_text == "First one."  # "Second one." was mid-sentence at the close


def test_the_transcript_is_recorded_before_any_audio():
    turn = Turn()
    gen = make(lambda *_: iter(["Hi."]), transcribe=lambda b64: "what was asked").respond(AUDIO, [], turn)
    next(gen)

    assert turn.user_text == "what was asked"


def test_an_empty_transcript_gets_no_reply():
    def stream_reply(*_):
        raise AssertionError("should not be called")

    turn = Turn()
    assert list(make(stream_reply, transcribe=lambda _: "").respond(AUDIO, [], turn)) == []
    assert turn.user_text == ""


def test_a_closed_turn_is_not_answered():
    spoken = []
    turn = Turn()
    turn.close()  # the next turn started (barge-in or reset) while this reply was queued

    def speak(sentence):
        spoken.append(sentence)
        yield chunk()

    list(make(lambda *_: iter(["First one. Second one."]), speak).respond(AUDIO, [], turn))

    assert spoken == []  # the LLM stream was abandoned before its first sentence


def test_a_failed_transcription_raises_and_leaves_an_empty_transcript():
    def broken(_):
        raise RuntimeError("llama-server fell over")

    turn = Turn()
    with pytest.raises(RuntimeError, match="llama-server fell over"):
        list(make(lambda *_: iter(["Hi."]), transcribe=broken).respond(AUDIO, [], turn))

    assert turn.user_text == ""  # history shows it as inaudible instead of waiting for it
