import numpy as np

from vaice.responders.cascade import CascadeResponder

AUDIO = (48000, np.zeros((1, 480), dtype=np.int16))


def test_sentences_are_spoken_while_llm_is_still_streaming():
    events = []

    def stream_reply(_wav_b64):
        for fragment in ["Hello", " there.", " How are", " you?"]:
            events.append(f"llm:{fragment}")
            yield fragment

    def speak(sentence):
        events.append(f"speak:{sentence}")
        yield (24000, np.zeros(1, dtype=np.float32))

    chunks = list(CascadeResponder(stream_reply, speak).respond(AUDIO))

    assert len(chunks) == 2
    assert [e for e in events if e.startswith("speak:")] == ["speak:Hello there.", "speak:How are you?"]
    # first sentence is spoken before the LLM has produced its last fragment
    assert events.index("speak:Hello there.") < events.index("llm: you?")


def test_empty_reply_yields_no_audio():
    chunks = list(CascadeResponder(lambda _: iter([]), lambda s: iter([])).respond(AUDIO))
    assert chunks == []
