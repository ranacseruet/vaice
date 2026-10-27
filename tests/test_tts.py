import numpy as np

from vaice import config, tts


class FakeResult:
    audio = np.zeros(4, dtype=np.float32)


class FakeModel:
    def __init__(self):
        self.texts = []

    def generate(self, text, voice, speed):
        self.texts.append(text)
        yield FakeResult()


def test_warm_up_synthesizes_once_so_the_first_reply_does_not(monkeypatch):
    model = FakeModel()
    monkeypatch.setattr(tts, "load_model", lambda: model)

    tts.warm_up()

    assert model.texts == ["Ready."]


def test_the_sample_rate_is_kokoros_own_and_not_configurable():
    assert tts.SAMPLE_RATE == 24000
    assert not hasattr(config, "TTS_SAMPLE_RATE")  # it is fixed by the model, not a setting


def test_speak_yields_the_sample_rate(monkeypatch):
    monkeypatch.setattr(tts, "load_model", lambda: FakeModel())

    (rate, audio), = list(tts.speak("Hello there."))

    assert rate == tts.SAMPLE_RATE
    assert audio.dtype == np.float32
