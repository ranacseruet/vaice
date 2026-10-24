import numpy as np

from vaice import tts


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
