"""Kokoro text-to-speech on Apple Silicon (MLX)."""
from collections.abc import Iterator
from functools import lru_cache

import numpy as np

from vaice import config
from vaice.sentences import clean_for_speech

SAMPLE_RATE = 24000  # fixed by Kokoro


@lru_cache(maxsize=1)
def load_model():
    from mlx_audio.tts.utils import load_model as _load  # heavy import, keep it out of module load

    return _load(config.TTS_MODEL)


def warm_up() -> None:
    """Load the model and synthesize once: Kokoro builds its text pipeline (~3 s) on the first generate."""
    for _ in speak("Ready."):
        pass


def speak(text: str) -> Iterator[tuple[int, np.ndarray]]:
    """Synthesize text, yielding (sample_rate, float32 samples) chunks."""
    text = clean_for_speech(text)
    if not text:
        return
    for res in load_model().generate(text=text, voice=config.TTS_VOICE, speed=config.TTS_SPEED):
        yield (SAMPLE_RATE, np.array(res.audio, dtype=np.float32))
