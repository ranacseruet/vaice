"""Mic audio helpers."""
import base64
import io

import numpy as np
import soundfile as sf


def to_wav_b64(audio: tuple[int, np.ndarray]) -> str:
    """Encode a fastrtc (sample_rate, samples) frame as a base64 WAV.

    fastrtc delivers int16 samples shaped (channels, n), typically at 48 kHz.
    llama.cpp resamples to the model's rate itself, so no resampling here.
    """
    sample_rate, samples = audio
    buf = io.BytesIO()
    sf.write(buf, samples.T if samples.ndim > 1 else samples, sample_rate, format="WAV")
    return base64.b64encode(buf.getvalue()).decode("utf-8")
