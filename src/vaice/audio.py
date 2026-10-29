"""Mic audio helpers."""
import math

import numpy as np
from scipy.signal import resample_poly


def to_mono_float(samples: np.ndarray) -> np.ndarray:
    """fastrtc int16 (channels, n) or (n,) → float32 mono in [-1, 1]."""
    scale = 32768.0 if np.issubdtype(samples.dtype, np.integer) else 1.0
    mono = samples.mean(axis=0) if samples.ndim > 1 else samples  # mean() promotes ints to float64
    return (mono / scale).astype(np.float32)


def to_pcm16(audio: tuple[int, np.ndarray], rate: int) -> bytes:
    """A fastrtc (sample_rate, samples) frame → mono little-endian PCM16 at `rate`.

    The realtime endpoint wants exactly its input rate (16 kHz for LFM3-Audio); the browser delivers
    48 kHz.
    """
    sample_rate, samples = audio
    mono = to_mono_float(samples)
    if sample_rate != rate:
        divisor = math.gcd(sample_rate, rate)
        mono = resample_poly(mono, rate // divisor, sample_rate // divisor)
    return np.clip(np.round(mono * 32768.0), -32768, 32767).astype("<i2").tobytes()


def from_pcm16(data: bytes) -> np.ndarray:
    """Mono little-endian PCM16 bytes → float32 in [-1, 1]."""
    return np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0
