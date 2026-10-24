"""The seam between the UI and however a spoken reply is produced."""
from collections.abc import Iterator
from typing import Protocol

import numpy as np

# fastrtc audio: (sample_rate, samples). Input is int16 shaped (channels, n);
# output we yield is float32 at the responder's own rate.
AudioFrame = tuple[int, np.ndarray]


class Responder(Protocol):
    """Turns what the user said into audio to play back, streaming as it goes."""

    name: str  # short label shown in logs, e.g. "native"

    def respond(self, audio: AudioFrame) -> Iterator[AudioFrame]: ...
