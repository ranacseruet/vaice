"""The seam between the UI and however a spoken reply is produced."""
from collections.abc import Iterator, Sequence
from typing import Protocol

import numpy as np

from vaice.conversation import Message, Turn

# fastrtc audio: (sample_rate, samples). Input is int16 shaped (channels, n);
# output we yield is float32 at the responder's own rate.
AudioFrame = tuple[int, np.ndarray]


class Responder(Protocol):
    """Turns what the user said into audio to play back, streaming as it goes.

    Besides the audio, a responder reports back on `turn`:
    - as audio is handed out, `turn.set_assistant_text(...)` with the text spoken so far
      (cumulative; text for audio that was never yielded must not be included);
    - `turn.set_user_text(...)` exactly once with the transcript of `audio`, also when the reply
      fails or is interrupted (on failure: log the exception and set ""). Transcription must not
      delay the first audio.
    `history` holds the earlier turns as text. `turn.closed` is set when the next turn starts;
    a responder may poll it to stop generating.
    """

    name: str  # short label shown in logs, e.g. "native"

    def respond(self, audio: AudioFrame, history: Sequence[Message], turn: Turn) -> Iterator[AudioFrame]: ...
