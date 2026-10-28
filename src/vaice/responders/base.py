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
      fails or is interrupted (on failure set "", then log or re-raise), so the next turn's history
      never waits for it in vain. Transcribing alongside or after the reply keeps it off the path
      to the first audio; transcribing first is fine when the reply needs the text.
    `history` holds the earlier turns as text. `turn.closed` is set when the next turn starts;
    a responder may poll it to stop generating.
    """

    name: str  # short label shown in logs, e.g. "llamacpp"

    def respond(self, audio: AudioFrame, history: Sequence[Message], turn: Turn) -> Iterator[AudioFrame]: ...


def spoken_part(text: str, total: int, handed_out: int) -> str:
    """The words of `text` that the first `handed_out` of `total` units of audio cover (approximate).

    Interleaved generation writes its text about twice as fast as speaking speed, so the text is
    often complete after the first second of audio. Taking the text generated so far would record a
    whole reply that was cut off early; split it by the share of audio handed out instead. The units
    (frames, samples) are the caller's; `total` is the audio generated so far, not the final length.
    """
    if not total:  # text-only reply: nothing was said
        return ""
    if handed_out >= total:
        return text
    words = text.split()
    # truncate, don't round: the same conservative estimate as Turn.close(), which floors and keeps
    # at least one word because some audio played — words the user had no time to hear stay uncounted
    keep = max(1, int(len(words) * handed_out / total)) if handed_out else 0
    return " ".join(words[:keep])
