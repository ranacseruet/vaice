"""Cascade backend: audio → transcript → LLM text (llama.cpp) → sentences → TTS (Kokoro) → audio."""
import logging
from collections.abc import Callable, Iterable, Iterator, Sequence

from vaice import llm, tts
from vaice.audio import to_wav_b64
from vaice.conversation import Message, Turn
from vaice.responders.base import AudioFrame
from vaice.sentences import clean_for_speech, split_sentences

log = logging.getLogger(__name__)


class CascadeResponder:
    """Transcribes first, then replies in text.

    Gemma hears audio well in a single-turn prompt, but with earlier turns in the prompt it often
    mishears the new clip or answers an earlier question. A text conversation keeps the context
    right, at the cost of one short transcription request before the reply.
    """

    name = "cascade"

    def __init__(
        self,
        stream_reply: Callable[[str, Sequence[Message]], Iterable[str]] = llm.stream_reply,
        speak: Callable[[str], Iterable[AudioFrame]] = tts.speak,
        transcribe: Callable[[str], str] = llm.transcribe,
    ):
        self._stream_reply = stream_reply
        self._speak = speak
        self._transcribe = transcribe

    def respond(self, audio: AudioFrame, history: Sequence[Message], turn: Turn) -> Iterator[AudioFrame]:
        try:
            question = self._transcribe(to_wav_b64(audio))
        except Exception:
            turn.set_user_text("")  # history shows it as inaudible; the error still surfaces
            raise
        turn.set_user_text(question)
        if not question:  # nothing intelligible was said (a cough, background noise)
            log.info("[cascade] empty transcript; not replying")
            return
        spoken: list[str] = []
        # Each sentence is spoken as soon as it completes, while the LLM keeps generating
        for sentence in split_sentences(self._stream_reply(question, history)):
            yield from self._speak(sentence)
            # Only after all its audio was handed out: a reply closed mid-sentence records the sentences before it
            if text := clean_for_speech(sentence):
                spoken.append(text)
                turn.set_assistant_text(" ".join(spoken))
