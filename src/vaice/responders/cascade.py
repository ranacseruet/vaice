"""Cascade backend: audio → LLM text (llama.cpp) → sentences → TTS (Kokoro) → audio."""
from collections.abc import Callable, Iterable, Iterator

from vaice import llm, tts
from vaice.audio import to_wav_b64
from vaice.responders.base import AudioFrame
from vaice.sentences import split_sentences


class CascadeResponder:
    name = "cascade"

    def __init__(
        self,
        stream_reply: Callable[[str], Iterable[str]] = llm.stream_reply,
        speak: Callable[[str], Iterable[AudioFrame]] = tts.speak,
    ):
        self._stream_reply = stream_reply
        self._speak = speak

    def respond(self, audio: AudioFrame) -> Iterator[AudioFrame]:
        # Each sentence is spoken as soon as it completes, while the LLM keeps generating
        for sentence in split_sentences(self._stream_reply(to_wav_b64(audio))):
            yield from self._speak(sentence)
