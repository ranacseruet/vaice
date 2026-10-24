"""Native backend: one speech-to-speech model (mlx-audio) turns audio into audio directly."""
import logging
import queue
import threading
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor

import mlx.core as mx
import numpy as np
from mlx_audio.sts.models.lfm_audio.model import AUDIO_EOS_TOKEN, LFM2AudioModel, LFMModality
from mlx_audio.sts.models.lfm_audio.processor import ChatState, LFM2AudioProcessor

from vaice import config
from vaice.audio import to_mono_float
from vaice.responders.base import AudioFrame

log = logging.getLogger(__name__)

OUTPUT_SAMPLE_RATE = 24000
# Decode this many audio frames at a time so playback starts before generation ends
FRAMES_PER_CHUNK = 10
# The detokenizer is stateless, so each chunk is decoded together with the previous frames as
# left context (then trimmed). Without it the output drifts from a whole-reply decode and
# chunk edges are audibly rougher.
LEFT_CONTEXT_FRAMES = FRAMES_PER_CHUNK
MAX_NEW_TOKENS = 512  # counts text tokens and audio frames together

_DONE = object()


class NativeResponder:
    """Runs every MLX call on one dedicated thread.

    MLX streams are bound to the thread that created them, but fastrtc resumes our
    generator from whichever worker thread is free on each chunk. So the model is
    loaded and run on a single worker, and respond() only passes chunks back through
    a queue.
    """

    name = "native"

    def __init__(self, model, processor, prompt: str = config.NATIVE_PROMPT, executor: ThreadPoolExecutor | None = None):
        self._model = model
        self._processor = processor
        self._prompt = prompt
        self._executor = executor or ThreadPoolExecutor(max_workers=1, thread_name_prefix="vaice-mlx")

    @classmethod
    def load(cls, model_name: str = config.NATIVE_MODEL) -> "NativeResponder":
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vaice-mlx")
        try:
            model, processor = executor.submit(
                lambda: (LFM2AudioModel.from_pretrained(model_name), LFM2AudioProcessor.from_pretrained(model_name))
            ).result()
            responder = cls(model, processor, executor=executor)
            # The first generation costs ~2s extra (kernel compilation); pay it at startup, not on the first question
            for _ in responder.respond((16000, np.zeros((1, 8000), dtype=np.int16))):
                pass
        except BaseException:
            executor.shutdown(wait=True)  # don't leave a live worker behind (auto mode falls back to cascade)
            raise
        return responder

    def respond(self, audio: AudioFrame) -> Iterator[AudioFrame]:
        chunks: queue.Queue = queue.Queue()
        cancelled = threading.Event()

        def work() -> None:
            try:
                for chunk in self._generate(audio, cancelled):
                    chunks.put(chunk)
            except BaseException as exc:  # re-raised in the consumer below
                chunks.put(exc)
            finally:
                chunks.put(_DONE)

        self._executor.submit(work)
        try:
            while (item := chunks.get()) is not _DONE:
                if isinstance(item, BaseException):
                    raise item
                yield item
        finally:
            cancelled.set()  # consumer stopped early (e.g. user interrupted): free the worker

    def _generate(self, audio: AudioFrame, cancelled: threading.Event) -> Iterator[AudioFrame]:
        if cancelled.is_set():  # abandoned while queued: skip the audio encode + prefill too
            return
        sample_rate, samples = audio
        # The processor resamples to the model's rate itself
        chat = ChatState(self._processor)
        chat.new_turn("system")
        chat.add_text(self._prompt)
        chat.end_turn()
        chat.new_turn("user")
        chat.add_audio(mx.array(to_mono_float(samples)), sample_rate=sample_rate)
        chat.end_turn()
        chat.new_turn("assistant")

        history: list[mx.array] = []  # every audio frame so far, for left context
        chunk_start = 0
        n_tokens = 0
        stream = self._model.generate_from_chat_state(
            chat, mode="interleaved", max_new_tokens=MAX_NEW_TOKENS, audio_temperature=1.0, audio_top_k=4
        )
        for token, modality in stream:
            if cancelled.is_set():
                return
            n_tokens += 1
            if modality != LFMModality.AUDIO_OUT or int(token[0].item()) == AUDIO_EOS_TOKEN:
                continue
            history.append(token)
            if len(history) - chunk_start == FRAMES_PER_CHUNK:
                yield self._decode(history, chunk_start)
                chunk_start = len(history)
        if chunk_start < len(history):
            yield self._decode(history, chunk_start)
        if n_tokens >= MAX_NEW_TOKENS:
            log.warning("native reply hit the %d-token limit and was cut off", MAX_NEW_TOKENS)

    def _decode(self, history: list[mx.array], chunk_start: int) -> AudioFrame:
        """Decode history[chunk_start:] with left context, returning only that span's audio."""
        context = min(LEFT_CONTEXT_FRAMES, chunk_start)
        window = history[chunk_start - context :]
        codes = mx.stack(window, axis=1)[None, :]  # detokenizer wants (batch, codebooks, time)
        wav = np.array(self._processor.decode_audio(codes), dtype=np.float32).reshape(-1)
        return (OUTPUT_SAMPLE_RATE, wav[context * (len(wav) // len(window)) :])
