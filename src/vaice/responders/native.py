"""Native backend: one speech-to-speech model (mlx-audio) turns audio into audio directly."""
import logging
import queue
import threading
import weakref
from collections.abc import Iterator, Sequence
from concurrent.futures import Future

import mlx.core as mx
import numpy as np
from mlx_audio.sts.models.lfm_audio.model import (
    AUDIO_EOS_TOKEN,
    AUDIO_START_TOKEN,
    IM_END_TOKEN,
    TEXT_END_TOKEN,
    LFM2AudioModel,
    LFMModality,
)
from mlx_audio.sts.models.lfm_audio.processor import ChatState, LFM2AudioProcessor

from vaice import config, lifecycle
from vaice.audio import to_mono_float
from vaice.conversation import Message, Turn
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
WARMUP_ASR_WAIT = 60  # seconds to wait for the warm-up transcription before giving up on the model
ASR_PROMPT = "Perform ASR."
ASR_MAX_NEW_TOKENS = 256
# How often the consumer checks whether the app is stopping while it waits for the model
POLL_SECONDS = 0.25
# Control tokens the model emits as text; decode_text would spell them out
_CONTROL_TOKENS = {AUDIO_START_TOKEN, IM_END_TOKEN, TEXT_END_TOKEN}

_DONE = object()


def _run_tasks(tasks: queue.Queue) -> None:
    """Run the queued tasks one at a time, until the queue is closed."""
    while True:
        task = tasks.get()
        if task is None:
            return
        _run_one(task)
        # The task holds the future, which may hold the exception of a failed reply; that
        # traceback holds the responder (and with it this worker). Drop the task before waiting
        # for the next one, or the worker would never be collected and its thread would never
        # stop.
        del task


def _run_one(task) -> None:
    """Run one task in its own frame, so that nothing of it outlives the call."""
    future, fn = task
    if not future.set_running_or_notify_cancel():
        return  # cancelled before it started
    try:
        future.set_result(fn())
    except BaseException as exc:
        future.set_exception(exc)


def _stop_tasks(tasks: queue.Queue) -> None:
    """Cancel the queued tasks and let the worker thread exit after the current one.

    The thread is never waited for: it may be stuck inside a model call, and waiting for it is
    exactly the hang the daemon thread avoids.
    """
    while True:
        try:
            task = tasks.get_nowait()
        except queue.Empty:
            break
        if task is not None:
            task[0].cancel()  # the caller gets a CancelledError instead of waiting forever
    tasks.put(None)


class _Worker:
    """One daemon thread that runs the MLX calls, one at a time.

    Not a `ThreadPoolExecutor`: `concurrent.futures.thread._python_exit` joins the pool's worker
    threads when the interpreter exits, and those threads are not daemons. A model call that
    never returns (a stuck generation) would then keep the app alive at exit. A daemon thread
    is abandoned with the process instead.
    """

    def __init__(self, name: str = "vaice-mlx"):
        self._tasks: queue.Queue = queue.Queue()
        self._stopped = False
        self._thread = threading.Thread(target=_run_tasks, args=(self._tasks,), name=name, daemon=True)
        self._thread.start()
        # The thread must not keep the worker alive, or it would never stop; when the worker is
        # gone, the thread is told to finish.
        self._finalizer = weakref.finalize(self, _stop_tasks, self._tasks)

    def submit(self, fn, *args) -> Future:
        """Run `fn(*args)` on the worker thread; the future carries the result or the exception."""
        # Read without a lock, on purpose: stop() is only ever called with nothing in flight (a
        # failed model load, or the finalizer), so it cannot race submit() today. If stop() is
        # ever called mid-reply, this check and the queue need a lock: a task enqueued after the
        # None sentinel would never be answered.
        if self._stopped:
            raise RuntimeError("the MLX worker has been stopped")
        future: Future = Future()
        self._tasks.put((future, lambda: fn(*args)))
        return future

    def stop(self) -> None:
        """Give up on the queued tasks and let the thread exit when it is done with the current one."""
        self._stopped = True
        _stop_tasks(self._tasks)


def _spoken_part(text: str, frames: int, handed_out: int) -> str:
    """The words of `text` that the first `handed_out` of `frames` audio frames cover (approximate).

    Interleaved generation emits 6 text tokens per 12 audio frames (~1 s), about twice speaking
    speed, so the text is often complete after the first second of audio. Taking the text
    generated so far would record a whole reply that was cut off early; split it by the share of
    audio handed out instead.
    """
    if not frames:  # text-only reply: nothing was said
        return ""
    if handed_out >= frames:
        return text
    words = text.split()
    # truncate, don't round: the same conservative estimate as Turn.close() — words the user had
    # no time to hear are never credited to the reply
    return " ".join(words[: int(len(words) * handed_out / frames)])


class NativeResponder:
    """Runs every MLX call on one dedicated thread.

    MLX streams are bound to the thread that created them, but fastrtc resumes our
    generator from whichever worker thread is free on each chunk. So the model is
    loaded and run on a single worker, and respond() only passes chunks back through
    a queue.
    """

    name = "native"

    def __init__(self, model, processor, prompt: str = config.NATIVE_PROMPT, worker: "_Worker | None" = None):
        self._model = model
        self._processor = processor
        self._prompt = prompt
        self._worker = worker or _Worker()

    @classmethod
    def load(cls, model_name: str = config.NATIVE_MODEL) -> "NativeResponder":
        worker = _Worker()
        try:
            model, processor = worker.submit(
                lambda: (LFM2AudioModel.from_pretrained(model_name), LFM2AudioProcessor.from_pretrained(model_name))
            ).result()
            responder = cls(model, processor, worker=worker)
            # The first generation costs ~2s extra (kernel compilation); pay it at startup, not on the first question
            # (this also compiles the ASR path, which runs after generation)
            turn = Turn()
            for _ in responder.respond((16000, np.zeros((1, 8000), dtype=np.int16)), [], turn):
                pass
        except BaseException:
            worker.stop()  # don't leave a live worker behind (auto mode falls back to cascade)
            raise
        if not turn.transcribed.wait(WARMUP_ASR_WAIT):  # the ASR warm-up; generation itself was consumed above
            # The worker may be stuck inside the ASR call: abandon it rather than hang startup here
            # (auto mode falls back to cascade; the daemon thread dies with the process if the call
            # never returns).
            worker.stop()
            raise RuntimeError(f"native warm-up transcription did not finish in {WARMUP_ASR_WAIT}s")
        return responder

    def respond(self, audio: AudioFrame, history: Sequence[Message], turn: Turn) -> Iterator[AudioFrame]:
        chunks: queue.Queue = queue.Queue()
        cancelled = threading.Event()
        progress = [("", 0)]  # (reply text, audio frames) generated so far; replaced whole by the worker
        # The queued work must not hold the responder strongly: the worker holds the queue, and the
        # queue would then hold the responder, which holds the worker - a cycle that would keep the
        # worker (and its thread) alive after the responder is gone. The generator below keeps the
        # responder alive while the reply is running, so the weak reference still resolves then.
        ref = weakref.ref(self)

        def work() -> None:
            responder = ref()
            try:
                if responder is None:  # the responder is gone: there is nothing to reply for
                    return
                for item in responder._generate(audio, history, turn, cancelled, progress):
                    chunks.put(item)
            except BaseException as exc:  # re-raised in the consumer below
                chunks.put(exc)
            finally:
                chunks.put(_DONE)  # first, so the consumer isn't held up by the transcription
            if responder is not None:
                responder._transcribe_into(audio, turn)

        future = self._worker.submit(work)
        handed_out = 0
        try:
            while True:
                try:
                    item = chunks.get(timeout=POLL_SECONDS)
                except queue.Empty:
                    if lifecycle.stopping.is_set():
                        # The app is stopping: don't wait for a model call that may never return.
                        # The worker thread is a daemon, so the process can exit with it stuck.
                        log.warning("the app is stopping; the reply is abandoned")
                        return
                    if future.cancelled():
                        # The worker was stopped before the reply started: nothing will ever arrive.
                        log.warning("the MLX worker was stopped; the reply is abandoned")
                        return
                    continue
                if item is _DONE:
                    break
                if isinstance(item, BaseException):
                    raise item
                frame, n_frames = item
                handed_out += n_frames
                turn.set_assistant_text(_spoken_part(*progress[0], handed_out))
                yield frame
            turn.set_assistant_text(_spoken_part(*progress[0], handed_out))  # all of it, now that it's all out
        finally:
            cancelled.set()  # consumer stopped early (e.g. user interrupted): free the worker

    def _chat(self, system: str, history: Sequence[Message], audio: AudioFrame) -> ChatState:
        """System prompt, earlier turns as text, then the audio question; ready for the assistant to answer.

        Only the current clip is audio: mlx-audio can't replay earlier assistant audio, and one clip
        per ChatState is what it is tested with.
        """
        sample_rate, samples = audio
        chat = ChatState(self._processor)
        for role, text in [("system", system), *history]:
            chat.new_turn(role)
            chat.add_text(text)
            chat.end_turn()
        chat.new_turn("user")
        # The processor resamples to the model's rate itself
        chat.add_audio(mx.array(to_mono_float(samples)), sample_rate=sample_rate)
        chat.end_turn()
        chat.new_turn("assistant")
        return chat

    def _text(self, token_ids: list[int]) -> str:
        return self._processor.decode_text(token_ids).strip()

    def _generate(
        self,
        audio: AudioFrame,
        history: Sequence[Message],
        turn: Turn,
        cancelled: threading.Event,
        progress: list[tuple[str, int]],
    ) -> Iterator[tuple[AudioFrame, int]]:
        """Yield (audio chunk, frames in it), keeping progress[0] = (reply text, frames) so far."""

        def stopped() -> bool:  # abandoned by the consumer, or the next turn has started
            return cancelled.is_set() or turn.closed.is_set()

        if stopped():  # abandoned while queued: skip the audio encode + prefill too
            return
        chat = self._chat(self._prompt, history, audio)

        frames: list[mx.array] = []  # every audio frame so far, for left context
        text_ids: list[int] = []
        chunk_start = 0
        n_tokens = 0
        stream = self._model.generate_from_chat_state(
            chat, mode="interleaved", max_new_tokens=MAX_NEW_TOKENS, audio_temperature=1.0, audio_top_k=4
        )
        for token, modality in stream:
            if stopped():
                return
            n_tokens += 1
            if modality == LFMModality.TEXT:
                if (token_id := int(token.item())) not in _CONTROL_TOKENS:
                    text_ids.append(token_id)
                continue
            if int(token[0].item()) == AUDIO_EOS_TOKEN:
                continue
            frames.append(token)
            if len(frames) - chunk_start == FRAMES_PER_CHUNK:
                progress[0] = (self._text(text_ids), len(frames))
                yield self._decode(frames, chunk_start), len(frames) - chunk_start
                chunk_start = len(frames)
        progress[0] = (self._text(text_ids), len(frames))
        if chunk_start < len(frames):
            yield self._decode(frames, chunk_start), len(frames) - chunk_start
        if n_tokens >= MAX_NEW_TOKENS:
            log.warning("native reply hit the %d-token limit and was cut off", MAX_NEW_TOKENS)

    def _transcribe_into(self, audio: AudioFrame, turn: Turn) -> None:
        try:
            text = self._transcribe(audio)
        except Exception:
            log.exception("transcribing the question failed; history will show it as inaudible")
            text = ""
        turn.set_user_text(text)

    def _transcribe(self, audio: AudioFrame) -> str:
        chat = self._chat(ASR_PROMPT, [], audio)
        text_ids = [
            int(token.item())
            for token, modality in self._model.generate_from_chat_state(
                chat, mode="sequential", max_new_tokens=ASR_MAX_NEW_TOKENS, temperature=None, top_k=None
            )
            if modality == LFMModality.TEXT and int(token.item()) not in _CONTROL_TOKENS
        ]
        return self._text(text_ids)

    def _decode(self, frames: list[mx.array], chunk_start: int) -> AudioFrame:
        """Decode frames[chunk_start:] with left context, returning only that span's audio."""
        context = min(LEFT_CONTEXT_FRAMES, chunk_start)
        window = frames[chunk_start - context :]
        codes = mx.stack(window, axis=1)[None, :]  # detokenizer wants (batch, codebooks, time)
        wav = np.array(self._processor.decode_audio(codes), dtype=np.float32).reshape(-1)
        return (OUTPUT_SAMPLE_RATE, wav[context * (len(wav) // len(window)) :])
