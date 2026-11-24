"""fastrtc's ReplyOnPause, minus the two barge-in races of fastrtc 0.0.34: the stale chunk that
plays after the interrupt, and the interrupting utterance lost when the reply ends mid-interrupt,
with stateful 32 ms Silero streaming VAD (to reject brief noise while tolerating mid-sentence pauses
within `VAICE_VAD_MIN_SILENCE_MS`) and pre-roll preservation to prevent clipped opening words.

It also serves an `Outbox` (outbox.py): speech queued by a worker thread outside any reply (the outcome of a call
the user approved with a click) is returned by the idle `emit()`, paced by the estimated end
of its playback, and dropped when the user starts talking."""
import asyncio
import logging
import math
import os
import threading
import time

import numpy as np
from fastrtc import ReplyOnPause
from fastrtc.pause_detection import silero
from fastrtc.pause_detection.silero import SileroVadOptions
from fastrtc.utils import create_message, split_output
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import LocalEntryNotFoundError
from scipy.signal import resample_poly

from vaice import audio as audio_helpers
from vaice import config
from vaice.outbox import Marker, Outbox, run_hook

log = logging.getLogger(__name__)

# How long an idle emit() waits for the pause event. fastrtc calls emit() in a tight loop while
# nobody is talking (utils.py: player_worker_decode), which burns most of a core (~32k calls/s,
# measured); waiting on the event instead of returning at once brings that down to <=50/s
# (37 measured).
# The wait ends as soon as the pause is detected, so it costs no latency.
IDLE_WAIT_SECONDS = 0.02

# fastrtc's default stateless 600 ms chunking inflated short noise when min_silence_duration_ms=2000, while lowering
# Silero's intra-chunk silence alone split natural mid-sentence pauses (300-600 ms) into separate turns (#64).
# With SileroVADModel, InterruptSafeReplyOnPause streams 32 ms windows across chunks, ending a turn once trailing
# silence reaches VAICE_VAD_MIN_SILENCE_MS (default 750 ms). Chunked fallback (fake/custom models) uses
# DEFAULT_VAD_OPTIONS directly. All three are env-tunable (VAICE_VAD_THRESHOLD, VAICE_VAD_MIN_SILENCE_MS,
# VAICE_VAD_SPEECH_PAD_MS).
DEFAULT_VAD_OPTIONS = SileroVadOptions(
    threshold=config.VAD_THRESHOLD,
    min_silence_duration_ms=config.VAD_MIN_SILENCE_MS,
    speech_pad_ms=config.VAD_SPEECH_PAD_MS,
)
STREAM_SAMPLE_RATE = 16000
STREAM_WINDOW_SAMPLES = 512  # Silero v5 expects 512-sample (32 ms) chunks at 16 kHz
ONSET_RESET_SECONDS = 0.096  # 3 silent windows reset an incomplete speech onset
NEG_THRESHOLD_OFFSET = 0.15  # Silero Schmitt-trigger hysteresis offset below threshold
# Tail of pre-speech audio kept before speech onset so opening consonants are not clipped.
PREROLL_SECONDS = 0.3
# Proactive speech is paced: fastrtc's playback queue is unbounded and emit() runs ~37 times a second, so serving
# the outbox as fast as it is polled would queue a whole outcome at once, and its markers would fire long before the
# audio played. The next item is served only while the audio served so far is estimated to finish less than this far
# ahead, which keeps markers (the recorded text) within this of the real playback.
LEAD_SECONDS = 0.4
# The Silero VAD model fastrtc 0.0.34 loads (`silero.SileroVADModel.download_model`), looked up in the cache first
SILERO_REPO, SILERO_FILE = "freddyaboulton/silero-vad", "silero_vad.onnx"


def use_cached_vad_model() -> None:
    """Load fastrtc's Silero VAD model from the Hugging Face cache when it is there, with no request (#143).

    fastrtc's `SileroVADModel.download_model()` calls `hf_hub_download` with no revision, which sends a `HEAD` to
    huggingface.co at every start to check for a newer one, even when the model is cached. Wrapped here, the cached
    copy is used as it is and only a missing one is downloaded (fastrtc's own call, the first start). Setting
    `HF_HUB_OFFLINE` from vaice would not do it: huggingface_hub reads it once, when it is imported, which gradio and
    fastrtc do before any vaice code runs. A user who set `HF_HUB_OFFLINE` (either way) gets fastrtc's loader as it is.
    Call before the first `ReplyOnPause` is built (it loads the model); calling it again changes nothing.
    """
    if "HF_HUB_OFFLINE" in os.environ:
        return
    model_class = getattr(silero, "SileroVADModel", None)
    download = getattr(model_class, "download_model", None)
    if download is None:  # a fastrtc upgrade moved the loader: its own check then runs, one request at start
        log.warning("cannot use the cached VAD model: fastrtc's SileroVADModel has no download_model()")
        return
    if getattr(download, "cached_first", False):
        return

    def download_model() -> str:
        try:
            return hf_hub_download(repo_id=SILERO_REPO, filename=SILERO_FILE, local_files_only=True)
        except LocalEntryNotFoundError:  # not downloaded yet: fastrtc fetches it
            return download()

    download_model.cached_first = True  # type: ignore[attr-defined]
    model_class.download_model = staticmethod(download_model)


def _clear_playback(handler) -> None:
    """Drop the audio fastrtc queued for playback on `handler`'s connection; a no-op before its audio track starts.

    fastrtc's `AudioCallback.clear_queue()` (wired up as `_clear_queue`) empties an `asyncio.Queue` and resets the
    playback clock (`_start`). The connection's event loop fills that queue (`player_worker_decode`) and drains it
    (`recv`), and neither is thread-safe, but the clear is asked for from other threads: fastrtc's barge-in in
    `receive()` (a worker thread), the drop of proactive speech there, and the "New conversation" button (a Gradio
    thread), #67. So it runs on the connection's loop (`_loop`, which fastrtc sets before it wires `_clear_queue`):
    directly when already there, else scheduled to run there next, without waiting: the loop never blocks on the
    caller, and nothing the caller does after it depends on the queue being empty yet. A handler with no loop
    (fastrtc's websocket path sets one too; only tests' rigs lack it) is cleared on the calling thread.
    """
    clear = getattr(handler, "_clear_queue", None)
    if clear is None:
        return
    loop = getattr(handler, "_loop", None)
    if loop is None or _running_loop() is loop:
        clear()
        return
    try:
        loop.call_soon_threadsafe(clear)
    except RuntimeError:  # the connection's loop is closed: the connection is gone and nothing plays
        log.debug("playback queue not cleared: its event loop is closed")


def _running_loop() -> asyncio.AbstractEventLoop | None:
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        return None


class _StreamingVad:
    """Per-utterance stateful 32 ms Silero VAD attached to `AppState` (reset automatically on `reset()`)."""

    def __init__(self, silero_state):
        self.silero_state = silero_state
        self.rate = 0
        self.raw_pending: np.ndarray | None = None
        self.buf16 = np.empty(0, dtype=np.float32)
        self.preroll_ring: np.ndarray | None = None
        self.onset_speech = 0.0
        self.onset_gap = 0.0
        self.in_speech = False
        self.silence_s = 0.0

    def _to_16k(self, raw_mono: np.ndarray, sampling_rate: int) -> None:
        if self.rate != sampling_rate:
            self.rate = sampling_rate
            self.raw_pending = None
        samples = audio_helpers.to_mono_float(raw_mono)
        self.raw_pending = samples if self.raw_pending is None else np.concatenate((self.raw_pending, samples))
        if sampling_rate == STREAM_SAMPLE_RATE:
            self.buf16 = np.concatenate((self.buf16, self.raw_pending))
            self.raw_pending = None
            return
        divisor = math.gcd(STREAM_SAMPLE_RATE, sampling_rate)
        up = STREAM_SAMPLE_RATE // divisor
        down = sampling_rate // divisor
        min_in = max(down, math.ceil(STREAM_WINDOW_SAMPLES * down / up))
        if len(self.raw_pending) < min_in:
            return
        take = (len(self.raw_pending) // down) * down
        resampled = resample_poly(self.raw_pending[:take], up, down, padtype="edge").astype(np.float32)
        self.buf16 = np.concatenate((self.buf16, resampled))
        rest = self.raw_pending[take:]
        self.raw_pending = rest if len(rest) > 0 else None

    def feed(
        self,
        audio: np.ndarray,
        sampling_rate: int,
        *,
        model,
        threshold: float,
        min_silence_s: float,
        started_talking_s: float,
        preroll_s: float,
        max_continuous_speech_s: float,
        state,
    ) -> tuple[bool, bool]:
        """Process one incoming audio chunk; returns `(just_started_talking, pause_detected)`."""
        raw = np.atleast_1d(np.squeeze(audio))
        if raw.size == 0 or sampling_rate <= 0:
            return False, False
        raw_mono = raw.mean(axis=0).astype(raw.dtype) if raw.ndim > 1 else raw

        if not state.started_talking:
            self.preroll_ring = raw_mono if self.preroll_ring is None else np.concatenate((self.preroll_ring, raw_mono))
            cap = max(
                len(raw_mono),
                int(sampling_rate * (started_talking_s + ONSET_RESET_SECONDS + preroll_s + 0.064)),
            )
            if cap > 0 and len(self.preroll_ring) > cap:
                self.preroll_ring = self.preroll_ring[-cap:]
        else:
            state.stream = raw_mono if state.stream is None else np.concatenate((state.stream, raw_mono))

        self._to_16k(raw_mono, sampling_rate)
        dt = STREAM_WINDOW_SAMPLES / STREAM_SAMPLE_RATE
        neg_th = max(0.0, threshold - NEG_THRESHOLD_OFFSET)
        just_started = False
        pause_detected = False
        offset = 0
        while offset + STREAM_WINDOW_SAMPLES <= len(self.buf16):
            window = self.buf16[offset : offset + STREAM_WINDOW_SAMPLES]
            offset += STREAM_WINDOW_SAMPLES
            prob, self.silero_state = model(window, self.silero_state, STREAM_SAMPLE_RATE)
            p = float(np.asarray(prob).flat[0])
            if not state.started_talking:
                if p >= threshold:
                    self.onset_speech += dt
                    self.onset_gap = 0.0
                    if self.onset_speech > started_talking_s:
                        state.started_talking = True
                        just_started = True
                        state.stream = self.preroll_ring
                        self.preroll_ring = None
                        self.in_speech = True
                        self.silence_s = 0.0
                elif p < neg_th:
                    self.onset_gap += dt
                    if self.onset_gap >= ONSET_RESET_SECONDS:
                        self.onset_speech = 0.0
                        self.onset_gap = 0.0
            else:
                if p >= threshold:
                    self.in_speech = True
                    self.silence_s = 0.0
                elif p < neg_th:
                    self.in_speech = False
                    self.silence_s += dt
                elif self.in_speech:
                    self.silence_s = 0.0
                else:
                    self.silence_s += dt
                if not self.in_speech and self.silence_s >= min_silence_s:
                    pause_detected = True
                    break
        if offset:
            self.buf16 = self.buf16[offset:]

        if (
            not pause_detected
            and state.started_talking
            and state.stream is not None
            and len(state.stream) / sampling_rate >= max_continuous_speech_s
        ):
            pause_detected = True
        return just_started, pause_detected


class InterruptSafeReplyOnPause(ReplyOnPause):
    """Drops the audio chunk that was being generated when the user interrupted the reply.

    fastrtc 0.0.34 (https://github.com/ranacseruet/vaice/issues/2): `receive()` closes the reply
    generator and clears the playback queue, but `emit()` is usually inside `next(generator)` on
    another thread, so `close()` fails ("generator already executing", logged at debug) and that
    `next()` then returns a normal chunk, which is queued after the clear and plays. The reply
    itself does stop, once CPython finalizes the dropped generator.

    `receive()` sets `self.generator = None` after an interrupt, so an `emit()` that started with
    another generator (or created its own) and finds none, or a different one, was interrupted
    while it ran. It returns None instead of the stale audio, which fastrtc skips. Tiny window
    left: an interrupt landing after that check but before fastrtc queues the chunk still plays it.

    It also keeps the interrupting utterance when the reply ends in the very moment the user
    interrupts (#13): `super().emit()` calls `reset()` when the generator runs out, which clears the
    pause event and replaces the state — and a barge-in pause that `receive()` detected during that
    call dies with them, so the interrupting question never gets a reply. The pause and the audio
    `receive()` accumulated are restored in that case — only for this end-of-reply cleanup: an
    explicit reset (the "New conversation" button, `clean_up`) is never undone, and audio `receive()`
    has begun collecting on the fresh state always wins, the restore sharing a lock with
    `receive()`. A reply that *errors* re-raises before this
    can happen and its barge-in is lost as in plain fastrtc; `app.py` catches errors inside the
    generator, so replies normally end by running out.

    It also defaults to tuned VAD options (`DEFAULT_VAD_OPTIONS`) and streams 32 ms Silero windows
    across chunks so brief room noises are rejected while natural mid-sentence pauses shorter than
    `VAICE_VAD_MIN_SILENCE_MS` do not split turns, keeping a 0.3 s pre-roll buffer before speech
    onset to avoid clipping opening words.

    It also throttles the idle polling: fastrtc calls `emit()` in a tight loop while nobody is
    talking, which burns most of a core. The idle `emit()` waits for the pause event instead of
    returning at once (the wait ends as soon as the pause is detected, so no latency is added).

    And it speaks when no reply is in flight: with an `outbox` (kept by `copy()`, so every connection drains the
    same one), an idle `emit()` — no pause pending, no reply generator, the user not talking — returns the next audio
    frame from it and runs the markers it reaches, but only while the audio served so far is estimated to finish less
    than `LEAD_SECONDS` ahead (`_proactive_until`; served is not played, see above), so a marker fires about when the
    audio before it has played. When the user starts talking with no reply running (`receive()`, and only with
    `can_interrupt`, as on the reply path), the outbox is cleared (the markers' `on_drop`s run) and, if proactive
    audio is estimated still playing, fastrtc's playback queue is cleared too. "New conversation" clears it too.
    """

    def __init__(self, *args, outbox: Outbox | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.outbox = outbox
        self._clock = time.monotonic  # tests substitute a fake
        self._proactive_lock = threading.Lock()  # emit() runs on a different worker thread per call
        self._proactive_until = 0.0  # when the proactive audio served so far is estimated to finish playing
        if self.model_options is None:
            self.model_options = DEFAULT_VAD_OPTIONS
        if not hasattr(self, "generator"):  # the attribute the fix relies on; a fastrtc upgrade may drop it
            raise RuntimeError(
                "fastrtc's ReplyOnPause has no `generator` attribute: InterruptSafeReplyOnPause needs updating"
            )
        if not hasattr(self, "event"):  # the pause event the idle wait relies on
            raise RuntimeError(
                "fastrtc's ReplyOnPause has no `event` attribute: InterruptSafeReplyOnPause needs updating"
            )
        # #13 machinery. The restore must tell the end-of-reply cleanup — super().emit() calling
        # reset() on the emit thread, inside the emit call — from an explicit reset (the "New
        # conversation" button's stop_playing_reply, fastrtc's clean_up): an explicit reset is the
        # user asking to clear everything and is never undone. The counters let an emit see which
        # kind ran while it was in flight. The lock serializes the restore with receive(), so a
        # newer utterance being collected is never replaced under its own frame-processing.
        self._state_lock = threading.Lock()
        self._emit_thread: threading.Thread | None = None
        self._own_resets = 0  # end-of-reply resets, counted on the emit thread
        self._foreign_resets = 0  # explicit resets, counted on any other thread

    @property
    def clear_queue(self):
        # fastrtc's barge-in in ReplyOnPause.receive() calls self.clear_queue() on its receive thread: run it on the
        # connection's loop like every other clear (#67)
        return lambda: _clear_playback(self)

    def reset(self):
        # reached by every reset on this handler, including super().emit()'s end-of-reply one
        # (it calls self.reset(), so the override stands). An executor reuses its threads, but a
        # stale _emit_thread cannot mislabel a concurrent reset: during an emit call no other code
        # runs on that thread, and the next emit re-snapshots the counters.
        with self._state_lock:
            if threading.current_thread() is self._emit_thread:
                self._own_resets += 1
            else:
                self._foreign_resets += 1
        super().reset()

    def _uses_streaming_vad(self) -> bool:
        return isinstance(self.model, silero.SileroVADModel)

    def determine_pause(self, audio: np.ndarray, sampling_rate: int, state) -> bool:
        if not self._uses_streaming_vad():
            return super().determine_pause(audio, sampling_rate, state)
        state.buffer = None
        stream_vad = getattr(state, "_stream_vad", None)
        if stream_vad is None:
            stream_vad = _StreamingVad(self.model.get_initial_state(1))
            state._stream_vad = stream_vad
        opts = self.model_options or DEFAULT_VAD_OPTIONS
        just_started, pause_detected = stream_vad.feed(
            audio,
            sampling_rate,
            model=self.model,
            threshold=opts.threshold,
            min_silence_s=opts.min_silence_duration_ms / 1000.0,
            started_talking_s=self.algo_options.started_talking_threshold,
            preroll_s=PREROLL_SECONDS,
            max_continuous_speech_s=self.algo_options.max_continuous_speech_s,
            state=state,
        )
        if just_started:
            log.debug("Started talking")
            self.send_message_sync(create_message("log", "started_talking"))
        return pause_detected

    def receive(self, frame):
        # one frame is one atomic step against the #13 restore in emit(): the state the restore
        # reads or replaces is never a half-processed frame. Holding the lock across the VAD work
        # costs emit() at most one frame of waiting, after its idle wait and reply are done.
        speech_started = over_reply = False
        with self._state_lock:
            state = self.state
            was_talking = state.started_talking
            buf_before = state.buffer
            super().receive(frame)
            if not was_talking and not (state.responding and not self.can_interrupt):
                if state.started_talking:
                    # no reply generator: the user talks over proactive speech, not over a reply (whose own
                    # barge-in below handles it); with can_interrupt off speech never clears what is playing
                    speech_started = self.can_interrupt
                    over_reply = self.generator is not None
                    preroll = getattr(state, "_preroll", None)
                    if preroll is not None and state.stream is not None:
                        state.stream = np.concatenate((preroll, state.stream))
                        state._preroll = None
                elif state.buffer is None and not self._uses_streaming_vad():
                    # determine_pause() just evaluated and discarded a full pre-speech chunk;
                    # keep its tail so speech starting near the chunk boundary isn't clipped.
                    rate, array = frame
                    array = np.squeeze(array)
                    chunk = array if buf_before is None else np.concatenate((buf_before, array))
                    keep = int((state.sampling_rate or rate) * PREROLL_SECONDS)
                    state._preroll = chunk[-keep:] if keep > 0 else None
        if speech_started and over_reply:  # fastrtc drops the reply's queued audio: it will never play
            with self._proactive_lock:
                self._proactive_until = 0.0
        elif speech_started:  # after the lock: the drop hooks may take locks of their own
            self._drop_proactive_speech(clear_playing_audio=True)

    def copy(self):
        """fastrtc copies the handler for each connection, and ReplyOnPause.copy() builds a plain ReplyOnPause.

        The arguments go by keyword: a fastrtc release that adds, drops or reorders a parameter
        then fails loudly here instead of silently handing the handler the wrong value.
        """
        handler = type(self)(**self._copy_kwargs())
        handler.outbox = self.outbox  # not one of fastrtc's arguments: set here, so `_copy_kwargs` stays theirs
        return handler

    def _copy_kwargs(self) -> dict:
        """The arguments `ReplyOnPause.__init__` takes, taken from this handler."""
        return {
            "fn": self.fn,
            "startup_fn": self.startup_fn,
            "algo_options": self.algo_options,
            "model_options": self.model_options,
            "can_interrupt": self.can_interrupt,
            "expected_layout": self.expected_layout,
            "output_sample_rate": self.output_sample_rate,
            "output_frame_size": self.output_frame_size,
            "input_sample_rate": self.input_sample_rate,
            "model": self.model,
            "needs_args": self.needs_args,
        }

    def _serve_outbox(self):
        """The next proactive audio frame, or None. Only when idle: no reply generator, the user not talking (a
        pause pending is the caller's `event` check). Markers reached on the way run here, on the emit thread."""
        outbox = self.outbox
        if outbox is None or self.generator is not None or self.state.started_talking:
            return None
        while True:
            now = self._clock()
            with self._proactive_lock:
                allowed = self._proactive_until - now < LEAD_SECONDS  # the audio served so far is nearly played
            item = outbox.poll(allowed)
            if item is None:
                return None
            if isinstance(item, Marker):
                run_hook(item.callback, "marker")
                continue
            rate, samples = item
            with self._proactive_lock:
                self._proactive_until = max(self._proactive_until, now) + len(samples) / rate
            return item

    def _drop_proactive_speech(self, clear_playing_audio: bool) -> None:
        """Forget what is queued for proactive speech (the markers' `on_drop`s run). `clear_playing_audio`: also
        drop what fastrtc queued for playback, when proactive audio is estimated still playing (the user started
        talking), the same queue clear fastrtc's own barge-in does."""
        if self.outbox is None:
            return
        self.outbox.clear()
        with self._proactive_lock:
            playing = self._clock() < self._proactive_until
            self._proactive_until = 0.0
        if clear_playing_audio and playing:
            _clear_playback(self)

    def emit(self):
        if self.outbox is not None:
            self.outbox.touch()  # a connection is polling: what is queued is not stale, even during a long reply
        if not self.event.is_set():
            # Nobody is talking: fastrtc calls emit() in a tight loop (utils.py:
            # player_worker_decode), which burns most of a core (~32k calls/s, measured). Waiting
            # for the pause event instead of returning at once throttles that to <=50/s (37
            # measured); the
            # wait ends as soon as the pause is detected, so the reply still starts right away.
            self.event.wait(timeout=IDLE_WAIT_SECONDS)
            if not self.event.is_set():
                return self._serve_outbox()  # still nobody talking: proactive speech, else None (fastrtc skips it)
        started_with = self.generator
        with self._state_lock:
            self._emit_thread = threading.current_thread()
            own0, foreign0 = self._own_resets, self._foreign_resets
            state_before = self.state  # the object receive() accumulates the interrupting utterance into
        output = super().emit()
        # #13: when this call's generator ran out, super() called reset(): the event is cleared and a
        # fresh AppState replaces the state. If receive() detected the barge-in pause during that call,
        # the reset wiped the pause and the audio with it, and the question is lost. state_before is
        # the object receive() updated, so restore both — never when: this emit began with no generator
        # attached (super() would have begun the reply from that pause itself; re-arming an exhausted
        # one lets an empty reply loop forever), an explicit reset ran meanwhile ("New conversation",
        # clean_up — the user asked to clear everything), a newer pause was detected after the reset,
        # or receive() already began collecting a newer utterance on the fresh state (its audio wins).
        with self._state_lock:
            if (
                started_with is not None
                and self._own_resets > own0  # end-of-reply cleanup ran during this call
                and self._foreign_resets == foreign0  # no explicit reset ran alongside it
                and not self.event.is_set()  # no newer pause was detected after the reset
                and self.state.stream is None  # no newer audio is being collected ...
                and self.state.buffer is None  # ... on the fresh state
                and getattr(self.state, "_stream_vad", None) is None
                and state_before.pause_detected  # receive() had detected the barge-in pause on it
            ):
                self.state = state_before
                self.event.set()
                log.debug("restored the barge-in pause the ending reply's reset() wiped")
        interrupted = self.generator is None or (started_with is not None and self.generator is not started_with)
        if output is None:
            return output
        audio, extra = split_output(output)
        if interrupted and audio is not None:
            log.debug("dropped a chunk of the interrupted reply")
            return extra
        if audio is not None:
            self._note_playing(audio)
        return output

    def _note_playing(self, audio) -> None:
        """Reply audio just handed to fastrtc plays before anything served after it: count it in the estimated playback
        end that paces proactive speech, so an outcome queued while a question is still playing waits behind it."""
        if self.outbox is None:
            return
        try:
            rate, samples = audio
            seconds = np.size(samples) / rate
        except (TypeError, ValueError, ZeroDivisionError):  # not an audio tuple: nothing to count
            return
        now = self._clock()
        with self._proactive_lock:
            self._proactive_until = max(self._proactive_until, now) + seconds


def stop_playing_reply(handler: ReplyOnPause) -> None:
    """Silence `handler`'s reply in flight, audio and all; safe (and a no-op) if none is playing.

    What receive() does on a barge-in, without waiting for the user to stop talking — for the
    "New conversation" button, which runs on a Gradio thread like fastrtc's own clean_up does:
    reset() drops the reply generator (a chunk it is still generating is dropped by emit() above,
    as on a barge-in) and clears the pause event, so the next reply only starts after the next
    utterance; the queue clear drops the audio fastrtc had already queued for playback (on the connection's event
    loop, `_clear_playback`).
    """
    if not hasattr(handler, "reset"):  # the internals this relies on; a fastrtc upgrade may restructure them
        log.warning("cannot stop the reply: fastrtc's ReplyOnPause has no reset()")
        return
    handler.reset()
    drop_proactive = getattr(handler, "_drop_proactive_speech", None)
    if drop_proactive is not None:  # speech queued with no reply (a click's outcome) belongs to the old conversation
        drop_proactive(clear_playing_audio=False)
    _clear_playback(handler)
