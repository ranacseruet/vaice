"""fastrtc's ReplyOnPause, minus the two barge-in races of fastrtc 0.0.34: the stale chunk that
plays after the interrupt, and the interrupting utterance lost when the reply ends mid-interrupt,
with tuned VAD options to reject brief room noise and pre-roll preservation to prevent clipped opening words."""
import logging
import threading

import numpy as np
from fastrtc import ReplyOnPause
from fastrtc.pause_detection.silero import SileroVadOptions
from fastrtc.utils import split_output

log = logging.getLogger(__name__)

# How long an idle emit() waits for the pause event. fastrtc calls emit() in a tight loop while
# nobody is talking (utils.py: player_worker_decode), which burns most of a core (~32k calls/s,
# measured); waiting on the event instead of returning at once brings that down to <=50/s
# (37 measured).
# The wait ends as soon as the pause is detected, so it costs no latency.
IDLE_WAIT_SECONDS = 0.02

# fastrtc's default SileroVadOptions uses min_silence_duration_ms=2000 and speech_pad_ms=400 inside
# 600 ms chunks (AlgoOptions.audio_chunk_duration=0.6), so Silero never closes a triggered segment
# inside a chunk and counts trailing silence as speech — inflating a 180 ms noise into 600 ms of
# speech. Closing segments after 150 ms of silence with 30 ms padding and a 0.6 probability
# threshold rejects brief noises while keeping normal speech detection intact.
DEFAULT_VAD_OPTIONS = SileroVadOptions(
    threshold=0.6,
    min_silence_duration_ms=150,
    speech_pad_ms=30,
)
# Tail of the last pre-speech chunk kept and prepended when `started_talking` flips to True, so an
# utterance that begins in the last ~250 ms of a 600 ms VAD chunk does not lose its opening words.
PREROLL_SECONDS = 0.3


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

    It also defaults to tuned VAD options (`DEFAULT_VAD_OPTIONS`) so brief room noises are not
    inflated into 600 ms speech segments, and keeps a 0.3 s pre-roll tail from the preceding
    pre-speech chunk to avoid clipping words spoken near chunk boundaries.

    It also throttles the idle polling: fastrtc calls `emit()` in a tight loop while nobody is
    talking, which burns most of a core. The idle `emit()` waits for the pause event instead of
    returning at once (the wait ends as soon as the pause is detected, so no latency is added).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
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

    def receive(self, frame):
        # one frame is one atomic step against the #13 restore in emit(): the state the restore
        # reads or replaces is never a half-processed frame. Holding the lock across the VAD work
        # costs emit() at most one frame of waiting, after its idle wait and reply are done.
        with self._state_lock:
            state = self.state
            was_talking = state.started_talking
            buf_before = state.buffer
            super().receive(frame)
            if not was_talking and not (state.responding and not self.can_interrupt):
                if state.started_talking:
                    preroll = getattr(state, "_preroll", None)
                    if preroll is not None and state.stream is not None:
                        state.stream = np.concatenate((preroll, state.stream))
                        state._preroll = None
                elif state.buffer is None:
                    # determine_pause() just evaluated and discarded a full pre-speech chunk;
                    # keep its tail so speech starting near the chunk boundary isn't clipped.
                    rate, array = frame
                    array = np.squeeze(array)
                    chunk = array if buf_before is None else np.concatenate((buf_before, array))
                    keep = int((state.sampling_rate or rate) * PREROLL_SECONDS)
                    state._preroll = chunk[-keep:] if keep > 0 else None

    def copy(self):
        """fastrtc copies the handler for each connection, and ReplyOnPause.copy() builds a plain ReplyOnPause.

        The arguments go by keyword: a fastrtc release that adds, drops or reorders a parameter
        then fails loudly here instead of silently handing the handler the wrong value.
        """
        return type(self)(**self._copy_kwargs())

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

    def emit(self):
        if not self.event.is_set():
            # Nobody is talking: fastrtc calls emit() in a tight loop (utils.py:
            # player_worker_decode), which burns most of a core (~32k calls/s, measured). Waiting
            # for the pause event instead of returning at once throttles that to <=50/s (37
            # measured); the
            # wait ends as soon as the pause is detected, so the reply still starts right away.
            self.event.wait(timeout=IDLE_WAIT_SECONDS)
            if not self.event.is_set():
                return None  # still nobody talking: fastrtc skips a None
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
                and state_before.pause_detected  # receive() had detected the barge-in pause on it
            ):
                self.state = state_before
                self.event.set()
                log.debug("restored the barge-in pause the ending reply's reset() wiped")
        interrupted = self.generator is None or (started_with is not None and self.generator is not started_with)
        if output is None or not interrupted:
            return output
        audio, extra = split_output(output)
        if audio is None:
            return output
        log.debug("dropped a chunk of the interrupted reply")
        return extra


def stop_playing_reply(handler: ReplyOnPause) -> None:
    """Silence `handler`'s reply in flight, audio and all; safe (and a no-op) if none is playing.

    What receive() does on a barge-in, without waiting for the user to stop talking — for the
    "New conversation" button, which runs on a Gradio thread like fastrtc's own clean_up does:
    reset() drops the reply generator (a chunk it is still generating is dropped by emit() above,
    as on a barge-in) and clears the pause event, so the next reply only starts after the next
    utterance; the queue clear drops the audio fastrtc had already queued for playback.
    """
    if not hasattr(handler, "reset"):  # the internals this relies on; a fastrtc upgrade may restructure them
        log.warning("cannot stop the reply: fastrtc's ReplyOnPause has no reset()")
        return
    handler.reset()
    clear_queue = getattr(handler, "_clear_queue", None)  # wired up when the audio track starts
    if clear_queue is not None:
        clear_queue()
