"""fastrtc's ReplyOnPause, minus the stale chunk that plays after a barge-in."""
import logging

from fastrtc import ReplyOnPause
from fastrtc.utils import split_output

log = logging.getLogger(__name__)


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
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not hasattr(self, "generator"):  # the attribute the fix relies on; a fastrtc upgrade may drop it
            raise RuntimeError("fastrtc's ReplyOnPause has no `generator` attribute: InterruptSafeReplyOnPause needs updating")

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
        started_with = self.generator
        output = super().emit()
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
