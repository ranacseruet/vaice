"""The real fastrtc ReplyOnPause, driven the way fastrtc drives it: emit() on a fresh thread per chunk
(`loop.run_in_executor`), receive() on another thread. Only the VAD model and the playback queue are fakes."""
import inspect
import threading

import numpy as np
from fastrtc import AdditionalOutputs, ReplyOnPause

from vaice.interruptible import InterruptSafeReplyOnPause, stop_playing_reply

ONE_SECOND = (48000, np.zeros((1, 48000), dtype=np.int16))  # at least AlgoOptions.audio_chunk_duration


class FakeVad:
    speech = 0.0  # seconds of speech the VAD reports for the next chunk

    def vad(self, audio, options):
        return self.speech, []

    def warmup(self):
        pass


def chunk(value):
    return (24000, np.full(10, value, dtype=np.float32))


class Rig:
    """A handler plus the helpers to speak to it, pause, and pull chunks like fastrtc does."""

    def __init__(self, handler_class, fn):
        self.vad = FakeVad()
        self.handler = handler_class(fn, model=self.vad)
        self.handler.send_message_sync = lambda msg: None  # needs a running event loop
        self.queue_clears = 0
        self.handler._clear_queue = self._clear_queue

    def _clear_queue(self):
        self.queue_clears += 1

    def user_speaks_then_pauses(self):
        """Two receive() calls from fastrtc's receive thread: speech, then silence = a detected pause."""
        for speech in (1.0, 0.0):
            self.vad.speech = speech
            run_on_thread(lambda: self.handler.receive(ONE_SECOND), "fastrtc-receive")

    def start_emit(self, name="fastrtc-emit"):
        return start_thread(self.handler.emit, name)


class Running:
    def __init__(self, target, name):
        self.result = None
        self.thread = threading.Thread(target=self._run, args=(target,), name=name)
        self.thread.start()

    def _run(self, target):
        try:
            self.result = target()
        except BaseException as exc:
            self.result = exc

    def join(self):
        self.thread.join(timeout=10)
        assert not self.thread.is_alive(), "thread is stuck"
        return self.result


def start_thread(target, name):
    return Running(target, name)


def run_on_thread(target, name):
    return Running(target, name).join()


def reply_that_blocks_on_its_second_chunk(entered, release):
    def reply(_audio):
        yield chunk(1.0)
        entered.set()  # the model is now generating chunk 2
        assert release.wait(10)
        yield chunk(2.0)
        yield chunk(3.0)

    return reply


def rig_interrupted_mid_chunk(handler_class):
    """Reply started and chunk 1 emitted; chunk 2 is being generated when the user speaks and pauses.
    Returns what the in-flight emit() returned once chunk 2 was ready."""
    entered, release = threading.Event(), threading.Event()
    rig = Rig(handler_class, reply_that_blocks_on_its_second_chunk(entered, release))
    rig.user_speaks_then_pauses()
    assert rig.start_emit().join()[1][0] == 1.0

    in_flight = rig.start_emit()
    assert entered.wait(10)
    rig.user_speaks_then_pauses()  # receive() closes the generator: it is running, so that fails
    release.set()
    return rig, in_flight.join()


def test_fastrtc_close_fails_on_a_running_generator_and_the_chunk_it_returns_is_not_dropped():
    # The bug (#2) in plain fastrtc: this is what the fix below has to work around
    rig, late_chunk = rig_interrupted_mid_chunk(ReplyOnPause)

    assert rig.handler.generator is None  # receive() let go of the generator ...
    assert late_chunk[1][0] == 2.0  # ... but the in-flight next() still handed out its chunk


def test_a_chunk_generated_while_the_user_interrupts_is_dropped():
    rig, late_chunk = rig_interrupted_mid_chunk(InterruptSafeReplyOnPause)

    assert late_chunk is None  # fastrtc skips None
    assert rig.queue_clears == 2  # fastrtc cleared the queue at both pauses, so a late chunk would have played


def test_a_chunk_generated_while_the_user_interrupts_is_dropped_even_if_the_reply_had_just_started():
    # The interrupt lands during the very first next(), so emit() itself created the generator
    entered, release = threading.Event(), threading.Event()

    def reply(_audio):
        entered.set()
        assert release.wait(10)
        yield chunk(1.0)

    rig = Rig(InterruptSafeReplyOnPause, reply)
    rig.user_speaks_then_pauses()
    in_flight = rig.start_emit()
    assert entered.wait(10)
    rig.user_speaks_then_pauses()
    release.set()

    assert in_flight.join() is None


def test_the_interrupting_question_gets_its_own_reply_untouched():
    rig, _ = rig_interrupted_mid_chunk(InterruptSafeReplyOnPause)

    # fastrtc keeps calling emit(): the next one starts a new generator for the new question
    new_reply = rig.start_emit().join()

    assert new_reply[1][0] == 1.0  # first chunk of the new reply, not dropped


def test_chunks_are_not_dropped_when_nobody_interrupts():
    rig = Rig(InterruptSafeReplyOnPause, lambda _audio: iter([chunk(1.0), chunk(2.0)]))
    rig.user_speaks_then_pauses()

    values = [rig.start_emit(f"fastrtc-emit-{i}").join()[1][0] for i in range(2)]

    assert values == [1.0, 2.0]


def test_an_interrupt_between_chunks_drops_nothing():
    # Not mid-next(): close() works and the next emit() just starts the new reply
    rig = Rig(InterruptSafeReplyOnPause, lambda _audio: iter([chunk(1.0), chunk(2.0)]))
    rig.user_speaks_then_pauses()
    assert rig.start_emit().join()[1][0] == 1.0

    rig.user_speaks_then_pauses()

    assert rig.start_emit().join()[1][0] == 1.0  # the new reply's first chunk


def test_a_transcript_update_returned_after_an_interrupt_is_kept():
    entered, release = threading.Event(), threading.Event()

    def reply(_audio):
        entered.set()
        assert release.wait(10)
        yield AdditionalOutputs("transcript")

    rig = Rig(InterruptSafeReplyOnPause, reply)
    rig.user_speaks_then_pauses()
    in_flight = rig.start_emit()
    assert entered.wait(10)
    rig.user_speaks_then_pauses()
    release.set()

    late = in_flight.join()
    assert isinstance(late, AdditionalOutputs)


def test_a_chunk_that_carries_a_transcript_update_loses_only_its_audio():
    entered, release = threading.Event(), threading.Event()
    update = AdditionalOutputs("transcript")

    def reply(_audio):
        entered.set()
        assert release.wait(10)
        yield (chunk(1.0), update)

    rig = Rig(InterruptSafeReplyOnPause, reply)
    rig.user_speaks_then_pauses()
    in_flight = rig.start_emit()
    assert entered.wait(10)
    rig.user_speaks_then_pauses()
    release.set()

    assert in_flight.join() is update


def test_copies_made_per_connection_keep_the_fix():
    original = InterruptSafeReplyOnPause(lambda _audio: iter([]), model=FakeVad(), can_interrupt=False)

    copy = original.copy()

    assert type(copy) is InterruptSafeReplyOnPause
    assert copy.fn is original.fn
    assert copy.can_interrupt is False
    assert copy.model is original.model
    assert copy is not original


def test_the_copy_passes_its_arguments_by_keyword_so_a_reordered_signature_cannot_misplace_them():
    class Reordered(InterruptSafeReplyOnPause):
        """fastrtc's arguments in a different order, as a future release might have them."""

        def __init__(
            self,
            needs_args=False,
            fn=None,
            startup_fn=None,
            algo_options=None,
            model_options=None,
            can_interrupt=True,
            expected_layout="mono",
            output_sample_rate=24000,
            output_frame_size=None,
            input_sample_rate=48000,
            model=None,
        ):
            super().__init__(
                fn=fn,
                startup_fn=startup_fn,
                algo_options=algo_options,
                model_options=model_options,
                can_interrupt=can_interrupt,
                expected_layout=expected_layout,
                output_sample_rate=output_sample_rate,
                output_frame_size=output_frame_size,
                input_sample_rate=input_sample_rate,
                model=model,
                needs_args=needs_args,
            )

    original = Reordered(fn=lambda _audio: iter([]), model=FakeVad(), can_interrupt=False, needs_args=True)

    copy = original.copy()

    assert copy.fn is original.fn
    assert copy.can_interrupt is False
    assert copy.needs_args is True
    assert copy.model is original.model


def test_the_copy_covers_every_argument_fastrtcs_reply_on_pause_takes():
    expected = set(inspect.signature(ReplyOnPause.__init__).parameters) - {"self"}
    handler = InterruptSafeReplyOnPause(lambda _audio: iter([]), model=FakeVad())

    assert set(handler._copy_kwargs()) == expected


def test_stop_playing_reply_silences_a_reply_in_flight():
    cleared = []
    entered, release = threading.Event(), threading.Event()

    def reply(_audio):
        entered.set()
        assert release.wait(10)
        yield chunk(1.0)
        yield chunk(2.0)

    rig = Rig(InterruptSafeReplyOnPause, reply)
    rig.handler._clear_queue = lambda: cleared.append(1)
    rig.user_speaks_then_pauses()
    emit = rig.start_emit()
    assert entered.wait(10)
    release.set()
    assert emit.join()[1][0] == 1.0  # the reply is playing

    stop_playing_reply(rig.handler)

    assert cleared and rig.handler.generator is None and not rig.handler.event.is_set()
    assert rig.start_emit().join() is None  # no reply restarts until the next utterance pauses


def test_stop_playing_reply_is_a_no_op_when_idle():
    cleared = []
    rig = Rig(InterruptSafeReplyOnPause, lambda _audio: iter([]))
    rig.handler._clear_queue = lambda: cleared.append(1)

    stop_playing_reply(rig.handler)  # nothing was playing: an empty queue clear is harmless

    assert rig.handler.generator is None and not rig.handler.event.is_set()
    assert rig.start_emit().join() is None


def test_stop_playing_reply_without_a_queue_or_reset_does_not_raise():
    class Bare:
        pass  # neither reset() nor a wired-up queue: an unconnected handler

    stop_playing_reply(Bare())
    stop_playing_reply(ReplyOnPause(lambda _audio: iter([]), model=FakeVad()))  # no audio track yet
