"""The real fastrtc ReplyOnPause, driven the way fastrtc drives it: emit() on a fresh thread per chunk
(`loop.run_in_executor`), receive() on another thread. Only the VAD model and the playback queue are fakes."""
import inspect
import threading
import time

import numpy as np
from fastrtc import AdditionalOutputs, ReplyOnPause
from fastrtc.utils import WebRTCError, webrtc_error_handler

from vaice import interruptible
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


def rig_interrupted_as_the_reply_ends(handler_class):
    """Issue #13 harness: chunk 1 was emitted and the last next() — the one that will exhaust the
    generator — is in flight when the user speaks and pauses.
    Returns the rig, what the in-flight emit() returned, and the audio each reply call received."""
    entered, release = threading.Event(), threading.Event()
    heard = []

    def reply(audio):
        heard.append(audio)
        yield chunk(1.0)
        entered.set()  # the model is generating its final chunk: releasing ends the generator
        assert release.wait(10)

    rig = Rig(handler_class, reply)
    rig.user_speaks_then_pauses()
    assert rig.start_emit().join()[1][0] == 1.0

    in_flight = rig.start_emit()
    assert entered.wait(10)
    rig.user_speaks_then_pauses()  # the barge-in lands while that next() is in flight
    release.set()
    return rig, in_flight.join(), heard


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


def test_fastrtc_loses_the_interrupting_utterance_when_the_reply_ends_mid_next():
    # The bug (#13) in plain fastrtc: the exhausted generator's StopIteration makes emit() call
    # reset(), which wipes the barge-in pause receive() had just detected — so nothing answers it
    rig, late, _ = rig_interrupted_as_the_reply_ends(ReplyOnPause)

    assert late is None
    assert not rig.handler.event.is_set()  # reset() cleared the pause ...
    assert rig.handler.state.stream is None  # ... and the interrupting audio went with the old state
    assert rig.start_emit().join() is None  # the interrupting question never gets a reply


def test_the_interrupting_utterance_survives_a_reply_that_ends_mid_next():
    rig, late, heard = rig_interrupted_as_the_reply_ends(InterruptSafeReplyOnPause)

    assert late is None  # the ending reply hands out nothing more
    assert rig.handler.event.is_set()  # the barge-in pause was restored, not wiped by reset()
    assert rig.handler.state.stream is not None  # the interrupting audio is kept for the new reply
    assert rig.queue_clears == 2  # receive() cleared the queue at both pauses

    new_reply = rig.start_emit().join()

    assert new_reply[1][0] == 1.0  # the interrupting question gets its own reply
    assert len(heard) == 2  # fn was called again, for the interrupting utterance
    assert heard[1][1].shape[-1] > 0  # its audio reached the reply: the accumulated stream, not an empty buffer


def test_a_newer_pause_that_arrived_during_the_reset_is_never_replaced_by_the_old_one():
    # The `not event.is_set()` guard: when receive() detects a newer pause between the ending
    # reply's own reset() and the wrapper's check, the newer utterance must win — emulated by
    # running the newer utterance's receive() right behind the reset on the emit thread
    entered, release = threading.Event(), threading.Event()

    def reply(_audio):
        yield chunk(1.0)
        entered.set()  # the in-flight next() will end the generator once released
        assert release.wait(10)

    rig = Rig(InterruptSafeReplyOnPause, reply)
    rig.user_speaks_then_pauses()
    assert rig.start_emit().join()[1][0] == 1.0

    in_flight = rig.start_emit()
    assert entered.wait(10)
    rig.user_speaks_then_pauses()  # the #13 barge-in: a pause on the reply's state

    newer = []
    reset = rig.handler.reset

    def reset_then_a_newer_pause():  # instance attribute: super().emit() calls self.reset()
        reset()  # the ending reply's reset: event cleared, fresh state
        newer.append(rig.handler.state)
        rig.user_speaks_then_pauses()  # a new utterance detected before the wrapper's check

    rig.handler.reset = reset_then_a_newer_pause
    release.set()

    assert in_flight.join() is None
    assert rig.handler.state is newer[0]  # the newer utterance's state survived, not the older pause
    assert rig.handler.event.is_set()
    assert rig.start_emit().join()[1][0] == 1.0  # and the newer utterance gets the reply


def test_the_new_conversation_button_during_the_last_next_is_never_undone_by_the_restore():
    # PR #37 review P1: the button resets the handler and clears the conversation; that explicit
    # reset must not be mistaken for the end-of-reply cleanup, or the pre-click question would be
    # answered as the new conversation's first turn
    entered, release = threading.Event(), threading.Event()

    def reply(_audio):
        yield chunk(1.0)
        entered.set()  # the in-flight next() will end the generator once released
        assert release.wait(10)

    rig = Rig(InterruptSafeReplyOnPause, reply)
    rig.user_speaks_then_pauses()
    assert rig.start_emit().join()[1][0] == 1.0

    in_flight = rig.start_emit()
    assert entered.wait(10)
    rig.user_speaks_then_pauses()  # the #13 barge-in pause lands on the reply's state
    stop_playing_reply(rig.handler)  # the click, on a Gradio thread, while the last next() runs
    release.set()

    assert in_flight.join() is None
    assert not rig.handler.event.is_set()  # the end-of-reply cleanup did not re-arm the pause
    assert rig.handler.state.stream is None  # the clicked-away utterance stays gone
    assert rig.start_emit().join() is None  # nothing answers it: the next reply waits for speech


def test_a_newer_utterance_still_being_collected_is_never_replaced_by_the_restore():
    # PR #37 review P2: after the reset, receive() can be mid-utterance (audio collected, no pause
    # yet, so the event is still unset) before the check runs; the newer audio must win. The hook
    # runs that frame right behind the reset on the emit thread, as the interleaving's worst case;
    # the lock in receive() is what keeps the real concurrency frame-atomic.
    entered, release = threading.Event(), threading.Event()

    def reply(_audio):
        yield chunk(1.0)
        entered.set()  # the in-flight next() will end the generator once released
        assert release.wait(10)

    rig = Rig(InterruptSafeReplyOnPause, reply)
    rig.user_speaks_then_pauses()
    assert rig.start_emit().join()[1][0] == 1.0

    in_flight = rig.start_emit()
    assert entered.wait(10)
    rig.user_speaks_then_pauses()  # the #13 barge-in pause lands on the reply's state

    newer = []
    reset = rig.handler.reset

    def reset_then_speech():  # instance attribute: super().emit() calls self.reset()
        reset()  # the ending reply's reset: event cleared, fresh state
        newer.append(rig.handler.state)
        rig.vad.speech = 1.0  # a speech frame of a new utterance: collected, no pause yet
        run_on_thread(lambda: rig.handler.receive(ONE_SECOND), "fastrtc-receive")

    rig.handler.reset = reset_then_speech
    release.set()

    assert in_flight.join() is None
    assert rig.handler.state is newer[0]  # the newer partial audio survived, not the old pause
    assert newer[0].started_talking and newer[0].stream is not None
    assert not rig.handler.event.is_set()  # and nothing fired early

    rig.vad.speech = 0.0  # the newer utterance pauses normally ...
    run_on_thread(lambda: rig.handler.receive(ONE_SECOND), "fastrtc-receive")
    assert rig.start_emit().join()[1][0] == 1.0  # ... and gets its reply


def test_audio_arriving_right_after_the_reset_wins_too():
    # The sub-threshold half of the same protection: a frame shorter than the VAD chunk duration
    # only accumulates into AppState.buffer (no started_talking, no stream) — still newer audio
    entered, release = threading.Event(), threading.Event()

    def reply(_audio):
        yield chunk(1.0)
        entered.set()
        assert release.wait(10)

    rig = Rig(InterruptSafeReplyOnPause, reply)
    rig.user_speaks_then_pauses()
    assert rig.start_emit().join()[1][0] == 1.0

    in_flight = rig.start_emit()
    assert entered.wait(10)
    rig.user_speaks_then_pauses()

    newer = []
    reset = rig.handler.reset

    def reset_then_a_short_frame():  # instance attribute: super().emit() calls self.reset()
        reset()
        newer.append(rig.handler.state)
        short = (48000, np.zeros((1, 4800), dtype=np.int16))  # 0.1 s < audio_chunk_duration
        run_on_thread(lambda: rig.handler.receive(short), "fastrtc-receive")

    rig.handler.reset = reset_then_a_short_frame
    release.set()

    assert in_flight.join() is None
    assert rig.handler.state is newer[0]  # the collected buffer survived, not the old pause
    assert newer[0].buffer is not None and newer[0].stream is None
    assert not rig.handler.event.is_set()


def test_a_reply_that_ends_on_its_own_keeps_the_next_reply_for_the_next_utterance():
    # The #13 restore must not fire without a barge-in: a naturally ending reply leaves no pause
    rig = Rig(InterruptSafeReplyOnPause, lambda _audio: iter([chunk(1.0)]))
    rig.user_speaks_then_pauses()
    assert rig.start_emit().join()[1][0] == 1.0

    assert rig.start_emit().join() is None  # the generator runs out: reset(), nothing to restore

    assert not rig.handler.event.is_set()
    assert rig.start_emit().join() is None  # and the next reply still waits for the next utterance


def test_an_exhausted_reply_started_from_the_pause_does_not_put_the_pause_back():
    # The restore must not re-arm a pause whose own reply already ran out: an fn that answers with
    # nothing would otherwise be restarted every emit() forever
    calls = []

    def empty_reply(audio):
        calls.append(audio)
        return iter([])  # a generator function would do too; this one answers with nothing

    rig = Rig(InterruptSafeReplyOnPause, empty_reply)
    rig.user_speaks_then_pauses()

    assert rig.start_emit().join() is None  # super() creates the generator, it ends at once, reset()
    assert not rig.handler.event.is_set()  # nothing was restored
    assert rig.start_emit().join() is None
    assert len(calls) == 1  # fn ran once, not once per emit()


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


def test_a_reply_error_that_escapes_reaches_fastrtc_as_a_webrtc_error():
    """What makes app.py's catch necessary: fastrtc wraps the per-connection emit() in
    webrtc_error_handler (webrtc_connection_mixin.py), so an error that escapes the reply
    generator comes back as a WebRTCError, which the player loop re-raises (utils.py:
    player_worker_decode) — the connection's audio task dies and the page stays mute until a
    reload."""
    def reply(_audio):
        yield chunk(1.0)
        raise RuntimeError("llama-server fell over")

    rig = Rig(InterruptSafeReplyOnPause, reply)
    rig.user_speaks_then_pauses()
    rig.handler.emit = webrtc_error_handler(rig.handler.emit)  # as fastrtc does per connection

    assert rig.start_emit().join()[1][0] == 1.0  # the first chunk
    error = rig.start_emit().join()  # the generator raises on the next chunk

    assert isinstance(error, WebRTCError)
    assert isinstance(error.__cause__, RuntimeError)


def test_the_app_keeps_a_reply_error_from_reaching_fastrtc():
    """The other half: the app catches the error inside the generator, so fastrtc gets a
    transcript update instead of a WebRTCError and the connection keeps working."""
    from vaice.app import make_handler
    from vaice.conversation import Conversation

    class Failing:
        name = "fake"

        def respond(self, audio, history, turn):
            yield chunk(1.0)
            raise RuntimeError("llama-server fell over")

    rig = Rig(InterruptSafeReplyOnPause, make_handler(Failing(), Conversation()))
    rig.user_speaks_then_pauses()
    rig.handler.emit = webrtc_error_handler(rig.handler.emit)  # as fastrtc does per connection

    assert rig.start_emit().join()[1][0] == 1.0
    second = rig.start_emit().join()

    assert isinstance(second, AdditionalOutputs)  # the error, shown in the chatbot

def test_an_idle_emit_waits_for_the_pause_event_instead_of_returning_at_once(monkeypatch):
    """fastrtc polls emit() in a tight loop; returning at once makes it spin."""
    monkeypatch.setattr(interruptible, "IDLE_WAIT_SECONDS", 0.05, raising=False)
    rig = Rig(InterruptSafeReplyOnPause, lambda _audio: iter([]))

    started = time.monotonic()
    assert rig.handler.emit() is None
    elapsed = time.monotonic() - started

    assert elapsed >= 0.04  # it waited for the pause event


def test_the_reply_starts_as_soon_as_the_pause_is_detected(monkeypatch):
    """The idle wait must not delay the reply: it ends when the pause event is set."""
    monkeypatch.setattr(interruptible, "IDLE_WAIT_SECONDS", 5.0, raising=False)  # a long wait: it has to end early
    rig = Rig(InterruptSafeReplyOnPause, lambda _audio: iter([chunk(1.0)]))

    emit = rig.start_emit()  # waits for the pause event
    time.sleep(0.05)
    assert emit.thread.is_alive()  # it is waiting, not spinning
    rig.handler.event.set()  # the VAD detected a pause

    started = time.monotonic()
    result = emit.join()

    assert result[1][0] == 1.0  # the reply's first chunk
    assert time.monotonic() - started < 1.0  # it did not wait for the timeout


def test_the_idle_emit_rate_is_bounded():
    """Without the wait, fastrtc's loop calls emit() ~32k times a second (measured)."""
    rig = Rig(InterruptSafeReplyOnPause, lambda _audio: iter([]))

    started = time.monotonic()
    calls = 0
    while time.monotonic() - started < 0.2:
        rig.handler.emit()
        calls += 1

    assert calls <= 30  # 0.2 s at 20 ms per call is 10 calls; without the wait it is thousands
