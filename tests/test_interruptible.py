"""The real fastrtc ReplyOnPause, driven the way fastrtc drives it: emit() on a fresh thread per chunk
(`loop.run_in_executor`), receive() on another thread. Only the VAD model and the playback queue are fakes."""
import asyncio
import inspect
import logging
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
from fastrtc import AdditionalOutputs, ReplyOnPause
from fastrtc.pause_detection import silero
from fastrtc.tracks import AudioCallback
from fastrtc.utils import WebRTCError, webrtc_error_handler
from huggingface_hub.errors import LocalEntryNotFoundError
from threads import run_on_thread, start_thread

from vaice import interruptible
from vaice.interruptible import InterruptSafeReplyOnPause, stop_playing_reply
from vaice.outbox import Outbox

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

    def __init__(self, handler_class, fn, **options):
        self.vad = FakeVad()
        self.handler = handler_class(fn, model=self.vad, **options)
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



class ConnectionLoop:
    """The connection's event loop on a thread of its own, as fastrtc runs it, with fastrtc's real playback queue:
    `AudioCallback.clear_queue` on a stand-in holding an `asyncio.Queue` and the playback clock (`_start`)."""

    def __init__(self):
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, name="connection-loop", daemon=True)
        self.thread.start()
        self.track = SimpleNamespace(queue=asyncio.Queue(), _start=None)
        self.cleared_on: list[str] = []

    def wire(self, handler):
        """What fastrtc does for a connection: the handler's loop, then the track's clear (webrtc_connection_mixin)."""
        handler._loop = self.loop
        handler._clear_queue = self.clear
        return handler

    def clear(self):
        self.cleared_on.append(threading.current_thread().name)
        AudioCallback.clear_queue(self.track)

    def queue_frames(self, count):
        """Frames queued for playback the way player_worker_decode does it: on the loop."""

        async def put():
            for i in range(count):
                await self.track.queue.put(i)
            self.track._start = 1.0  # playback has begun

        asyncio.run_coroutine_threadsafe(put(), self.loop).result(10)

    def settle(self):
        """Wait until the callbacks already scheduled on the loop have run (they run before this coroutine)."""
        asyncio.run_coroutine_threadsafe(asyncio.sleep(0), self.loop).result(10)

    def close(self):
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(10)
        self.loop.close()


@pytest.fixture
def connection():
    loop = ConnectionLoop()
    yield loop
    if not loop.loop.is_closed():
        loop.close()


def test_the_new_conversation_button_clears_the_playback_queue_on_the_connections_loop(connection):
    rig = Rig(InterruptSafeReplyOnPause, lambda _audio: iter([]))
    handler = connection.wire(rig.handler)
    connection.queue_frames(5)

    run_on_thread(lambda: stop_playing_reply(handler), "gradio-click")  # #67: not on the click's thread
    connection.settle()

    assert connection.cleared_on == ["connection-loop"]
    assert connection.track.queue.empty() and connection.track._start is None


def test_fastrtcs_own_barge_in_clears_the_playback_queue_on_the_connections_loop(connection):
    rig = Rig(InterruptSafeReplyOnPause, lambda _audio: iter([chunk(1.0), chunk(2.0)]))
    connection.wire(rig.handler)
    rig.user_speaks_then_pauses()
    assert rig.start_emit().join()[1][0] == 1.0  # the reply is playing
    connection.queue_frames(5)

    rig.user_speaks_then_pauses()  # ReplyOnPause.receive() calls self.clear_queue() on its receive thread
    connection.settle()

    assert connection.cleared_on and set(connection.cleared_on) == {"connection-loop"}
    assert connection.track.queue.empty()


def test_speech_over_playing_proactive_audio_clears_the_playback_queue_on_the_connections_loop(monkeypatch, connection):
    rig = proactive_rig(monkeypatch)
    connection.wire(rig.handler)
    rig.outbox.put(second_of_speech(1.0))
    assert rig.start_emit().join() is not None  # proactive audio is playing
    connection.queue_frames(5)

    speech_only(rig)
    connection.settle()

    assert connection.cleared_on == ["connection-loop"] and connection.track.queue.empty()


def test_a_clear_asked_for_on_the_connections_loop_runs_at_once(connection):
    handler = connection.wire(Rig(InterruptSafeReplyOnPause, lambda _audio: iter([])).handler)

    async def clear_and_look():
        handler.clear_queue()
        return list(connection.cleared_on)  # before this coroutine yields: it ran inline, nothing was scheduled

    assert asyncio.run_coroutine_threadsafe(clear_and_look(), connection.loop).result(10) == ["connection-loop"]


def test_stop_playing_reply_after_the_connections_loop_closed_does_not_raise(connection):
    handler = connection.wire(Rig(InterruptSafeReplyOnPause, lambda _audio: iter([])).handler)
    connection.close()

    stop_playing_reply(handler)  # the connection is gone: nothing is playing, nothing to clear

    assert connection.cleared_on == []


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


def test_the_handler_defaults_to_tuned_vad_options_and_keeps_explicit_ones():
    rig = Rig(InterruptSafeReplyOnPause, lambda _audio: iter([]))
    assert rig.handler.model_options == interruptible.DEFAULT_VAD_OPTIONS
    assert rig.handler.model_options.threshold == 0.6
    assert rig.handler.model_options.min_silence_duration_ms == 750
    assert rig.handler.model_options.speech_pad_ms == 30
    assert rig.handler.copy().model_options == interruptible.DEFAULT_VAD_OPTIONS

    custom = interruptible.SileroVadOptions(threshold=0.8)
    handler = InterruptSafeReplyOnPause(lambda _audio: iter([]), model=FakeVad(), model_options=custom)
    assert handler.model_options is custom
    assert handler.copy().model_options is custom


class _StubSilero(silero.SileroVADModel):
    """In-memory SileroVADModel returning scripted 32 ms window probabilities without loading ONNX."""

    def __init__(self, probs):
        self._probs = list(probs)
        self._idx = 0

    def __call__(self, x, state, sr: int):
        prob = self._probs[self._idx] if self._idx < len(self._probs) else 0.05
        self._idx += 1
        return np.array([[prob]], dtype=np.float32), state


def _make_streaming_handler(fn, probs, *, min_silence_ms=750, max_continuous_speech_s=float("inf"), rate=48000):
    opts = silero.SileroVadOptions(threshold=0.6, min_silence_duration_ms=min_silence_ms, speech_pad_ms=30)
    algo = interruptible.ReplyOnPause(fn).algo_options
    algo.max_continuous_speech_s = max_continuous_speech_s
    handler = InterruptSafeReplyOnPause(
        fn, model=_StubSilero(probs), model_options=opts, algo_options=algo, input_sample_rate=rate,
    )
    handler.send_message_sync = lambda _msg: None
    handler._clear_queue = lambda: None
    return handler


def test_a_brief_noise_burst_inside_a_chunk_is_rejected_by_streaming_vad():
    # 6 x 32 ms windows (192 ms <= 200 ms started_talking_threshold) stays idle; 7 windows (224 ms) starts talking
    frame_32ms = (16000, np.zeros((1, 512), dtype=np.int16))
    idle = _make_streaming_handler(lambda _audio: iter([]), [0.9] * 6 + [0.05] * 10, rate=16000)
    for _ in range(16):
        idle.receive(frame_32ms)
    assert not idle.state.started_talking and not idle.event.is_set()

    speech = _make_streaming_handler(lambda _audio: iter([]), [0.9] * 7, rate=16000)
    for _ in range(7):
        speech.receive(frame_32ms)
    assert speech.state.started_talking and not speech.event.is_set()


def test_streaming_vad_tolerates_mid_sentence_pauses_shorter_than_min_silence():
    # 8 speech windows (256 ms), 19 silent windows (608 ms < 750 ms), 4 speech windows,
    # 1 hysteresis window (0.50 in [0.45, 0.60) resets silence while in_speech), then 25 silent windows -> 1 turn.
    probs = [0.9] * 8 + [0.05] * 19 + [0.9] * 4 + [0.50] + [0.05] * 25
    heard = []
    handler = _make_streaming_handler(lambda audio: (heard.append(audio), iter(()))[1], probs, rate=16000)
    for idx in range(len(probs)):
        handler.receive((16000, np.full((1, 512), idx + 1, dtype=np.int16)))
        if idx < len(probs) - 2:
            assert not handler.event.is_set()
    # Window idx == 56 is the 24th silence after 0.50 (24 * 32 ms = 768 ms >= 750 ms)
    assert handler.event.is_set()
    handler.emit()
    assert len(heard) == 1
    samples = heard[0][1].squeeze()
    # Every sample index 1..57 was captured contiguously without losing the middle segment
    assert samples[0] == 1 and samples[-1] == 57


def test_streaming_vad_hysteresis_and_zero_min_silence():
    # After leaving speech (< 0.45), a mid-band 0.50 window still counts toward silence; min_silence_ms=0 is instant
    frame = (16000, np.ones((1, 512), dtype=np.int16))
    h_hyst = _make_streaming_handler(
        lambda _a: iter([]), [0.9] * 7 + [0.05] + [0.50] * 2, min_silence_ms=90, rate=16000,
    )
    for _ in range(10):
        h_hyst.receive(frame)
    assert h_hyst.event.is_set()

    h_zero = _make_streaming_handler(lambda _a: iter([]), [0.9] * 7 + [0.05], min_silence_ms=0, rate=16000)
    for _ in range(8):
        h_zero.receive(frame)
    assert h_zero.event.is_set()


def test_streaming_vad_preroll_cap_onset_reset_and_new_conversation_reset():
    # 1) Long silence capped by pre-roll, interrupted incomplete onset (3x speech + 3x silence), then New conversation
    heard = []
    probs = [0.05] * 30 + [0.9] * 3 + [0.05] * 4 + [0.05] * 15 + [0.9] * 8 + [0.05] * 25
    handler = _make_streaming_handler(lambda audio: (heard.append(audio), iter(()))[1], probs, rate=16000)
    # 30 windows (~960 ms) old silence marked 100 -> trimmed to cap -> incomplete onset -> stop_playing_reply clears it
    for _ in range(37):
        handler.receive((16000, np.full((1, 512), 100, dtype=np.int16)))
    assert not handler.state.started_talking
    assert handler.state._stream_vad.onset_speech == 0.0  # 3 silent windows reset onset_speech

    stop_playing_reply(handler)
    assert getattr(handler.state, "_stream_vad", None) is None

    # New conversation: 15 pre-roll windows marked 200, then 8 speech + 25 silence marked 300
    for _ in range(15):
        handler.receive((16000, np.full((1, 512), 200, dtype=np.int16)))
    for _ in range(33):
        handler.receive((16000, np.full((1, 512), 300, dtype=np.int16)))
    assert handler.event.is_set()
    handler.emit()
    samples = heard[0][1].squeeze()
    assert not np.any(samples == 100)
    assert np.any(samples == 200) and samples[-1] == 300


@pytest.mark.parametrize("sample_rate", [24000, 44100])
def test_streaming_vad_supports_stereo_empty_max_speech_and_other_rates(sample_rate):
    probs = [0.9] * 15
    handler = _make_streaming_handler(
        lambda _a: iter([]), probs, max_continuous_speech_s=0.35, rate=sample_rate,
    )
    assert not handler.determine_pause(np.array([], dtype=np.int16), sample_rate, handler.state)
    assert not handler.determine_pause(np.zeros((1, 100), dtype=np.int16), 0, handler.state)

    step = sample_rate * 20 // 1000
    for _ in range(25):  # 500 ms stereo input > max_continuous_speech_s=0.35
        handler.receive((sample_rate, np.ones((2, step), dtype=np.int16)))
        if handler.event.is_set():
            break
    assert handler.event.is_set()

    # Sample rate switch mid-stream resets pending resampler buffer cleanly without crashing
    handler.reset()
    assert not handler.determine_pause(np.zeros((1, step), dtype=np.int16), sample_rate, handler.state)
    assert not handler.determine_pause(np.zeros((1, 320), dtype=np.int16), 16000, handler.state)


def test_streaming_vad_issue_13_restore_yields_to_newer_streaming_frame():
    entered, release = threading.Event(), threading.Event()

    def reply(_audio):
        yield chunk(1.0)
        entered.set()
        assert release.wait(10)

    handler = _make_streaming_handler(reply, [0.9] * 8 + [0.05] * 25, rate=16000)

    def feed_utterance():
        for _ in range(33):
            handler.receive((16000, np.zeros((1, 512), dtype=np.int16)))

    run_on_thread(feed_utterance, "fastrtc-receive")
    assert handler.event.is_set()
    assert start_thread(handler.emit, "fastrtc-emit").join()[1][0] == 1.0

    handler.model = _StubSilero([0.9] * 8 + [0.05] * 25 + [0.05] * 5)
    in_flight = start_thread(handler.emit, "fastrtc-emit")
    assert entered.wait(10)
    run_on_thread(feed_utterance, "fastrtc-receive")  # #13 pause lands on old state

    newer = []
    orig_reset = handler.reset

    def reset_then_short_frame():
        orig_reset()
        newer.append(handler.state)
        handler.receive((16000, np.zeros((1, 512), dtype=np.int16)))  # 32 ms frame on fresh state

    handler.reset = reset_then_short_frame
    release.set()

    assert in_flight.join() is None
    assert handler.state is newer[0] and getattr(newer[0], "_stream_vad", None) is not None
    assert not handler.event.is_set()


def test_streaming_vad_issue_13_restores_barge_in_when_fresh_state_is_clean():
    entered, release = threading.Event(), threading.Event()
    heard = []

    def reply(audio):
        heard.append(audio)
        yield chunk(1.0)
        if len(heard) == 1:
            entered.set()
            assert release.wait(10)

    handler = _make_streaming_handler(reply, [0.9] * 8 + [0.05] * 25, rate=16000)

    def feed_utterance(value: int):
        for _ in range(33):
            handler.receive((16000, np.full((1, 512), value, dtype=np.int16)))

    run_on_thread(lambda: feed_utterance(1), "fastrtc-receive")
    assert start_thread(handler.emit, "fastrtc-emit").join()[1][0] == 1.0

    handler.model = _StubSilero([0.9] * 8 + [0.05] * 25)
    in_flight = start_thread(handler.emit, "fastrtc-emit")
    assert entered.wait(10)
    run_on_thread(lambda: feed_utterance(2), "fastrtc-receive")  # #13 pause lands while next() is in flight
    barge_in_state = handler.state
    release.set()

    assert in_flight.join() is None  # chunk 2 skipped and generator ended
    assert handler.state is barge_in_state and handler.event.is_set()
    assert start_thread(handler.emit, "fastrtc-emit").join()[1][0] == 1.0
    assert heard[1][1].squeeze()[0] == 2


def test_the_preroll_from_the_preceding_chunk_is_prepended_when_speech_starts():
    heard = []

    def reply(audio):
        heard.append(audio)
        yield chunk(1.0)

    rig = Rig(InterruptSafeReplyOnPause, reply)

    # Pre-speech chunk (accumulated across two half-second frames): VAD says no speech, chunk is discarded
    half_a = (48000, np.full((1, 24000), 10, dtype=np.int16))
    half_b = (48000, np.full((1, 24000), 100, dtype=np.int16))
    rig.vad.speech = 0.0
    run_on_thread(lambda: rig.handler.receive(half_a), "fastrtc-receive")
    run_on_thread(lambda: rig.handler.receive(half_b), "fastrtc-receive")

    # Speech chunk: started_talking flips from False to True
    speech = (48000, np.full((1, 48000), 200, dtype=np.int16))
    rig.vad.speech = 1.0
    run_on_thread(lambda: rig.handler.receive(speech), "fastrtc-receive")

    # Pause chunk: speech ends
    rig.vad.speech = 0.0
    run_on_thread(lambda: rig.handler.receive(ONE_SECOND), "fastrtc-receive")

    assert rig.start_emit().join()[1][0] == 1.0
    assert len(heard) == 1
    samples = heard[0][1].squeeze()
    preroll_len = int(48000 * interruptible.PREROLL_SECONDS)
    np.testing.assert_array_equal(samples[:preroll_len], 100)
    np.testing.assert_array_equal(samples[preroll_len : preroll_len + 48000], 200)


def test_reset_clears_any_saved_preroll_so_it_never_leaks_into_a_new_conversation():
    heard = []

    def reply(audio):
        heard.append(audio)
        yield chunk(1.0)

    rig = Rig(InterruptSafeReplyOnPause, reply)
    rig.vad.speech = 0.0
    run_on_thread(lambda: rig.handler.receive((48000, np.full((1, 48000), 100, dtype=np.int16))), "fastrtc-receive")
    assert getattr(rig.handler.state, "_preroll", None) is not None

    stop_playing_reply(rig.handler)
    assert getattr(rig.handler.state, "_preroll", None) is None

    rig.user_speaks_then_pauses()
    assert rig.start_emit().join()[1][0] == 1.0
    # 2 x ONE_SECOND (speech + pause), with no 100s prepended from before the reset
    assert heard[0][1].shape[-1] == 2 * 48000
    assert not np.any(heard[0][1] == 100)


def test_a_fastrtc_without_the_generator_attribute_fails_at_construction(monkeypatch):
    """A fastrtc upgrade that drops the attribute the interrupt fix relies on must fail loudly
    at construction with the actionable message, not mysteriously at runtime."""

    def bare_init(self, *args, **kwargs):
        self.model_options = None  # the VAD default branch reads it

    monkeypatch.setattr(ReplyOnPause, "__init__", bare_init)
    with pytest.raises(RuntimeError, match="generator` attribute"):
        InterruptSafeReplyOnPause(lambda _: iter(()))


def test_a_fastrtc_without_the_pause_event_fails_at_construction(monkeypatch):
    def bare_init(self, *args, **kwargs):
        self.model_options = None
        self.generator = None  # the first guard passes; the pause event is gone

    monkeypatch.setattr(ReplyOnPause, "__init__", bare_init)
    with pytest.raises(RuntimeError, match="event` attribute"):
        InterruptSafeReplyOnPause(lambda _: iter(()))


# --- proactive speech: the outbox the idle emit() serves, paced by the estimated end of playback

class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def second_of_speech(value=0.0):
    """One second of audio at 24 kHz: it takes one second to play."""
    return 24000, np.full(24000, value, dtype=np.float32)


def proactive_rig(monkeypatch, **options):
    """A handler with an outbox, both on one fake clock, and an idle wait too short to slow the tests."""
    monkeypatch.setattr(interruptible, "IDLE_WAIT_SECONDS", 0.001)
    clock = FakeClock()
    rig = Rig(InterruptSafeReplyOnPause, lambda _audio: iter([]), outbox=Outbox(clock), **options)
    rig.handler._clock = clock
    rig.clock = clock
    rig.outbox = rig.handler.outbox
    assert rig.start_emit().join() is None  # a first poll: from here on a connection is draining the outbox
    return rig


def speech_only(rig):
    """The user starts talking and has not stopped: `started_talking` flips, no pause is detected."""
    rig.vad.speech = 1.0
    run_on_thread(lambda: rig.handler.receive(ONE_SECOND), "fastrtc-receive")


def test_an_idle_emit_serves_the_frames_in_the_outbox(monkeypatch):
    rig = proactive_rig(monkeypatch)
    frame = second_of_speech(1.0)
    rig.outbox.put(frame)

    assert rig.start_emit().join() is frame
    assert rig.start_emit().join() is None  # nothing else queued


def test_an_emit_with_no_outbox_attached_behaves_as_before():
    rig = Rig(InterruptSafeReplyOnPause, lambda _audio: iter([]))

    assert rig.handler.outbox is None
    assert rig.start_emit().join() is None


def test_frames_are_served_only_while_the_estimated_playback_end_is_less_than_the_lead_ahead(monkeypatch):
    rig = proactive_rig(monkeypatch)
    frames = [second_of_speech(n) for n in (1.0, 2.0, 3.0)]
    for f in frames:
        rig.outbox.put(f)

    assert rig.start_emit().join() is frames[0]  # now plays for 1 s
    assert rig.start_emit().join() is None  # 1 s ahead: not yet
    rig.clock.now += 0.5
    assert rig.start_emit().join() is None  # 0.5 s ahead is still more than the lead
    rig.clock.now += 0.2 + 0.001
    assert rig.start_emit().join() is frames[1]  # 0.3 s ahead: the next one may be queued
    assert rig.start_emit().join() is None
    assert len(rig.outbox) == 1  # the third waits for its turn, it is not dropped


def test_markers_run_in_order_after_the_frames_in_front_of_them_and_within_the_lead_of_their_playback(monkeypatch):
    rig = proactive_rig(monkeypatch)
    events = []
    first, second = second_of_speech(1.0), second_of_speech(2.0)
    rig.outbox.put(first)
    rig.outbox.put_marker(lambda: events.append("after first"))
    rig.outbox.put(second)
    rig.outbox.put_marker(lambda: events.append("after second"))

    assert rig.start_emit().join() is first
    assert events == []  # the first frame has only just been handed out, not played
    assert rig.start_emit().join() is None and events == []

    rig.clock.now += 0.7  # 0.3 s of the first frame left: the marker behind it is due, and the next frame
    assert rig.start_emit().join() is second  # one emit call: the marker ran, then the frame was returned
    assert events == ["after first"]
    assert rig.start_emit().join() is None and events == ["after first"]

    rig.clock.now += 1.0  # the second frame was queued at 1000.7 and runs to 1002: 0.3 s left at 1001.7
    assert rig.start_emit().join() is None  # the last marker needs no frame to follow it
    assert events == ["after first", "after second"]


def test_a_marker_on_its_own_runs_at_once_when_nothing_is_playing(monkeypatch):
    rig = proactive_rig(monkeypatch)
    events = []
    rig.outbox.put_marker(lambda: events.append(1))

    assert rig.start_emit().join() is None
    assert events == [1]


def test_a_failing_marker_callback_is_logged_and_the_next_item_is_still_served(monkeypatch, caplog):
    rig = proactive_rig(monkeypatch)
    frame = second_of_speech(1.0)

    def broken():
        raise RuntimeError("marker blew up")

    rig.outbox.put_marker(broken)
    rig.outbox.put(frame)

    with caplog.at_level(logging.WARNING, logger="vaice.interruptible"):
        assert rig.start_emit().join() is frame

    assert "marker blew up" in caplog.text


def test_nothing_is_served_while_a_reply_is_starting_or_running_or_the_user_is_talking(monkeypatch):
    rig = proactive_rig(monkeypatch)
    rig.outbox.put(second_of_speech(1.0))

    rig.handler.event.set()  # a pause was detected: the reply (empty here) takes over
    assert rig.start_emit().join() is None
    assert len(rig.outbox) == 1
    rig.handler.reset()

    rig.handler.generator = iter([])  # a reply generator exists
    assert rig.start_emit().join() is None
    assert len(rig.outbox) == 1
    rig.handler.generator = None

    rig.handler.state.started_talking = True  # the user is talking
    assert rig.start_emit().join() is None
    assert len(rig.outbox) == 1
    rig.handler.state.started_talking = False

    assert rig.start_emit().join() is not None  # all clear: served


def test_a_connection_busy_with_a_reply_is_still_a_live_connection_for_the_staleness_check(monkeypatch):
    rig = proactive_rig(monkeypatch)
    frame = second_of_speech(1.0)
    rig.outbox.put(frame)
    rig.handler.generator = iter([])  # a long reply: nothing is served meanwhile
    for _ in range(2):
        rig.clock.now += 20
        assert rig.start_emit().join() is None
    rig.handler.generator = None

    assert rig.start_emit().join() is frame  # 40 s since the last serving poll, but emit() was called throughout


def test_speech_starting_clears_the_outbox_runs_the_drop_hooks_and_cuts_the_audio_still_playing(monkeypatch):
    rig = proactive_rig(monkeypatch)
    dropped = []
    rig.outbox.put(second_of_speech(1.0))
    rig.outbox.put_marker(lambda: None, on_drop=lambda: dropped.append("arm"))
    rig.outbox.put(second_of_speech(2.0))
    assert rig.start_emit().join() is not None  # the first second is queued for playback

    speech_only(rig)

    assert len(rig.outbox) == 0 and dropped == ["arm"]
    assert rig.queue_clears == 1  # what fastrtc had queued is dropped too
    assert rig.handler._proactive_until == 0


def test_speech_starting_after_the_proactive_audio_finished_clears_no_queue(monkeypatch):
    rig = proactive_rig(monkeypatch)
    dropped = []
    rig.outbox.put(second_of_speech(1.0))
    assert rig.start_emit().join() is not None
    rig.clock.now += 2.0  # it has played
    rig.outbox.put_marker(lambda: None, on_drop=lambda: dropped.append("late"))

    speech_only(rig)

    assert dropped == ["late"] and rig.queue_clears == 0


def test_speech_starting_with_nothing_queued_and_nothing_playing_changes_nothing(monkeypatch):
    rig = proactive_rig(monkeypatch)

    speech_only(rig)

    assert rig.queue_clears == 0 and len(rig.outbox) == 0


def test_speech_does_not_clear_proactive_audio_when_interrupting_is_off(monkeypatch):
    rig = proactive_rig(monkeypatch, can_interrupt=False)
    dropped = []
    rig.outbox.put(second_of_speech(1.0))
    rig.outbox.put_marker(lambda: None, on_drop=lambda: dropped.append(1))
    assert rig.start_emit().join() is not None

    speech_only(rig)

    assert dropped == [] and len(rig.outbox) == 1 and rig.queue_clears == 0


def test_speech_during_a_reply_is_left_to_the_barge_in_which_does_not_touch_the_outbox(monkeypatch):
    monkeypatch.setattr(interruptible, "IDLE_WAIT_SECONDS", 0.001)
    clock = FakeClock()
    rig = Rig(InterruptSafeReplyOnPause, lambda _audio: iter([chunk(1.0), chunk(2.0)]), outbox=Outbox(clock))
    rig.handler._clock = clock
    dropped = []
    rig.user_speaks_then_pauses()
    assert rig.start_emit().join()[1][0] == 1.0  # the reply is running
    rig.handler.outbox.poll(True)
    rig.handler.outbox.put_marker(lambda: None, on_drop=lambda: dropped.append(1))

    speech_only(rig)

    assert dropped == [] and len(rig.handler.outbox) == 1


def test_the_copy_made_for_a_connection_keeps_the_outbox_and_the_clock_free_constructor_argument():
    outbox = Outbox()
    original = InterruptSafeReplyOnPause(lambda _audio: iter([]), model=FakeVad(), outbox=outbox)

    copy = original.copy()

    assert type(copy) is InterruptSafeReplyOnPause and copy.outbox is outbox
    assert "outbox" not in original._copy_kwargs()  # fastrtc's own signature, pinned by the test above


def test_a_copy_of_a_handler_without_an_outbox_has_none():
    assert InterruptSafeReplyOnPause(lambda _audio: iter([]), model=FakeVad()).copy().outbox is None


def test_stop_playing_reply_also_clears_the_outbox_and_forgets_the_playing_estimate(monkeypatch):
    rig = proactive_rig(monkeypatch)
    dropped = []
    rig.outbox.put(second_of_speech(1.0))
    rig.outbox.put_marker(lambda: None, on_drop=lambda: dropped.append(1))
    rig.outbox.put(second_of_speech(2.0))
    assert rig.start_emit().join() is not None

    stop_playing_reply(rig.handler)  # "New conversation"

    assert len(rig.outbox) == 0 and dropped == [1]
    assert rig.handler._proactive_until == 0
    assert rig.queue_clears == 1


def test_the_idle_emit_rate_with_an_outbox_attached_is_still_bounded():
    rig = Rig(InterruptSafeReplyOnPause, lambda _audio: iter([]), outbox=Outbox())

    started = time.monotonic()
    calls = 0
    while time.monotonic() - started < 0.2:
        rig.handler.emit()
        calls += 1

    assert calls <= 30



def test_reply_audio_still_queued_in_fastrtc_holds_back_the_proactive_frames_behind_it(monkeypatch):
    # a click while the question is still playing: the reply generator is done, but its second of audio is queued
    monkeypatch.setattr(interruptible, "IDLE_WAIT_SECONDS", 0.001)
    clock = FakeClock()
    reply_frame, outcome = second_of_speech(1.0), second_of_speech(2.0)
    rig = Rig(InterruptSafeReplyOnPause, lambda _audio: iter([reply_frame]), outbox=Outbox(clock))
    rig.handler._clock = clock
    rig.handler.outbox.touch()
    rig.user_speaks_then_pauses()
    assert rig.start_emit().join() is reply_frame  # handed out: it plays for the next second
    assert rig.start_emit().join() is None  # the generator ends here
    rig.handler.outbox.put(outcome)
    rig.handler.outbox.put_marker(lambda: None)

    assert rig.start_emit().join() is None  # 1 s of the question still to play: the outcome waits behind it
    assert len(rig.handler.outbox) == 2
    clock.now += 0.7
    assert rig.start_emit().join() is outcome  # 0.3 s left: it may be queued now


def test_interrupting_an_ordinary_reply_forgets_its_audio_in_the_playback_estimate(monkeypatch):
    # fastrtc drops the queued reply audio when it is interrupted: the pacing must not wait for audio that never plays
    monkeypatch.setattr(interruptible, "IDLE_WAIT_SECONDS", 0.001)
    clock = FakeClock()
    rig = Rig(InterruptSafeReplyOnPause, lambda _audio: iter([second_of_speech(1.0), second_of_speech(2.0)]),
              outbox=Outbox(clock))
    rig.handler._clock = clock
    dropped = []
    rig.handler.outbox.touch()  # a connection is draining it, so nothing is stale
    rig.user_speaks_then_pauses()
    assert rig.start_emit().join() is not None  # a second of reply audio queued: the estimate now runs ahead
    rig.handler.outbox.put_marker(lambda: None, on_drop=lambda: dropped.append(1))  # queued while the reply plays
    assert rig.handler._proactive_until > clock.now

    speech_only(rig)  # the user talks over the reply (the reply generator is still attached)

    assert rig.handler._proactive_until == 0.0
    assert dropped == []  # the outbox itself is for the reply path's own barge-in: untouched here


# -- the Silero VAD model from the Hugging Face cache, with no request (#143)


class FakeHub:
    """`hf_hub_download` as `use_cached_vad_model` calls it, and fastrtc's own loader: which of the two ran."""

    def __init__(self, cached):
        self.cached, self.calls = cached, []

    def hf_hub_download(self, *, repo_id, filename, local_files_only):
        self.calls.append(("cache", repo_id, filename, local_files_only))
        if not self.cached:
            raise LocalEntryNotFoundError("not in the cache")
        return "/cache/silero_vad.onnx"

    def fastrtc_download(self):
        self.calls.append(("fastrtc",))
        return "/downloaded/silero_vad.onnx"


@pytest.fixture
def hub(monkeypatch):
    def make(cached):
        fake = FakeHub(cached)
        monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
        monkeypatch.setattr(interruptible, "hf_hub_download", fake.hf_hub_download)
        # fastrtc's loader, restored after the test (use_cached_vad_model replaces it)
        monkeypatch.setattr(silero.SileroVADModel, "download_model", staticmethod(fake.fastrtc_download))
        return fake

    return make


def test_a_cached_vad_model_is_loaded_from_the_cache_without_fastrtcs_request(hub):
    fake = hub(cached=True)

    interruptible.use_cached_vad_model()

    assert silero.SileroVADModel.download_model() == "/cache/silero_vad.onnx"
    assert fake.calls == [("cache", "freddyaboulton/silero-vad", "silero_vad.onnx", True)]


def test_a_vad_model_not_cached_yet_is_downloaded_by_fastrtc(hub):
    fake = hub(cached=False)

    interruptible.use_cached_vad_model()

    assert silero.SileroVADModel.download_model() == "/downloaded/silero_vad.onnx"
    assert [call[0] for call in fake.calls] == ["cache", "fastrtc"]


@pytest.mark.parametrize("value", ["1", "0"])
def test_a_users_hf_hub_offline_leaves_fastrtcs_loader_as_it_is(hub, monkeypatch, value):
    fake = hub(cached=True)
    monkeypatch.setenv("HF_HUB_OFFLINE", value)

    interruptible.use_cached_vad_model()

    assert silero.SileroVADModel.download_model() == "/downloaded/silero_vad.onnx" and fake.calls == [("fastrtc",)]


def test_using_the_cached_vad_model_twice_wraps_the_loader_once(hub):
    fake = hub(cached=False)

    interruptible.use_cached_vad_model()
    interruptible.use_cached_vad_model()

    silero.SileroVADModel.download_model()
    assert [call[0] for call in fake.calls] == ["cache", "fastrtc"]


def test_a_fastrtc_without_the_loader_is_left_alone_with_a_warning(monkeypatch, caplog):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.setattr(silero, "SileroVADModel", type("SileroVADModel", (), {}))

    with caplog.at_level(logging.WARNING, logger="vaice.interruptible"):
        interruptible.use_cached_vad_model()

    assert "no download_model()" in caplog.text
