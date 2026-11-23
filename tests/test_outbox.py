"""The proactive-speech outbox: audio and markers queued by a worker thread, drained by the handler's idle emit()."""
import logging
import threading
import time
from functools import partial

import numpy as np

from vaice import outbox as outbox_module
from vaice.outbox import STALE_SECONDS, Marker, Outbox


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def frame(value=0.0, n=10):
    return 24000, np.full(n, value, dtype=np.float32)


def drain(box, allowed=True):
    items = []
    while (item := box.poll(allowed)) is not None:
        items.append(item)
    return items


def test_items_come_out_in_the_order_they_went_in():
    clock = Clock()
    box = Outbox(clock)
    box.poll(True)  # a connection is draining
    first, second = frame(1.0), frame(2.0)
    box.put(first)
    marker = box.put_marker(lambda: None)
    box.put(second)

    items = drain(box)

    assert items[0] is first and items[2] is second
    assert isinstance(items[1], Marker) and items[1] is marker
    assert len(box) == 0


def test_poll_returns_nothing_while_it_is_not_allowed_and_keeps_the_items():
    box = Outbox(Clock())
    box.poll(True)
    box.put(frame(1.0))

    assert box.poll(False) is None
    assert len(box) == 1
    assert box.poll(True) is not None


def test_clear_runs_the_on_drop_of_the_markers_it_discards_and_only_those():
    box = Outbox(Clock())
    box.poll(True)
    dropped = []
    box.put(frame())
    box.put_marker(lambda: dropped.append("ran"), on_drop=lambda: dropped.append("a"))
    box.put_marker(lambda: None)  # no hook
    box.put_marker(lambda: None, on_drop=lambda: dropped.append("b"))

    box.clear()

    assert dropped == ["a", "b"]  # in order; the frame and the hook-less marker have nothing to run
    assert len(box) == 0
    box.clear()
    assert dropped == ["a", "b"]  # nothing is left to drop twice


def test_a_marker_that_was_served_is_not_dropped_by_a_later_clear():
    box = Outbox(Clock())
    box.poll(True)
    dropped = []
    box.put_marker(lambda: None, on_drop=lambda: dropped.append(1))

    assert isinstance(box.poll(True), Marker)
    box.clear()

    assert dropped == []


def test_a_failing_drop_hook_does_not_stop_the_others_and_is_logged(caplog):
    box = Outbox(Clock())
    box.poll(True)
    dropped = []

    def broken():
        raise RuntimeError("hook blew up")

    box.put_marker(lambda: None, on_drop=broken)
    box.put_marker(lambda: None, on_drop=lambda: dropped.append("second"))

    with caplog.at_level(logging.WARNING, logger="vaice.outbox"):
        box.clear()  # must not raise

    assert dropped == ["second"]
    assert "hook blew up" in caplog.text


def test_hooks_run_outside_the_lock_so_a_hook_may_use_the_outbox():
    box = Outbox(Clock())
    box.poll(True)
    box.put_marker(lambda: None, on_drop=lambda: box.put(frame()))  # would deadlock if run under the lock

    done = threading.Thread(target=box.clear, daemon=True)
    done.start()
    done.join(5)

    assert not done.is_alive()
    assert len(box) == 1


def test_the_first_poll_clears_what_was_queued_before_any_connection_drained_it():
    box = Outbox(Clock())
    dropped = []
    box.put(frame())
    box.put_marker(lambda: None, on_drop=lambda: dropped.append(1))

    assert box.poll(True) is None  # nobody was draining: that speech is stale

    assert dropped == [1] and len(box) == 0


def test_a_gap_of_more_than_the_stale_limit_between_polls_clears_the_outbox_first():
    clock = Clock()
    box = Outbox(clock)
    box.poll(True)
    dropped = []
    box.put(frame())
    box.put_marker(lambda: None, on_drop=lambda: dropped.append(1))
    clock.now += STALE_SECONDS + 1  # the page was closed; nobody polled

    assert box.poll(True) is None

    assert dropped == [1] and len(box) == 0


def test_a_poll_resets_the_staleness_even_when_pacing_returned_nothing():
    clock = Clock()
    box = Outbox(clock)
    box.poll(True)
    box.put(frame())
    for _ in range(3):  # a long outcome waits behind its own playback: polled, nothing allowed
        clock.now += STALE_SECONDS * 0.9
        assert box.poll(False) is None

    assert len(box) == 1  # never a gap of more than the limit: not stale
    assert box.poll(True) is not None


def test_an_item_does_not_go_stale_by_age_alone():
    clock = Clock()
    box = Outbox(clock)
    box.poll(True)
    box.put(frame())
    clock.now += STALE_SECONDS * 0.9
    box.poll(False)
    clock.now += STALE_SECONDS * 0.9

    assert box.poll(True) is not None  # queued 54 s ago, but the connection kept polling


def test_a_heartbeat_counts_as_a_poll():
    clock = Clock()
    box = Outbox(clock)
    box.poll(True)
    box.put(frame())
    clock.now += STALE_SECONDS * 0.9
    box.touch()  # a reply is running: nothing is served, the connection is alive
    clock.now += STALE_SECONDS * 0.9

    assert box.poll(True) is not None


def test_constants():
    assert STALE_SECONDS == 30
    assert outbox_module.STALE_SECONDS == STALE_SECONDS


def test_concurrent_puts_clears_and_polls_lose_nothing_and_raise_nothing():
    """Three producers each put 200 numbered frames, each followed by its marker, while a consumer polls and clears
    until they are done and the box is empty: no frame was served twice and every marker ended exactly one way, it ran
    or its `on_drop` did. The threads start together and the producers yield after each put, so they interleave: before
    that, each producer ran to its end in one time slice and the consumer's first clear dropped nearly everything. The
    bound this replaced added `len(box)`, which counts markers too, to the frames served, and failed whenever the
    producers got ahead of the consumer (#119)."""
    box = Outbox()
    box.poll(True)
    errors, seen, ran, dropped = [], [], [], []
    start_together = threading.Barrier(4)

    def producer(start):
        try:
            start_together.wait(5)
            for i in range(start, start + 200):
                box.put((24000, np.full(1, i, dtype=np.float32)))
                box.put_marker(partial(ran.append, i), on_drop=partial(dropped.append, i))
                time.sleep(0)  # let the other threads in
        except BaseException as exc:
            errors.append(exc)

    def serve(item):
        if isinstance(item, Marker):
            item.callback()
        elif item is not None:
            seen.append(int(item[1][0]))

    producers = [threading.Thread(target=producer, args=(n * 1000,)) for n in range(3)]

    def consumer():
        try:
            start_together.wait(5)
            polls = 0
            while any(p.is_alive() for p in producers) or len(box):
                serve(box.poll(True))
                polls += 1
                if polls % 97 == 0:
                    box.clear()
                time.sleep(0)
        except BaseException as exc:
            errors.append(exc)

    threads = [*producers, threading.Thread(target=consumer)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)
    assert not any(t.is_alive() for t in threads) and errors == []

    numbers = [n * 1000 + i for n in range(3) for i in range(200)]
    assert len(seen) == len(set(seen)) and set(seen) <= set(numbers)  # no frame twice, none made up
    assert sorted(ran + dropped) == numbers  # every marker ran or was dropped, once


def test_a_producer_that_captured_the_generation_is_refused_after_a_clear():
    box = Outbox(Clock())
    box.poll(True)
    generation = box.generation
    assert box.put(frame(1.0), generation) is True

    box.clear()  # a barge-in

    assert box.generation == generation + 1
    assert box.put(frame(2.0), generation) is False and len(box) == 0
    assert box.put(frame(3.0), box.generation) is True and len(box) == 1
    assert box.put(frame(4.0)) is True  # no generation given: always queued


def test_a_marker_refused_because_of_a_clear_runs_its_drop_hook_at_once_and_never_its_callback():
    box = Outbox(Clock())
    box.poll(True)
    generation = box.generation
    box.clear()
    events = []

    marker = box.put_marker(lambda: events.append("ran"), on_drop=lambda: events.append("dropped"),
                            generation=generation)

    assert marker is None and events == ["dropped"] and len(box) == 0


def test_a_stale_clear_starts_a_new_generation_too():
    clock = Clock()
    box = Outbox(clock)
    box.poll(True)
    generation = box.generation
    clock.now += STALE_SECONDS + 1
    box.poll(True)

    assert box.put(frame(), generation) is False
