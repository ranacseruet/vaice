"""Speech that no reply is carrying: audio and markers queued by a worker thread, served by the handler's idle emit().

A click on *Run* runs a call outside any reply (and a browser task ends long after its reply), so there is no reply
generator to hand the outcome's audio to fastrtc. The worker puts it here instead and
`InterruptSafeReplyOnPause.emit()` drains it while nobody is talking (fastrtc plays whatever an idle
`emit()` returns). Two kinds of item, in one FIFO:

- an audio frame, `(sample_rate, float32 samples)`, exactly what a reply yields;
- a `Marker`: a callback the handler runs, on its own thread, once everything queued before it was served (and so
  nearly played, the handler paces what it serves), with an optional `on_drop` that `clear()` runs instead when the
  marker is discarded without having run. "The text was spoken" is a marker: if the speech never
  played, nothing is recorded.

Staleness is not the age of an item (a long outcome legitimately waits behind its own playback) but whether a
connection is draining at all: the handler calls `poll()` / `touch()` on every `emit()`, and a gap of more than
`STALE_SECONDS` since the last one (or none yet) means speech queued meanwhile belongs to nobody's connection, so it
is cleared before the next poll serves anything.

Each `clear()` (a barge-in, a new reply, "New conversation", staleness) starts a new `generation`. A producer that
captured the generation when it began and passes it to `put` / `put_marker` has what it queues after a clear refused
(a marker's `on_drop` runs at once), so speech that was cancelled and cleared cannot slip one more frame, or its
record marker, in behind the clear.

Callbacks and hooks run outside the lock, and a failing one is logged, never raised.
"""
import logging
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

from vaice.audio import AudioFrame

log = logging.getLogger(__name__)

# No emit() for this long means no connection was draining the outbox: what was queued meanwhile is dropped
STALE_SECONDS = 30.0


@dataclass(frozen=True)
class Marker:
    """A callback in the queue: run by the handler when it is reached; `on_drop` runs instead if it never is."""

    callback: Callable[[], None]
    on_drop: Callable[[], None] | None = None


Item = AudioFrame | Marker


def run_hook(hook: Callable[[], None] | None, what: str) -> None:
    """Run a marker's callback or drop hook; a failure is logged and never escapes."""
    if hook is None:
        return
    try:
        hook()
    except Exception as exc:  # noqa: BLE001 - a hook must not stop the others, or the handler's emit()
        log.warning("outbox %s failed: %s: %s", what, type(exc).__name__, exc)


class Outbox:
    """A thread-safe FIFO of audio frames and markers; see the module docstring."""

    def __init__(self, clock: Callable[[], float] = time.monotonic):
        self._clock = clock
        self._lock = threading.Lock()
        self._items: deque[Item] = deque()
        self._last_poll: float | None = None
        self._generation = 0

    @property
    def generation(self) -> int:
        """Counts the clears so far; see the module docstring."""
        with self._lock:
            return self._generation

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)

    def put(self, frame: AudioFrame, generation: int | None = None) -> bool:
        """Queue `frame`; False (nothing queued) when `generation` is given and a clear happened since."""
        with self._lock:
            if generation is not None and generation != self._generation:
                return False
            self._items.append(frame)
            return True

    def put_marker(self, callback: Callable[[], None], on_drop: Callable[[], None] | None = None,
                   generation: int | None = None) -> Marker | None:
        """Queue a marker. When `generation` is given and a clear happened since, it is not queued: its `on_drop` runs
        at once (it never ran and never will) and None is returned."""
        marker = Marker(callback, on_drop)
        with self._lock:
            if generation is None or generation == self._generation:
                self._items.append(marker)
                return marker
        self._drop([marker])
        return None

    def touch(self) -> None:
        """The handler's `emit()` ran: a connection is draining. Clears first when it was not (see `STALE_SECONDS`)."""
        self.poll(False)

    def poll(self, allowed: bool) -> Item | None:
        """The next item when `allowed` (the handler's pacing says the audio served so far is nearly played), else
        None; None too when empty. Counts as a poll either way, and clears first after a gap of more than
        `STALE_SECONDS` (or when nobody has polled yet), so speech queued with no one listening is never served."""
        now = self._clock()
        with self._lock:
            stale = self._last_poll is None or now - self._last_poll > STALE_SECONDS
            self._last_poll = now
            dropped = self._take_all() if stale else []
            item = self._items.popleft() if allowed and self._items else None
        self._drop(dropped)
        return item

    def clear(self) -> None:
        """Discard everything, running the `on_drop` of every marker that never ran (outside the lock, in order)."""
        with self._lock:
            dropped = self._take_all()
        self._drop(dropped)

    def _take_all(self) -> list[Item]:
        """Empty the queue and return what it held. Call with the lock held."""
        items = list(self._items)
        self._items.clear()
        self._generation += 1
        return items

    @staticmethod
    def _drop(items: list[Item]) -> None:
        for item in items:
            if isinstance(item, Marker):
                run_hook(item.on_drop, "drop hook")
