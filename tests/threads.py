"""The shared thread harness for tests that drive fastrtc-facing code.

fastrtc resumes a reply generator on whichever worker thread is free for each chunk, so a harness
that calls the target from one thread lets thread-local bugs slip through (a reply generator
built on the creating thread passes while the live app fails). Every fastrtc-facing test runs its
target on a fresh thread per call — see the note in `InterruptSafeReplyOnPause`.
"""
import threading


class Running:
    """A target started on its own thread. `join()` returns the value or the exception it raised."""

    def __init__(self, target, name):
        self.result = None
        self.thread = threading.Thread(target=self._run, args=(target,), name=name)
        self.thread.start()

    def _run(self, target):
        try:
            self.result = target()
        except BaseException as exc:
            self.result = exc

    def join(self, timeout=10):
        self.thread.join(timeout)
        assert not self.thread.is_alive(), "thread is stuck"
        return self.result


def start_thread(target, name):
    """Start `target` on a fresh thread and return the Running (for when the test must keep working
    while it runs, e.g. a reply that blocks mid-way)."""
    return Running(target, name)


def run_on_thread(target, name, timeout=10):
    """Run `target()` on a fresh thread, wait for it, return the value or the exception."""
    return Running(target, name).join(timeout)


def next_on_fresh_thread(gen, name="fastrtc-worker"):
    """next(gen) on a thread of its own, like fastrtc resumes a reply generator for every chunk.

    Returns the value, or the exception it raised (StopIteration at the end)."""
    return run_on_thread(lambda: next(gen), name)
