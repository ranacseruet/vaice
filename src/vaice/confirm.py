"""Confirmation of a skill that cannot be undone, held by vaice and never decided by the model.

When the brain asks for a `confirm` skill, the reply speaks a question and the request waits here. The
user approves by saying one of `YES` as the whole utterance, or by clicking *Run* in the UI (a request that is
not `voice`, such as a shell command, only by the click); anything else drops it. The model never sees or
interprets the answer. A request with `auto_seconds` also runs by itself that long after its question was fully
spoken, unless one of those settles it first (the timer lives in `responders/assistant.py`).

A request with a `waiter` is a different kind: a question of a task that is already running (a step of the browser
agent), not a skill call to run. Its answer goes to the waiter, which the task blocks on. Whoever settles the request
settles the waiter in the same step, under the lock: `answer` is yes or no, and every other way out (`take`, `clear`,
a newer request, expiry) is a no, so the task can never see a request that is gone but not answered.
"""
import re
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field

from vaice.conversation import Turn

CONFIRM_SECONDS = 60  # how long an asked question stays answerable, counted from when it was fully spoken

YES = frozenset({"yes", "yes please", "yeah", "yep", "go ahead", "yes go ahead", "do it", "yes do it",
                 "please do", "confirm", "confirmed"})
NO = frozenset({"no", "nope", "no thanks", "no thank you", "cancel", "cancel that", "stop", "never mind",
                "don't", "do not"})

def _words(text: str) -> str:
    """Lower case, no punctuation (so "don't" is "dont", with either apostrophe), single spaces."""
    return " ".join(re.sub(r"[^\w\s]", "", text.casefold()).split())


_YES_WORDS = frozenset(_words(phrase) for phrase in YES)
_NO_WORDS = frozenset(_words(phrase) for phrase in NO)


def spoken_answer(transcript: str) -> str | None:
    """"yes" or "no" when the WHOLE utterance is one of the phrases (case, punctuation and spacing ignored),
    else None. Never a substring: "yes" inside a longer sentence is not approval."""
    words = _words(transcript)
    if words in _YES_WORDS:
        return "yes"
    if words in _NO_WORDS:
        return "no"
    return None


class Waiter:
    """Where the answer to a question of a running task goes. Set once; the first answer wins."""

    def __init__(self) -> None:
        self._set = threading.Event()
        self._lock = threading.Lock()
        self._approved = False

    def resolve(self, approved: bool) -> None:
        with self._lock:
            if not self._set.is_set():
                self._approved = approved
                self._set.set()

    def wait(self, timeout: float) -> bool | None:
        """The answer once there is one (within `timeout` seconds), else None."""
        return self._approved if self._set.wait(timeout) else None


@dataclass
class Pending:
    """A skill call waiting for the user: `arguments` is the model's raw JSON, exactly what will run."""

    name: str
    arguments: str
    question: str  # what was spoken
    call_id: str  # the tool call's id in `messages`
    messages: list[dict]  # the brain's chat so far, ready for the tool result to be appended
    turn: Turn  # the turn that asked
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    armed_at: float | None = None  # when the question was fully handed out; None until then
    voice: bool = True  # whether a spoken yes may approve it; False: only the UI's Run button
    auto_seconds: float = 0  # > 0: runs by itself this long after the question's audio has played, unless settled first
    auto_delay: float = 0  # set by `arm`: the estimated seconds of the question still to play when it was armed
    task: object | None = None  # the asking reply's `assistant.Task` (step budget, taint), carried over a spoken yes
    waiter: Waiter | None = None  # a question of a running task: the answer goes here, nothing is run (see above)

    def remaining(self, now: float) -> float | None:
        """Seconds left of an armed auto request at clock time `now` (never negative), else None: the time the
        question still had to play when it was armed, then `auto_seconds`."""
        if self.auto_seconds <= 0 or self.armed_at is None:
            return None
        return max(0.0, self.auto_seconds + self.auto_delay - (now - self.armed_at))


class Confirmations:
    """At most one `Pending`, shared by the reply thread, the next reply and the UI's click handlers.

    A request is `ask`ed when its question starts to be spoken (so the screen shows it at once) and `arm`ed
    when all of the question's audio was handed out: only then can a spoken answer approve it, so audio
    that is still playing cannot answer itself. `take` is the one way out, and it is atomic: of a voice
    answer and a click arriving together, exactly one gets the request.
    """

    def __init__(self, clock: Callable[[], float] = time.monotonic):
        self._clock = clock
        self._lock = threading.Lock()
        self._pending: Pending | None = None

    def ask(self, pending: Pending) -> None:
        """Hold `pending`, replacing any earlier request (a replaced waiter is answered no)."""
        with self._lock:
            replaced, self._pending = self._pending, pending
            _refuse(replaced)

    def arm(self, pending_id: str, playing: float = 0.0) -> None:
        """Start the clock, if `pending_id` is still the current request. `playing`: the seconds of the question's
        audio estimated still to play (handed out is not played: the playback queue is unbounded); only an auto
        request's countdown waits for it, answerability and expiry run from now."""
        with self._lock:
            if self._pending is not None and self._pending.id == pending_id:
                self._pending.armed_at = self._clock()
                self._pending.auto_delay = max(0.0, playing)

    def current(self) -> Pending | None:
        """The request, unless it expired `CONFIRM_SECONDS` after it was armed (then it is dropped)."""
        with self._lock:
            return self._live()

    def remaining(self, pending: Pending) -> float | None:
        """Seconds until `pending`, an armed auto request, runs by itself; None when it is not that, or no longer
        the current request."""
        with self._lock:
            live = self._live()
            if live is None or live.id != pending.id:
                return None
            return live.remaining(self._clock())

    def take(self, pending_id: str | None = None) -> Pending | None:
        """Remove and return the request; with `pending_id`, only if that is the current one, so a click on
        an old panel cannot approve a newer request."""
        with self._lock:
            pending = self._live()
            if pending is None or (pending_id is not None and pending.id != pending_id):
                return None
            self._pending = None
            _refuse(pending)  # whoever takes a waiter's request and does not `answer` it, said no
            return pending

    def answer(self, pending_id: str, approved: bool) -> Pending | None:
        """`take` for a request with a waiter, answering it in the same step: the request, once it was the current one
        (None: it was not, or somebody else settled it first)."""
        with self._lock:
            pending = self._live()
            if pending is None or pending.id != pending_id:
                return None
            self._pending = None
            if pending.waiter is not None:
                pending.waiter.resolve(approved)
            return pending

    def clear(self) -> None:
        with self._lock:
            replaced, self._pending = self._pending, None
            _refuse(replaced)

    def _live(self) -> Pending | None:
        """`_pending`, or None after dropping it when expired. Call with the lock held."""
        pending = self._pending
        if pending is not None and pending.armed_at is not None \
                and self._clock() - pending.armed_at >= CONFIRM_SECONDS:
            # said in the Tool activity panel too, where the request would otherwise wait for ever
            pending.turn.add_tool_event(pending.name, pending.arguments, "expired")
            self._pending = None
            _refuse(pending)
            pending = None
        return pending


def _refuse(pending: Pending | None) -> None:
    """A request that leaves without an answer is a no to its waiter, if it has one."""
    if pending is not None and pending.waiter is not None:
        pending.waiter.resolve(False)
