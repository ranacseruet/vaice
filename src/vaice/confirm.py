"""Confirmation of a skill that cannot be undone, held by vaice and never decided by the model.

When the brain asks for a `confirm` skill, the reply speaks a question and the request waits here. The
user approves by saying one of `YES` as the whole utterance, or by clicking *Run* in the UI (a request that is
not `voice`, such as a shell command, only by the click); anything else drops it. The model never sees or
interprets the answer.
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
        """Hold `pending`, replacing any earlier request."""
        with self._lock:
            self._pending = pending

    def arm(self, pending_id: str) -> None:
        """Start the clock, if `pending_id` is still the current request."""
        with self._lock:
            if self._pending is not None and self._pending.id == pending_id:
                self._pending.armed_at = self._clock()

    def current(self) -> Pending | None:
        """The request, unless it expired `CONFIRM_SECONDS` after it was armed (then it is dropped)."""
        with self._lock:
            return self._live()

    def take(self, pending_id: str | None = None) -> Pending | None:
        """Remove and return the request; with `pending_id`, only if that is the current one, so a click on
        an old panel cannot approve a newer request."""
        with self._lock:
            pending = self._live()
            if pending is None or (pending_id is not None and pending.id != pending_id):
                return None
            self._pending = None
            return pending

    def clear(self) -> None:
        with self._lock:
            self._pending = None

    def _live(self) -> Pending | None:
        """`_pending`, or None after dropping it when expired. Call with the lock held."""
        pending = self._pending
        if pending is not None and pending.armed_at is not None \
                and self._clock() - pending.armed_at >= CONFIRM_SECONDS:
            # said in the Tool activity panel too, where the request would otherwise wait for ever
            pending.turn.add_tool_event(pending.name, pending.arguments, "expired")
            self._pending = pending = None
        return pending
