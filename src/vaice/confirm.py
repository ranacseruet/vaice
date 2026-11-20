"""Confirmation of a skill that cannot be undone, held by vaice and never decided by the model.

When the model calls a `confirm` skill, the call does not run: it waits here as one request, shown in the UI at once.
Only the *Run* button approves it (it then runs exactly the stored name and arguments, never what the model says
afterwards) and *Cancel* drops it; nothing is ever approved by voice, so the model never sees or interprets an answer.
A request that nobody settles expires `CONFIRM_SECONDS` after it was asked.

A request with a `waiter` is a different kind: a question of a task that is already running (a step of the browser
agent), not a skill call to run. Its answer goes to the waiter, which the task blocks on. Whoever settles the request
settles the waiter in the same step, under the lock: `answer` is yes or no, and every other way out (`take`, `clear`,
a newer request, expiry) is a no, so the task can never see a request that is gone but not answered.
"""
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field

from vaice.conversation import Turn

CONFIRM_SECONDS = 60  # how long an asked question stays answerable, counted from when it was asked


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
    question: str  # what the panel asks
    turn: Turn  # the turn that asked
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    asked_at: float | None = None  # set by `Confirmations.ask`; the expiry counts from it
    waiter: Waiter | None = None  # a question of a running task: the answer goes here, nothing is run (see above)


class Confirmations:
    """At most one `Pending`, shared by the reply thread and the UI's click handlers.

    A request is answerable from the moment it is `ask`ed. `take` is the one way out of a plain request, and it is
    atomic: of two clicks arriving together, exactly one gets the request.
    """

    def __init__(self, clock: Callable[[], float] = time.monotonic):
        self._clock = clock
        self._lock = threading.Lock()
        self._pending: Pending | None = None
        self._epoch = 0  # how many times `clear` ran ("New conversation")

    @property
    def epoch(self) -> int:
        """Changes with every `clear`: an asker takes it when it begins and passes it to `ask`."""
        with self._lock:
            return self._epoch

    def ask(self, pending: Pending, epoch: int | None = None) -> bool:
        """Hold `pending`, replacing any earlier request (a replaced waiter is answered no). With `epoch` (the
        `epoch` the asker saw when it began), refused when a `clear` came since: a reply or a task that "New
        conversation" abandoned must not leave a request that Run would still run. False when refused."""
        with self._lock:
            if epoch is not None and epoch != self._epoch:
                _refuse(pending)
                return False
            replaced, self._pending = self._pending, pending
            pending.asked_at = self._clock()
            _refuse(replaced)
            return True

    def current(self) -> Pending | None:
        """The request, unless it expired `CONFIRM_SECONDS` after it was asked (then it is dropped)."""
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
            self._epoch += 1
            replaced, self._pending = self._pending, None
            _refuse(replaced)

    def _live(self) -> Pending | None:
        """`_pending`, or None after dropping it when expired. Call with the lock held."""
        pending = self._pending
        if pending is not None and pending.asked_at is not None \
                and self._clock() - pending.asked_at >= CONFIRM_SECONDS:
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
