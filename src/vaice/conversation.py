"""Conversation history: what was said and what was spoken, persisted across restarts."""
import json
import logging
import os
import tempfile
import threading
from collections.abc import Callable
from contextlib import AbstractContextManager
from pathlib import Path

log = logging.getLogger(__name__)

# What responders receive as history: (role, text) with role "user" or "assistant"
Message = tuple[str, str]

TRANSCRIPT_WAIT = 5.0  # seconds history() waits for the previous turn's transcript
INAUDIBLE = "(inaudible)"
INTERRUPTED = " (interrupted)"
_VERSION = 1


class Turn:
    """One exchange: what the user said, and the part of the reply that was handed out.

    Thread-safe: responders, fastrtc's workers and the UI all touch it. Once closed (the next
    turn started, or the reply finished) the assistant text is frozen, so a generator that
    fastrtc finalizes late can't change history.
    """

    def __init__(self, lock: AbstractContextManager | None = None, on_change: Callable[[], None] | None = None):
        # A Conversation shares its lock with its turns so saves are serialized and can't deadlock
        self._lock = lock or threading.RLock()
        self._on_change = on_change
        self.user_text: str | None = None  # None until transcribed
        self.assistant_text = ""  # cumulative "spoken so far"
        self.interrupted = False
        self.complete = False
        self.closed = threading.Event()  # responders poll this to stop generating
        self.transcribed = threading.Event()

    def set_user_text(self, text: str) -> None:
        """Always accepted, even after the turn closed: the transcript is not part of the reply."""
        with self._lock:
            self.user_text = text
            self.transcribed.set()
            self._changed()

    def set_assistant_text(self, text: str) -> None:
        with self._lock:
            if self.closed.is_set():
                return
            self.assistant_text = text
            self._changed()

    def finish(self) -> None:
        """The reply ran to the end. A turn already closed (interrupted) stays as it was."""
        with self._lock:
            if self.closed.is_set():
                return
            self.complete = True
            self.closed.set()
            self._changed()

    def close(self) -> None:
        """Freeze the turn; one that never finished was interrupted."""
        with self._lock:
            if not self.complete:
                self.interrupted = True
            self.closed.set()

    def _changed(self) -> None:
        if self._on_change:
            self._on_change()

    def to_dict(self) -> dict:
        with self._lock:
            return {
                "user": self.user_text,
                "assistant": self.assistant_text,
                "interrupted": self.interrupted,
                "complete": self.complete,
            }

    @classmethod
    def from_dict(cls, data: dict, lock: AbstractContextManager, on_change: Callable[[], None]) -> "Turn":
        """A turn loaded from disk is finished business: nothing is still producing it."""
        turn = cls(lock, on_change)
        turn.user_text = data["user"] or ""  # never transcribed (e.g. the app stopped first)
        turn.assistant_text = data["assistant"]
        turn.interrupted = bool(data["interrupted"]) or not data["complete"]
        turn.complete = True
        turn.closed.set()
        turn.transcribed.set()
        return turn


def window_start(count: int, window: int) -> int:
    """Index of the first of `count` turns to keep when at most `window` fit.

    Drops whole blocks of the oldest turns, so the start only moves every `window // 2` turns
    and the prompt prefix stays identical in between (llama-server's cache keeps hitting).
    """
    if window <= 0:
        return count
    excess = count - window
    if excess <= 0:
        return 0
    block = max(1, window // 2)
    return -(-excess // block) * block  # excess rounded up to a whole block


def _spoken(turn: Turn) -> str:
    return turn.assistant_text + (INTERRUPTED if turn.interrupted else "")


def to_messages(turns: list[Turn]) -> list[Message]:
    messages: list[Message] = []
    for turn in turns:
        messages.append(("user", turn.user_text or INAUDIBLE))
        if turn.assistant_text:
            messages.append(("assistant", _spoken(turn)))
    return messages


class Conversation:
    """Ordered turns, optionally persisted as JSON at `path`. One per conversation, shared by all replies."""

    def __init__(self, path: Path | None = None, window: int = 10):
        self._path = path
        self._window = window
        self._lock = threading.RLock()
        self._turns: list[Turn] = self._load() if path else []

    def __len__(self) -> int:
        with self._lock:
            return len(self._turns)

    def start_turn(self) -> Turn:
        """Begin a new exchange; the previous one, if still unfinished, was interrupted."""
        with self._lock:
            if self._turns:
                self._turns[-1].close()
            turn = Turn(self._lock, self._save)
            self._turns.append(turn)
            self._save()
            return turn

    def history(self, before: Turn) -> list[Message]:
        """The (windowed) messages from the turns before `before`, ready for a model prompt."""
        with self._lock:
            turns = self._turns[: next((i for i, t in enumerate(self._turns) if t is before), 0)]
        if turns:  # its transcript is produced off the critical path; don't leave a hole if it's close
            turns[-1].transcribed.wait(TRANSCRIPT_WAIT)
        with self._lock:
            return to_messages(turns[window_start(len(turns), self._window) :])

    def messages_for_ui(self) -> list[dict]:
        """Every turn in Gradio's chatbot "messages" format, including ones still in progress."""
        with self._lock:
            messages = []
            for turn in self._turns:
                user = "…" if turn.user_text is None else turn.user_text or INAUDIBLE  # … = still transcribing
                messages.append({"role": "user", "content": user})
                if turn.assistant_text:
                    messages.append({"role": "assistant", "content": _spoken(turn)})
            return messages

    def reset(self) -> None:
        """Forget everything. Turns started before this are detached: their later updates go nowhere."""
        with self._lock:
            if self._turns:
                self._turns[-1].close()
            self._turns = []
            self._save()

    def _save(self) -> None:
        if not self._path:
            return
        with self._lock:  # re-entrant: also serializes writers so they can't clobber each other's temp file
            data = {"version": _VERSION, "turns": [t.to_dict() for t in self._turns]}
            self._path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=self._path.parent, prefix=self._path.name, suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(data, f)
                os.replace(tmp, self._path)  # readers see the old file or the new one, never half of one
            except BaseException:
                os.unlink(tmp)
                raise

    def _load(self) -> list[Turn]:
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            if data["version"] != _VERSION:
                raise ValueError(f"unsupported version {data['version']!r}")
            return [Turn.from_dict(t, self._lock, self._save) for t in data["turns"]]
        except FileNotFoundError:
            return []
        except (OSError, ValueError, KeyError, TypeError) as exc:  # unreadable, not JSON, or not our shape
            aside = self._path.with_name(self._path.name + ".corrupt")
            os.replace(self._path, aside)
            log.warning("could not read conversation %s (%s: %s); moved it to %s and starting empty",
                        self._path, type(exc).__name__, exc, aside)
            return []
