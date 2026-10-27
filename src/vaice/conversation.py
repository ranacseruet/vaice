"""Conversation history: what was said and what was spoken, persisted across restarts."""
import json
import logging
import os
import tempfile
import threading
import time
from collections.abc import Callable
from contextlib import AbstractContextManager
from pathlib import Path

log = logging.getLogger(__name__)

# What responders receive as history: (role, text) with role "user" or "assistant"
Message = tuple[str, str]

TRANSCRIPT_WAIT = 5.0  # seconds history() waits for the previous turn's transcript
INAUDIBLE = "(inaudible)"
INTERRUPTED = " (interrupted)"
_clock = time.monotonic  # when playback started/stopped; tests substitute a fake
_VERSION = 1


class Turn:
    """One exchange: what the user said, and the part of the reply that was played.

    Thread-safe: responders, fastrtc's workers and the UI all touch it. Once closed (the next
    turn started, or the reply finished) the assistant text is frozen, so a generator that
    fastrtc finalizes late can't change history.

    Responders report the text of the audio they handed out, but handed out is not played: fastrtc
    queues chunks for playback (an unbounded queue, so it can run ahead) and cuts playback only
    when the user's interrupting utterance ends. So when the next turn starts, close() also
    compares the time since the first audio with the audio handed out, and keeps only the share
    of the words that had time to play.
    """

    def __init__(self, lock: AbstractContextManager | None = None, on_change: Callable[[], None] | None = None):
        # A Conversation shares its lock with its turns so saves are serialized and can't deadlock
        self._lock = lock or threading.RLock()
        self._on_change = on_change
        self.user_text: str | None = None  # None until transcribed
        self.assistant_text = ""  # cumulative "spoken so far"
        self.error = ""  # why the reply failed, for the UI only; never sent to the model
        self.interrupted = False
        self.complete = False
        self.closed = threading.Event()  # responders poll this to stop generating
        self.transcribed = threading.Event()
        self._audio_started: float | None = None  # _clock() when the first audio went out
        self._audio_seconds = 0.0  # audio handed out so far

    def set_user_text(self, text: str) -> None:
        """Always accepted, even after the turn closed: the transcript is not part of the reply."""
        with self._lock:
            self.user_text = text
            self.transcribed.set()
            self._changed()

    def set_assistant_text(self, text: str) -> None:
        with self._lock:
            if self.closed.is_set() or text == self.assistant_text:  # unchanged: skip the save
                return
            self.assistant_text = text
            self._changed()

    def audio_handed_out(self, seconds: float, now: float | None = None) -> None:
        """Playback starts with the first chunk and runs in real time from there."""
        with self._lock:
            if self._audio_started is None:
                self._audio_started = _clock() if now is None else now
            self._audio_seconds += seconds

    def fail(self, message: str) -> None:
        """The reply failed. `message` is shown in the UI, never sent to the model; a transcript
        the responder never reported counts as inaudible so the next turn's history doesn't wait."""
        with self._lock:
            try:
                self.error = message
                if self.user_text is None:
                    self.user_text = ""
                    self.transcribed.set()
                self._changed()
            finally:
                self.finish()  # also a no-op if the turn was already closed

    def finish(self) -> None:
        """The reply ran to the end. A turn already closed (interrupted) stays as it was."""
        with self._lock:
            if self.closed.is_set():
                return
            self.complete = True
            self.closed.set()
            self._changed()

    def close(self, now: float | None = None) -> None:
        """Freeze the turn. It was interrupted if it never finished, or if its audio was still playing."""
        with self._lock:
            if self._audio_started is not None:
                played = (_clock() if now is None else now) - self._audio_started
                if played < self._audio_seconds:
                    self.interrupted = True
                    words = self.assistant_text.split()
                    # At least one word: something was played, so the reply was not empty
                    keep = max(1, int(len(words) * played / self._audio_seconds)) if words else 0
                    self.assistant_text = " ".join(words[:keep])
            if not self.complete:
                self.interrupted = True
            self.closed.set()
            self._changed()

    def _changed(self) -> None:
        if self._on_change:
            self._on_change()

    def to_dict(self) -> dict:
        with self._lock:
            return {
                "user": self.user_text,
                "assistant": self.assistant_text,
                "error": self.error,
                "interrupted": self.interrupted,
                "complete": self.complete,
            }

    @classmethod
    def from_dict(cls, data: dict, lock: AbstractContextManager, on_change: Callable[[], None]) -> "Turn":
        """A turn loaded from disk is finished business: nothing is still producing it."""
        turn = cls(lock, on_change)
        turn.user_text = data["user"] or ""  # never transcribed (e.g. the app stopped first)
        turn.assistant_text = data["assistant"]
        turn.error = data.get("error", "")  # written by older versions without it
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
    """The prompt messages for `turns`: what the user said and what was answered.

    A turn whose transcript is still pending is dropped whole: the question has not been
    transcribed yet, so the answer to it would be a dangling reply, and `(inaudible)` would
    wrongly say the user was unintelligible rather than still being transcribed. A transcript
    that arrived but is empty counts as inaudible.
    """
    messages: list[Message] = []
    for turn in turns:
        if turn.user_text is None:
            log.debug("dropping a turn from the prompt: its transcript has not arrived yet")
            continue
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
            turns = turns[window_start(len(turns), self._window) :]
        if turns:  # its transcript is produced off the critical path; don't leave a hole if it's close
            turns[-1].transcribed.wait(TRANSCRIPT_WAIT)
        with self._lock:
            return to_messages(turns)

    def messages_for_ui(self) -> list[dict]:
        """Every turn in Gradio's chatbot "messages" format, including ones still in progress."""
        with self._lock:
            messages = []
            for turn in self._turns:
                user = "…" if turn.user_text is None else turn.user_text or INAUDIBLE  # … = still transcribing
                messages.append({"role": "user", "content": user})
                spoken = _spoken(turn)
                if turn.error:  # a failed reply: what was heard of it, plus why it stopped
                    spoken = f"{spoken} (error: {turn.error})" if spoken else f"(error: {turn.error})"
                if spoken:
                    messages.append({"role": "assistant", "content": spoken})
            return messages

    def reset(self) -> None:
        """Forget everything. Turns started before this are detached: their later updates go nowhere."""
        with self._lock:
            if self._turns:
                self._turns[-1].close()
            self._turns = []
            self._save()

    def _save(self) -> None:
        """Write the conversation to disk.

        A failed write is a warning, not an error: the conversation stays in memory and the reply
        continues (a full disk or a read-only file must not end a conversation). Only the
        persistence is lost.
        """
        if not self._path:
            return
        with self._lock:  # re-entrant: also serializes writers so they can't clobber each other's temp file
            tmp: str | None = None
            try:
                data = {"version": _VERSION, "turns": [t.to_dict() for t in self._turns]}
                self._path.parent.mkdir(parents=True, exist_ok=True)
                fd, tmp = tempfile.mkstemp(dir=self._path.parent, prefix=self._path.name, suffix=".tmp")
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(data, f)
                os.replace(tmp, self._path)  # readers see the old file or the new one, never half of one
                tmp = None
            except OSError as exc:
                log.warning("could not save the conversation to %s (%s: %s); continuing in memory",
                            self._path, type(exc).__name__, exc)
            finally:
                if tmp is not None:
                    try:
                        os.unlink(tmp)
                    except OSError:
                        pass  # the temp file is already gone, or cannot be removed either

    def _load(self) -> list[Turn]:
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            if data["version"] != _VERSION:
                raise ValueError(f"unsupported version {data['version']!r}")
            return [Turn.from_dict(t, self._lock, self._save) for t in data["turns"]]
        except FileNotFoundError:
            return []
        # Not JSON or not our shape. Other read errors (permissions, I/O) propagate: the file may be fine.
        except (ValueError, KeyError, TypeError) as exc:
            aside = self._path.with_name(self._path.name + ".corrupt")
            os.replace(self._path, aside)
            log.warning("could not read conversation %s (%s: %s); moved it to %s and starting empty",
                        self._path, type(exc).__name__, exc, aside)
            return []
