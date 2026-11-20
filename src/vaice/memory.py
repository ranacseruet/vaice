"""Long-term memory: short notes the user asked the assistant to keep, across conversations.

Explicit only: a note is stored when the user asks to remember something (the `remember` skill) and nothing is
extracted from what is said. The notes are read at the start of every reply and put into its instructions (see
`assistant.Assistant`), so a note is model-written text that every later prompt carries: an
injection path that outlives the conversation. Hence `clean_note` (one line, no control characters or chat
markers, a length cap), the cap on how many there are, and the skill being `direct_only` (see `skills.Skill`).

The notes live in `<VAICE_DATA_DIR>/memory.json` (mode 0600, written through a temp file and `os.replace`, like
the conversation), or only in memory when there is no path. "New conversation" does not touch them.
"""
import json
import logging
import re
import threading
import unicodedata
from pathlib import Path

from vaice.conversation import write_json_atomically
from vaice.skills import Question, Skill, SkillError, SkillResult, _plural

log = logging.getLogger(__name__)

MAX_NOTE_CHARS = 200
MAX_NOTES = 30
MIN_NOTE_WORDS = 3  # a fact about the user is a sentence, not a word
MIN_FORGET_CHARS = 3  # `forget("a")` would otherwise erase nearly everything with no confirmation
_VERSION = 1

# `<|im_start|>`, `<|anything|>`; and role/thinking tags a chat template or the markup stripper would act on
_CONTROL_TOKEN = re.compile(r"<\|.*?\|>")
_ROLE_TAG = re.compile(r"</?\s*(?:system|user|assistant|tool|tool_call|think|answer|function\b[^>]*)\s*>", re.I)
_ROLE_LABEL = re.compile(r"^(?:system|assistant|user|tool)\s*:\s*", re.I)
# A note bound to a moment is a reminder, not a fact about the user; vaice has nothing that would fire later.
# `timer` / `alarm` alone are nouns that facts use ("afraid of fire alarms"): only a request to set one counts
_REMINDER = re.compile(
    r"\b(?:remind(?:er|ed|s)?|tomorrow|tonight|today|in \d+ (?:minutes?|hours?)"
    # a singular weekday ("on Friday"); "works on Fridays" passes
    r"|(?:on|this|next|by) (?:monday|tuesday|wednesday|thursday|friday|saturday|sunday))\b"
    r"|\b(?:set|start|cancel)\b[^.]*\b(?:timers?|alarms?)\b|\b(?:timers?|alarms?) (?:for|at|in)\b", re.I)
# A clock time is a reminder too ("Call Mom at 9"), unless the note says it is a habit ("wakes up at 7 every day")
_CLOCK_TIME = re.compile(r"\bo'clock\b|\b\d{1,2}\s?(?:am|pm)\b|\bat \d{1,2}:\d{2}\b|\bat \d{1,2}\s*[.!?]*$", re.I)
_HABIT = re.compile(r"\b(?:every|each|daily|usually|always|normally|weekdays?|weekends?|mornings?|evenings?|nights?)\b",
                    re.I)
# The user's question written down as a note: "Where do I usually park my car?" as the first turn of a conversation
# became remember("The user asks where they usually park their car") on the 2B, 10 of 10 (live eval, no history).
# Only asking followed by a question word: "The user asks to be called Sam" is a preference and is kept
_QUESTION = re.compile(r"\?\s*$|^(?:the\s+)?user\s+(?:asks|asked|is\s+asking|wants\s+to\s+know|wonders|wondered)\s+"
                       r"(?:where|what|when|who|whom|whose|which|why|how|whether|if)\b", re.I)
_FENCE = re.compile(r"`{3,}")
_LIST_MARKER = re.compile(r"^(?:[-*+•>#]+|\d+[.)])(?:\s+|$)")


def _squash(text: str) -> str:
    """`text` with control and format characters (newlines, tabs, zero-width and bidi marks) as spaces,
    whitespace collapsed to single spaces."""
    return " ".join("".join(" " if unicodedata.category(c).startswith("C") else c for c in text).split())


def clean_note(text) -> str:
    """The note as it may be stored and put into a prompt: one line, no control characters, chat markers or
    code fences, no leading list marker. Raises `SkillError` (spoken) for text that is empty after that or longer
    than `MAX_NOTE_CHARS`."""
    if not isinstance(text, str):
        raise SkillError("the note must be text")
    note = _squash(text)
    note = _CONTROL_TOKEN.sub(" ", note)
    note = note.replace("<|", " ").replace("|>", " ")  # a token that was never closed
    note = _FENCE.sub(" ", _ROLE_TAG.sub(" ", note))
    note = " ".join(note.split())
    while True:  # "- - x", "1. # x"
        stripped = _ROLE_LABEL.sub("", _LIST_MARKER.sub("", note)).strip()
        if stripped == note:
            break
        note = stripped
    if len(note.split()) < MIN_NOTE_WORDS:  # "Goodbye": the 2B model saved a farewell as a note (live eval, 5 of 5)
        raise SkillError("there was nothing to remember")
    if _QUESTION.search(note):
        raise SkillError("nothing was saved: that is the user's question, not a fact about them; answer it")
    # "Call Mom tomorrow at 9": saved as a note, then announced as a reminder (live eval)
    if _REMINDER.search(note) or (_CLOCK_TIME.search(note) and not _HABIT.search(note)):
        raise SkillError("nothing was saved: I can't set reminders or timers, I only keep lasting facts about you")
    if len(note) > MAX_NOTE_CHARS:
        raise SkillError(f"that is too long to remember: keep a note under {MAX_NOTE_CHARS} characters")
    return note


def _key(note: str) -> str:
    return _squash(note).casefold()


class Memory:
    """An ordered list of notes, thread-safe. With a `path` it is loaded from it and saved after every change;
    without one (`VAICE_DATA_DIR=""`) nothing is read or written.

    A failed save is a warning (the note stays in memory until vaice stops). A file that cannot be read, or is
    not `{"version": 1, "notes": [str, ...]}`, is a warning and an empty memory; the file is left where it is,
    and the next successful save replaces it. Notes in a loaded file go through `clean_note` again and the ones
    that fail it are skipped, since the file is as good as any other source of prompt text."""

    def __init__(self, path: Path | None = None):
        self.path = path
        self._lock = threading.Lock()
        self._notes: list[str] = self._load() if path else []

    def __len__(self) -> int:
        with self._lock:
            return len(self._notes)

    def notes(self) -> list[str]:
        with self._lock:
            return list(self._notes)

    def has(self, note: str) -> bool:
        """Whether `note` (compared as `add` would store it, case ignored) is already kept."""
        try:
            key = _key(clean_note(note))
        except SkillError:
            return False
        with self._lock:
            return any(_key(kept) == key for kept in self._notes)

    def add(self, note: str) -> str:
        """Keep `note` and return it as stored (cleaned). A note already kept (ignoring case) is not added twice;
        the kept one comes back. Raises `SkillError` for a bad note or when `MAX_NOTES` are kept."""
        cleaned = clean_note(note)
        with self._lock:
            for kept in self._notes:
                if _key(kept) == _key(cleaned):
                    return kept
            if len(self._notes) >= MAX_NOTES:
                raise SkillError(f"my memory is full, {MAX_NOTES} notes: ask me to forget something first")
            self._notes.append(cleaned)
            self._save()
        return cleaned

    def remove(self, text: str) -> list[str]:
        """Remove every note that contains `text` (case ignored, whitespace collapsed); the removed notes."""
        needle = _key(text)
        if not needle:
            return []
        with self._lock:
            gone = [n for n in self._notes if needle in _key(n)]
            if gone:
                self._notes = [n for n in self._notes if n not in gone]
                self._save()
        return gone

    def clear(self) -> int:
        """Forget every note; how many there were."""
        with self._lock:
            count = len(self._notes)
            self._notes = []
            if count:
                self._save()
        return count

    def _save(self) -> None:
        """Call with the lock held (it also serializes the writers)."""
        if self.path is None:
            return
        try:
            write_json_atomically(self.path, {"version": _VERSION, "notes": self._notes})
        except OSError as exc:
            log.warning("could not save the notes to %s (%s: %s); keeping them in memory until vaice stops",
                        self.path, type(exc).__name__, exc)

    def _load(self) -> list[str]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if data["version"] != _VERSION:
                raise ValueError(f"unsupported version {data['version']!r}")
            raw = data["notes"]
            if not isinstance(raw, list) or not all(isinstance(n, str) for n in raw):
                raise ValueError("notes must be a list of text")
        except FileNotFoundError:
            return []
        except (OSError, ValueError, KeyError, TypeError) as exc:  # unreadable or not our shape
            log.warning("could not read the notes in %s (%s: %s); starting with none, and the file is left "
                        "as it is until a note is saved", self.path, type(exc).__name__, exc)
            return []
        notes: list[str] = []
        skipped = 0
        for text in raw:
            try:
                note = clean_note(text)
            except SkillError:
                skipped += 1
                continue
            if _key(note) in {_key(n) for n in notes} or len(notes) >= MAX_NOTES:
                skipped += 1
                continue
            notes.append(note)
        if skipped:
            log.warning("skipped %d note(s) in %s that were empty, too long, repeated or beyond the limit of %d",
                        skipped, self.path, MAX_NOTES)
        return notes


# --- the skills

REFUSED_AFTER_TOOL = "I only remember what you tell me directly, not what a tool reported."


def _remember(memory: Memory):
    def run(args: dict) -> SkillResult:
        note = args.get("note")
        known = isinstance(note, str) and memory.has(note)
        stored = memory.add(note)  # raises SkillError for a bad note or a full memory
        return SkillResult(f"already remembered: {stored}" if known else f"remembered: {stored}")

    return run


def _forget(memory: Memory):
    def run(args: dict) -> SkillResult:
        about = args.get("about")
        if not isinstance(about, str) or len(_squash(about)) < MIN_FORGET_CHARS:
            raise SkillError(f"say a word or phrase of at least {MIN_FORGET_CHARS} letters from the note to forget")
        about = _squash(about)[:MAX_NOTE_CHARS]
        gone = memory.remove(about)
        return SkillResult(f"forgot: {'; '.join(gone)}" if gone else f"no note mentions {about}")

    return run


def _ask_forget_everything(memory: Memory):
    def ask(_args: dict) -> Question:
        count = len(memory)
        if not count:
            raise SkillError("there are no notes to erase")
        return Question(f"Shall I erase all {count} notes I keep?" if count > 1
                        else "Shall I erase the one note I keep?")

    return ask


def _forget_everything(memory: Memory):
    def run(_args: dict) -> SkillResult:
        return SkillResult(f"erased {_plural(memory.clear(), 'note')}")

    return run


def memory_skills(memory: Memory) -> list[Skill]:
    """`remember`, `forget` and `forget_everything` over `memory`. `remember` is `direct_only`: it does not store
    anything once another skill has run in the same reply (what a tool reported is not what the user said)."""
    return [
        Skill("remember", "Save a lasting fact about the user or their preferences, only when the user asks you to "
                          "remember it; write it in the third person, such as The user parks on level three. "
                          "Not for reminders, timers or things to do later: you cannot remind the user.",
              {"type": "object",
               "properties": {"note": {"type": "string",
                                       "description": "The note, one short sentence about the user."}},
               "required": ["note"]},
              _remember(memory), kind="action", direct_only=True),
        Skill("forget", "Delete the saved notes that mention a word, when the user asks you to forget something.",
              {"type": "object",
               "properties": {"about": {"type": "string",
                                        "description": "One distinctive word from the note, such as sister."}},
               "required": ["about"]},
              _forget(memory), kind="action"),
        Skill("forget_everything", "Erase every saved note about the user. The user is asked to confirm first.",
              {"type": "object", "properties": {}}, _forget_everything(memory), kind="confirm",
              ask=_ask_forget_everything(memory)),
    ]
