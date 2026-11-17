"""Shell commands for assistant mode (`VAICE_ASSISTANT_SHELL`): the `run_shell` skill.

Every command is a `confirm` skill: nothing runs when the model asks. `classify` decides how it may be
approved, from the command string alone (it is called again when the command runs, so nothing else is trusted):

- by voice or click: a command vaice can show to be harmless. The string holds no shell syntax at all, parses
  with `shlex` into one program and its arguments, and the program is a bare name from `SAFE_PROGRAMS`, which
  only list or report. It runs as an argv list, with no shell. A leading `~/` in an argument is the only
  expansion, done by vaice (and only in a command without quotes, so the shell would have agreed).
- by click only: everything else, through `/bin/zsh -c`. A spoken yes does not approve it.

The set is an allowlist, not a denylist: a list of read-only program names is not read-only (`find -delete`,
`sed -i`, `sort -o`, `awk` with `system()`, `xargs`, `env`, `tee`), so the voice set is tiny and its arguments
are checked. Programs that print file contents (`cat`, `head`) are click-only on purpose: their output goes to
the model.

`sudo` is refused before any question: it needs an administrator password, which vaice cannot give. The check
is a courtesy, not a boundary (`su"do"` gets past it); the boundary is that a person reads the command and clicks.
"""
import atexit
import codecs
import contextlib
import logging
import os
import re
import select
import shlex
import signal
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from vaice.skills import Question, Skill, SkillError, SkillResult

log = logging.getLogger(__name__)

RUN_SHELL = "run_shell"
MAX_COMMAND = 500  # characters
SHELL_TIMEOUT = 30  # seconds a command may run
SHELL_OUTPUT_CHARS = 20000  # output kept; the rest is read and thrown away so the command never blocks on a full pipe
SPEECH_OUTPUT_CHARS = 600  # output the model gets to answer from
POLL_SECONDS = 0.05  # how often a running command is checked against `stop()` and the deadline
KILL_WAIT = 1.0  # seconds between SIGTERM and SIGKILL
LEFTOVER_WAIT = 0.5  # seconds a finished command's background process may keep its output open
DRAIN_WAIT = 2.0  # seconds to read what is left after a command was killed

SAFE_PROGRAMS = frozenset({"ls", "df", "du", "date", "uptime", "whoami", "pwd", "sw_vers", "hostname", "uname"})
NO_ARGUMENTS = frozenset({"hostname", "whoami", "pwd", "uptime"})  # `hostname NAME` renames the Mac
_SYNTAX = frozenset("|&;<>()$`{}*?[]!#\\")  # with any control character (a newline too): shell syntax
_SUDO = re.compile(r"\bsudo\b", re.IGNORECASE)  # anywhere, quoted or not: a false refusal beats a missed sudo

ASK_VOICE = "Shall I run the command {program}? The full command is on the screen."
ASK_CLICK = "I would run a command for this request. The command is on the screen; click Run to allow the run."
NEEDS_BUTTON = "This one needs the Run button on the screen."  # a spoken yes to a click-only request


@dataclass(frozen=True)
class Plan:
    """How a command runs: `argv` for a harmless one (no shell), None for one that needs `/bin/zsh -c`."""

    command: str
    argv: list[str] | None
    program: str  # of an argv plan; empty for a shell one

    @property
    def voice(self) -> bool:
        """Whether a spoken yes may approve it (a click always may)."""
        return self.argv is not None


def _harmless_argv(command: str) -> list[str] | None:
    """The argv of a command that is harmless by its form, else None."""
    if any(c in _SYNTAX or not c.isprintable() for c in command):
        return None
    if "~" in command and ("'" in command or '"' in command):  # quoted, a shell would not expand it
        return None
    try:
        words = shlex.split(command)
    except ValueError:  # an unclosed quote: let the shell say so
        return None
    if not words or words[0] not in SAFE_PROGRAMS:  # a path (`/bin/ls`) is never in the set
        return None
    program, args = words[0], words[1:]
    if program in NO_ARGUMENTS and args:
        return None
    if program == "date" and not all(arg.startswith(("+", "-")) for arg in args):  # a bare one sets the clock
        return None
    argv = [program]
    for arg in args:
        if arg.startswith("~/"):
            arg = str(Path.home()) + arg[1:]
        elif "~" in arg:  # `~`, `~user`: a shell would expand them, argv would not
            return None
        argv.append(arg)
    return argv


def classify(command: str) -> Plan:
    """How `command` may run. Raises `SkillError` (spoken) for one that is refused outright."""
    if not isinstance(command, str):
        raise SkillError("the command must be text")
    if not command.strip():
        raise SkillError("no command was given")
    if len(command) > MAX_COMMAND:
        raise SkillError(f"that command is longer than {MAX_COMMAND} characters")
    if "\0" in command:
        raise SkillError("a command cannot hold a null character")
    if _SUDO.search(command):
        raise SkillError("sudo needs an administrator password, which vaice cannot give")
    argv = _harmless_argv(command)
    return Plan(command, argv, argv[0] if argv else "")


# --- running

_live: set[subprocess.Popen] = set()  # commands in flight, for the app's exit
_live_lock = threading.Lock()


def _signal_group(pgid: int, sig: int) -> None:
    # the group is gone (ESRCH), or only zombies are left (macOS answers EPERM): nothing to stop
    with contextlib.suppress(OSError):
        os.killpg(pgid, sig)


def _stop_group(proc: subprocess.Popen) -> None:
    """End the command's whole process group: SIGTERM, a short wait, then SIGKILL for whatever is left (also
    the children of a shell that already exited)."""
    _signal_group(proc.pid, signal.SIGTERM)  # the command is its own session leader: its pid is its group's id
    with contextlib.suppress(subprocess.TimeoutExpired):
        proc.wait(timeout=KILL_WAIT)
    _signal_group(proc.pid, signal.SIGKILL)
    try:
        proc.wait(timeout=KILL_WAIT)
    except subprocess.TimeoutExpired:
        log.warning("a command did not end after SIGKILL (pid %d)", proc.pid)


@atexit.register
def _stop_all() -> None:
    """The app is ending: a command has a session of its own, so nothing else would stop it."""
    with _live_lock:
        running = list(_live)
    for proc in running:
        _stop_group(proc)


class _Output:
    """The command's output as text: the first `SHELL_OUTPUT_CHARS` characters; `cut` says there was more."""

    def __init__(self) -> None:
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self._parts: list[str] = []
        self._size = 0
        self.cut = False

    def add(self, data: bytes) -> None:
        text = self._decoder.decode(data)
        room = SHELL_OUTPUT_CHARS - self._size
        if len(text) > room:
            text, self.cut = text[:room], True
        self._parts.append(text)
        self._size += len(text)

    @property
    def text(self) -> str:
        return "".join(self._parts)


def _read(fd: int, output: _Output, wait: float) -> bool:
    """Read what `fd` has within `wait` seconds into `output`; True at the end of the stream."""
    if not select.select([fd], [], [], wait)[0]:
        return False
    data = os.read(fd, 65536)
    output.add(data)
    return not data


def _status(proc: subprocess.Popen, ended: str | None) -> str:
    if ended == "timeout":
        return f"the command was stopped after {SHELL_TIMEOUT:g} seconds"
    if ended == "stopped":
        return "the command was stopped"
    return "the command finished" if proc.returncode == 0 else f"the command failed with status {proc.returncode}"


def run(command: str, stop: Callable[[], bool]) -> SkillResult:
    """Run `command` (classified afresh), however it was approved. Polls `stop()` and the deadline every
    `POLL_SECONDS`; on a stop or a timeout the command's whole process group is ended. Raises `SkillError`
    for a refused or unstartable command."""
    plan = classify(command)
    argv = plan.argv or ["/bin/zsh", "-c", plan.command]
    if stop():  # stopped between the approval and here (New conversation, a barge-in): nothing is started
        status = "the command was stopped"
        return SkillResult(status, f"$ {plan.command}\n({status})")
    try:
        proc = subprocess.Popen(argv, cwd=Path.home(), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, start_new_session=True)
    except FileNotFoundError:
        raise SkillError(f"{argv[0]} is not available") from None
    except OSError as exc:
        raise SkillError(f"the command could not be started: {exc.strerror or exc}") from None
    with _live_lock:
        _live.add(proc)
    output = _Output()
    ended: str | None = None  # "timeout" or "stopped" when vaice ended it
    try:
        assert proc.stdout is not None
        fd = proc.stdout.fileno()
        deadline = time.monotonic() + SHELL_TIMEOUT
        eof, exited_at = False, None
        while True:
            if eof:
                with contextlib.suppress(subprocess.TimeoutExpired):
                    proc.wait(timeout=POLL_SECONDS)
            else:
                eof = _read(fd, output, POLL_SECONDS)
            now = time.monotonic()
            if proc.poll() is not None:
                if eof:
                    # whatever it left running in its group (`server >/dev/null &`) ends with it: nothing
                    # would time it out or stop it later, since only the command itself is tracked
                    _stop_group(proc)
                    break
                exited_at = exited_at or now
                if now - exited_at >= LEFTOVER_WAIT:  # a background process keeps the output open: end it too
                    _stop_group(proc)
                    break
            if stop():
                ended = "stopped"
            elif now >= deadline:
                ended = "timeout"
            if ended:
                _stop_group(proc)
                break
        if not eof:  # killed: what the group wrote before it died, at most `DRAIN_WAIT` more
            drained = time.monotonic() + DRAIN_WAIT
            while not eof and time.monotonic() < drained:
                eof = _read(fd, output, POLL_SECONDS)
    finally:
        with _live_lock:
            _live.discard(proc)
        if proc.poll() is None:  # an error above: never leave it running
            _stop_group(proc)
        if proc.stdout is not None:
            proc.stdout.close()

    status = _status(proc, ended)
    flat = " ".join(output.text.split())
    speech = status
    if flat:
        speech += f". Output: {flat[:SPEECH_OUTPUT_CHARS]}"
        if len(flat) > SPEECH_OUTPUT_CHARS or output.cut:
            speech += " (more on screen)"
    return SkillResult(speech, f"$ {plan.command}\n({status})\n{output.text}".rstrip())


# --- the skill


def _command_of(args: dict) -> str:
    command = args.get("command")
    if not isinstance(command, str):
        raise SkillError("the command must be text")
    return command


def _ask(args: dict) -> Question:
    """The question, built here and never from the model's text; the command's arguments are not read aloud."""
    plan = classify(_command_of(args))
    if plan.voice:
        return Question(ASK_VOICE.format(program=plan.program))
    return Question(ASK_CLICK, voice=False)


def _run(args: dict, stop: Callable[[], bool]) -> SkillResult:
    return run(_command_of(args), stop)


def shell_skill() -> Skill:
    return Skill(
        RUN_SHELL,
        "Run a command in the terminal of this Mac. Only when the user says to run a command or to use the "
        "terminal. The user must approve it first.",
        {"type": "object",
         "properties": {"command": {"type": "string", "description": "The exact command line to run."}},
         "required": ["command"]},
        lambda args: _run(args, lambda: False), kind="confirm", ask=_ask, run_stoppable=_run)
