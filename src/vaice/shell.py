"""Shell commands for the assistant (`VAICE_SHELL`): the `run_shell` skill.

Every command is a `confirm` skill: nothing runs when the model asks. The request is on the screen with the exact
command line, and only the Run button runs it, through `/bin/zsh -c`. `check` is called again when the command runs,
so nothing but the stored string is trusted.

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
import signal
import subprocess
import threading
import time
from collections.abc import Callable
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

_SUDO = re.compile(r"\bsudo\b", re.IGNORECASE)  # anywhere, quoted or not: a false refusal beats a missed sudo

ASK = "I would run a command for this request. The command is on the screen; click Run to allow the run."


def check(command: str) -> str:
    """`command` if it may be asked about and run. Raises `SkillError` (the message is the result the model reads) for
    one that is refused outright."""
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
    return command


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
    """Run `command` (checked afresh) through `/bin/zsh -c`. Polls `stop()` and the deadline every
    `POLL_SECONDS`; on a stop or a timeout the command's whole process group is ended. Raises `SkillError`
    for a refused or unstartable command."""
    check(command)
    argv = ["/bin/zsh", "-c", command]
    if stop():  # stopped between the approval and here (New conversation, the app ending): nothing is started
        status = "the command was stopped"
        return SkillResult(status, f"$ {command}\n({status})")
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
    return SkillResult(speech, f"$ {command}\n({status})\n{output.text}".rstrip())


# --- the skill


def _command_of(args: dict) -> str:
    command = args.get("command")
    if not isinstance(command, str):
        raise SkillError("the command must be text")
    return command


def _ask(args: dict) -> Question:
    """The question, built here and never from the model's text; the panel shows the command itself."""
    check(_command_of(args))
    return Question(ASK)


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
