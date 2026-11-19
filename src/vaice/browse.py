"""A browsing agent for assistant mode: `PiAgent` hands a goal ("open python.org and tell me its title") to the Pi
coding agent (`pi`) running in its RPC mode, which drives Chrome through the chrome-devtools MCP server, and returns
the answer text. Nothing here is wired into the app yet (a later change adds the `browse` skill and its settings).

How it fits together:

- **One long-lived child** (`pi --mode rpc`, started on the first task, restarted after it dies), speaking JSONL on
  stdin/stdout. Its own session (`start_new_session`), stderr appended to a 0600 log file, an `atexit` backstop that
  `close()` unregisters. One task at a time; a task runs on the caller's thread and polls `stop()` every
  `POLL_SECONDS`, like `shell.run`.
- **Isolated from the user's Pi.** `write_pi_home` writes a config directory of its own (`models.json`,
  `settings.json`, `mcp.json`; the child gets it as `PI_CODING_AGENT_DIR`, with `HOME` inside it), so the user's Pi
  skills, extensions, models and sessions are never loaded. The child sees an allowlisted environment, not vaice's
  (the other API keys stay out), plus the variables the caller passes. The model's API key reaches Pi through that
  environment (`"$NAME"` interpolation in `models.json`), never a file.
- **Every task starts with `new_session`** (an abort does not clear Pi's context), and `prompt` is sent only after
  that command's response: sent back to back, the prompt starts on the old session, is replaced, and never settles
  (measured against Pi 1.5.2; one run hung for hours). `agent_end` is not the end of a task, `agent_settled` is.
- **The gate** (`browse_gate.ts`, a Pi extension) classifies each tool call by `GATE_POLICY`: the read-only browser
  tools run, the ones that change a page ask, anything else asks. An ask arrives as a `confirm` dialog and goes to
  `on_ask`. A refusal also aborts the task: a model told no routes around it (retries, another tool, navigates
  directly). Pi carries on without an extension that failed to load, so the start checks `get_commands` for the
  gate's `vaice-gate` command and refuses to run tasks without it. An extension error during a task aborts it.
- **The answer is the text of the last assistant message that had any**, not every delta: "I'll open the page"
  before a tool call is chatter, not the answer.
- **Ending a task early** (`stop()`, the deadline, a refused step) sends `abort`, which settles in about 10 ms. A
  child that does not settle in `ABORT_WAIT` is ended (SIGTERM to its group, then SIGKILL) and replaced by the next
  task. A child that exits mid-task is a `BrowseError`, and the next task restarts it.

Pi stops its MCP servers when it shuts down, which is why `close()` closes stdin and waits before it signals.
A SIGKILLed Pi can leave its MCP server and the browser it started running: they are NOT in Pi's process group
(measured against Pi 1.5.2: `npm exec` and chrome-devtools-mcp each lead a group of their own, Chrome another), so
`killpg` does not reach them. With a `chrome_profile`, the processes that hold that profile (`--user-data-dir=` for
Chrome, `--userDataDir=` for chrome-devtools-mcp and the `npm exec` that started it) are ended before every new child
is started and after `close()`: only those, matched on the exact profile path, never another Chrome.

`BrowseAgent` is what the rest of vaice depends on (the `browse` skill and the responder), so another backend can
take Pi's place. Failures are `BrowseError`s with a `spoken` sentence; `BrowseStopped` is the caller's own stop. No
message, log line or error holds the API key or the page's content: tool names and timings only.
"""
import atexit
import contextlib
import json
import logging
import os
import queue
import re
import select
import signal
import subprocess
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from vaice.conversation import write_json_atomically

log = logging.getLogger(__name__)

POLL_SECONDS = 0.05  # how often a running task is checked against `stop()` and the deadline
START_TIMEOUT = 30.0  # seconds a fresh child gets to answer its first command (Pi, its extension, the config)
NEW_SESSION_TIMEOUT = 10.0  # seconds `new_session` may take before the child counts as wedged
ABORT_WAIT = 10.0  # seconds after `abort` for `agent_settled`, normally about 10 ms; then the child is ended
CLOSE_WAIT = 5.0  # seconds a child gets to exit after its stdin is closed
TERM_WAIT = 3.0  # seconds between SIGTERM and SIGKILL
KILL_WAIT = 2.0  # seconds to reap a SIGKILLed child
WRITE_TIMEOUT = 5.0  # seconds one command may take to go into the child's stdin; then the child counts as wedged
ASK_GRACE = 5.0  # seconds the gate waits longer than `ask_timeout_s`, so vaice's own "no" (and abort) comes first
PS_TIMEOUT = 10.0  # seconds `ps` may take when looking for what still holds the browser profile
MIN_PI_VERSION = (1, 5, 2)  # the Pi this module was measured against (its RPC mode and extension API)

CHROME_DEVTOOLS_MCP = "chrome-devtools-mcp@1.13.0"  # pinned: its flags and tool names are what the policy names
API_KEY_ENV = "VAICE_PI_API_KEY"  # the usual name of the child's environment variable holding the model's key
GATE_FILE = Path(__file__).with_name("browse_gate.ts")
GATE_COMMAND = "vaice-gate"  # the command browse_gate.ts registers
PROVIDER = "vaice"  # the provider id in models.json

_MUTATING_TOOLS = ("click", "fill", "fill_form", "type_text", "press_key", "evaluate_script", "upload_file", "drag",
                   "handle_dialog")
GATE_POLICY = {
    "mcp__chrome__*": "run",
    **{f"mcp__chrome__{name}": "ask" for name in _MUTATING_TOOLS},
    "default": "ask",
}  # tools Pi knows by exact name win over the `*` pattern; anything not named asks

# what the child may inherit: enough to find node and reach the network, nothing else (not vaice's other keys)
_INHERITED_ENV = ("PATH", "TMPDIR", "LANG", "LC_ALL", "LC_CTYPE", "TERM", "USER", "LOGNAME", "SHELL", "SSL_CERT_FILE",
                  "SSL_CERT_DIR", "NODE_EXTRA_CA_CERTS", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY",
                  "http_proxy", "https_proxy", "all_proxy", "no_proxy", "npm_config_cache", "npm_config_registry")
_EOF: dict = {"type": "_eof"}  # the reader's last record, compared by identity
_DIALOGS = ("select", "confirm", "input", "editor")  # the extension UI methods that wait for an answer
_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")

PROMPT = "Use the chrome browser tools to do this task in the browser, then answer in plain text. Task: {goal}"
CANT_START = "I couldn't start the browser agent."
FAILED = "The browser agent ran into a problem."
NO_ANSWER = "The browser agent did not give an answer."
DECLINED = "Okay, I stopped there."


class BrowseError(RuntimeError):
    """The browser task failed. `str(exc)` is for the log and the UI; `spoken` is the short sentence to say."""

    def __init__(self, message: str, spoken: str = FAILED):
        super().__init__(message)
        self.spoken = spoken


class BrowseDeclined(BrowseError):
    """The user refused a step; the task was aborted."""

    def __init__(self) -> None:
        super().__init__("the user did not approve a step", DECLINED)


class BrowseStopped(Exception):
    """`stop()` ended the task (a barge-in, the app stopping); the browser agent was aborted."""


class BrowseAgent(Protocol):
    """A browser agent: does a goal in the browser and returns the answer text. `PiAgent` is the one backend."""

    def run(self, goal: str, *, stop: Callable[[], bool], on_progress: Callable[[str, dict], None],
            on_ask: Callable[[str, str, Callable[[], bool]], bool], deadline_s: float) -> str:
        """Do `goal`. `on_progress(tool, args)` is told of each tool as it starts; `on_ask(title, message, stop)`
        answers a question about a step that changes a page (True = go ahead) and may block, honoring its `stop`.
        Raises `BrowseStopped` when `stop()` ended it, `BrowseDeclined` after a no, `BrowseError` otherwise."""
        ...

    def close(self) -> None:
        """End the agent and what it started. Idempotent."""
        ...


# --- what is left when a child was killed


def _profile_pids(profile: Path, listing: str) -> list[int]:
    """The pids in `listing` (`ps -axo pid=,command=`) whose command line holds `--user-data-dir=<profile>` or
    `--userDataDir=<profile>` as a whole argument: the path must be followed by white space or the end, so a sibling
    such as `<profile>-2` is not matched. Never this process."""
    pattern = re.compile(rf"(?:^|\s)--(?:user-data-dir|userDataDir)={re.escape(str(profile))}(?=\s|$)")
    pids = []
    for line in listing.splitlines():
        pid, _, command = line.strip().partition(" ")
        if pid.isdigit() and int(pid) != os.getpid() and pattern.search(command):
            pids.append(int(pid))
    return pids


def end_profile_processes(profile: Path) -> list[int]:
    """End whatever still has `profile` open (SIGTERM, then SIGKILL after `TERM_WAIT`) and return the pids found.
    The browser a killed Pi started is not in Pi's process group, so nothing else ends it, and it holds the profile's
    lock. Best effort: a `ps` that fails is a warning, and nothing but exact profile matches is ever signalled."""
    try:
        listing = subprocess.run(["ps", "-axww", "-o", "pid=,command="], capture_output=True, text=True,
                                 stdin=subprocess.DEVNULL, timeout=PS_TIMEOUT, check=True).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        log.warning("browse: could not look for processes holding the browser profile: %s", type(exc).__name__)
        return []
    pids = _profile_pids(profile, listing)
    if not pids:
        return []
    log.info("browse: ending %d process(es) still holding the browser profile", len(pids))
    _signal(pids, signal.SIGTERM)
    give_up = time.monotonic() + TERM_WAIT
    while time.monotonic() < give_up and any(_alive(pid) for pid in pids):
        time.sleep(POLL_SECONDS)
    _signal([pid for pid in pids if _alive(pid)], signal.SIGKILL)
    return pids


def _signal(pids: Sequence[int], sig: int) -> None:
    for pid in pids:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.kill(pid, sig)


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:  # a child of ours that ended is a zombie until reaped: not alive for this purpose
        return os.waitpid(pid, os.WNOHANG) == (0, 0)
    except ChildProcessError:
        return True  # not ours: it is there


def pi_version(command: str = "pi") -> tuple[int, ...] | None:
    """The version `command --version` prints (`1.5.2`) as a tuple, or None when it cannot be read."""
    try:
        done = subprocess.run([command, "--version"], capture_output=True, text=True, stdin=subprocess.DEVNULL,
                              timeout=PS_TIMEOUT, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    match = re.search(r"(?<![\d.])(\d+)\.(\d+)\.(\d+)(?![\d.])", f"{done.stdout}\n{done.stderr}")
    return tuple(int(part) for part in match.groups()) if match else None


# --- Pi's config directory


def home_dir(pi_home: Path) -> Path:
    """The child's `HOME`: inside the config directory, so nothing of the user's is read or written."""
    return pi_home / "home"


def work_dir(pi_home: Path) -> Path:
    """The child's working directory (Pi has no built-in tools here; it only needs one that holds no project)."""
    return pi_home / "work"


def default_browser_args(profile_dir: Path) -> list[str]:
    """chrome-devtools-mcp's arguments for the everyday setup: a dedicated persistent profile and a visible
    window (so the user sees what is clicked, and a login survives). Not `--isolated`, not `--headless`."""
    return [f"--userDataDir={profile_dir}"]


def write_pi_home(path: Path, *, base_url: str, model: str, api_key_env: str | None,
                  browser_args: Sequence[str]) -> None:
    """Write the config directory the child runs from (`path`, mode 0700; its files 0600): `models.json` (one
    OpenAI-compatible provider `vaice` at `base_url`), `settings.json` (it is the default, project trust off),
    `mcp.json` (the pinned chrome-devtools server, `browser_args` after the package). `api_key_env` is the NAME of
    the child's environment variable holding the key (`"$NAME"` goes into the file; the key itself never does);
    `None` writes a placeholder, which Pi wants for an endpoint that needs none. Raises OSError."""
    if api_key_env is not None and not _ENV_NAME.match(api_key_env):
        raise ValueError("api_key_env must be an environment variable name")
    for directory in (path, home_dir(path), work_dir(path)):
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    write_json_atomically(path / "models.json", {"providers": {PROVIDER: {
        "baseUrl": base_url,
        "api": "openai-completions",
        "apiKey": "vaice" if api_key_env is None else f"${api_key_env}",
        "models": [{"id": model}],
    }}})
    write_json_atomically(path / "settings.json", {
        "defaultProvider": PROVIDER, "defaultModel": model, "defaultProjectTrust": "never"})
    write_json_atomically(path / "mcp.json", {"mcpServers": {"chrome": {
        "command": "npx",
        # no usage statistics to Google, and no trace URLs to the CrUX API (no trace tool is wanted anyway)
        "args": ["-y", CHROME_DEVTOOLS_MCP, "--no-usage-statistics", "--no-performance-crux", *browser_args],
        "exposure": "direct",
    }}})


# --- the child


class _Child:
    """One `pi --mode rpc` process: its stdin, and a reader thread that parses its stdout into `events`."""

    def __init__(self, proc: subprocess.Popen):
        self.proc = proc
        self.events: queue.Queue[dict] = queue.Queue()
        self.dead = False  # set when it is known unusable (ended, wedged): the next task starts a new one
        self._ids = 0
        self._ended = False
        self._end_lock = threading.Lock()
        assert proc.stdin is not None
        self._stdin_fd = proc.stdin.fileno()
        os.set_blocking(self._stdin_fd, False)  # writes go through os.write + select, bounded (see write)
        threading.Thread(target=self._read, name="vaice-browse-reader", daemon=True).start()

    def _read(self) -> None:
        assert self.proc.stdout is not None
        # a binary pipe: readline() splits on LF only (U+2028 is valid inside a JSON string)
        with self.proc.stdout:
            for line in iter(self.proc.stdout.readline, b""):
                try:
                    record = json.loads(line.rstrip(b"\r\n"))
                except ValueError:
                    record = None
                self.events.put(record if isinstance(record, dict) else {"type": "_unparsed", "size": len(line)})
        self.dead = True  # at once, not when the task thread reads _EOF: an open question polls this
        self.events.put(_EOF)

    def next_id(self) -> str:
        self._ids += 1
        return f"v{self._ids}"

    def write(self, record: dict) -> None:
        """Send one JSONL record. Bounded: a child that stops reading its stdin (a full pipe) is ended after
        `WRITE_TIMEOUT`, and one marked dead (`close()`, `end()`) stops the write within `POLL_SECONDS`. The
        buffered `proc.stdin` object is never written to, so closing it never waits on a blocked writer."""
        data = json.dumps(record).encode("utf-8") + b"\n"
        give_up = time.monotonic() + WRITE_TIMEOUT
        while data:
            if self.dead:
                raise BrowseError("the browser agent's process is gone", FAILED)
            if time.monotonic() >= give_up:
                self.end()
                raise BrowseError("the browser agent stopped reading its input", FAILED)
            try:
                _, writable, _ = select.select([], [self._stdin_fd], [], POLL_SECONDS)
                if writable:
                    data = data[os.write(self._stdin_fd, data):]
            except BlockingIOError:
                continue
            except (OSError, ValueError):  # a closed pipe: the child is gone
                self.dead = True
                raise BrowseError("the browser agent's process is gone", FAILED) from None

    def send(self, command: str, **fields) -> str:
        """Send a command with a fresh id and return the id."""
        rid = self.next_id()
        self.write({"id": rid, "type": command, **fields})
        return rid

    def end(self) -> None:
        """Mark it unusable and end it: SIGTERM to its process group (Pi then stops its MCP servers), then, after
        Pi exits or `TERM_WAIT` passes, SIGKILL to the group anyway (a member that ignores SIGTERM must not outlive
        Pi), reap it, close stdin. Whatever it left running in the group ends with it, as in `shell.run`. Bounded:
        stdout is closed by the reader at EOF, never here, because a process outside the group (one that called
        setsid) may still hold the pipe, and closing a buffered pipe the reader is blocked on would wait for it.
        Once only."""
        with self._end_lock:
            self.dead = True
            if self._ended:
                return
            self._ended = True
        proc = self.proc
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(proc.pid, signal.SIGTERM)
        with contextlib.suppress(subprocess.TimeoutExpired):
            proc.wait(timeout=TERM_WAIT)
        with contextlib.suppress(ProcessLookupError, PermissionError):  # the rest of the group, Pi exited or not
            os.killpg(proc.pid, signal.SIGKILL)
        try:
            proc.wait(timeout=KILL_WAIT)
        except subprocess.TimeoutExpired:
            log.warning("browse: the browser agent (pid %d) did not exit after SIGKILL", proc.pid)
        self.release()

    def release(self) -> None:
        """Close the parent's end of stdin (stdout belongs to the reader thread, which closes it at EOF)."""
        if self.proc.stdin is not None:
            with contextlib.suppress(OSError, ValueError):
                self.proc.stdin.close()


@dataclass(frozen=True)
class _Budget:
    """What ends a task early: the caller's `stop()` and the deadline (`seconds` after the task began)."""

    stop: Callable[[], bool]
    deadline: float
    seconds: float

    def check(self) -> None:
        """Raise what ends the task now, if anything (the caller aborts the child first when one is running)."""
        if self.stop():
            raise BrowseStopped
        if time.monotonic() >= self.deadline:
            raise BrowseError(f"the task's {self.seconds:g} seconds passed",
                              f"I stopped the browser task after {self.seconds:g} seconds.")


class PiAgent:
    """Drives one Pi RPC child. `run()` is thread-safe (tasks are serialized); `close()` ends the child."""

    def __init__(self, pi_home: Path, *, log_path: Path, command: str | Sequence[str] = "pi",
                 ask_timeout_s: float = 60.0, env: Mapping[str, str] | None = None,
                 chrome_profile: Path | None = None):
        """`pi_home` is a directory `write_pi_home` filled. `log_path` receives the child's stderr (0600).
        `command` is the pi executable (or a whole argv prefix). `ask_timeout_s` is how long a question gets before
        it counts as a no. `env` is added to the child's environment: the model's API key goes here.
        `chrome_profile` is the browser profile `write_pi_home` was given (`--userDataDir`): what still holds it is
        ended before a new child starts and after `close()` (see the module docstring)."""
        self._home = pi_home
        self._log_path = log_path
        self._argv = [command] if isinstance(command, str) else list(command)
        self._ask_timeout = ask_timeout_s
        self._extra_env = dict(env or {})
        self._chrome_profile = chrome_profile
        self._child: _Child | None = None
        self._closed = False
        self._registered = False
        self._state_lock = threading.Lock()  # the child, `_closed` (close() must never wait for a running task)
        self._run_lock = threading.Lock()  # one task at a time

    # -- starting

    def _environment(self) -> dict[str, str]:
        env = {name: os.environ[name] for name in _INHERITED_ENV if name in os.environ}
        env.setdefault("npm_config_cache", str(Path.home() / ".npm"))  # HOME moves, the package cache must not
        env.update(PI_CODING_AGENT_DIR=str(self._home), HOME=str(home_dir(self._home)), PI_OFFLINE="1",
                   PI_SKIP_VERSION_CHECK="1", PI_TELEMETRY="0", VAICE_GATE_POLICY=json.dumps(GATE_POLICY),
                   VAICE_GATE_TIMEOUT_MS=str(int((self._ask_timeout + ASK_GRACE) * 1000)))
        env.update(self._extra_env)
        return env

    def _scrub(self, text: str) -> str:
        """`text` for a log line or an error: the child's secrets (every `env` value) blanked."""
        for value in self._extra_env.values():
            if value:
                text = text.replace(value, "***")
        return text

    def _spawn(self) -> _Child:
        home_dir(self._home).mkdir(mode=0o700, parents=True, exist_ok=True)
        work = work_dir(self._home)
        work.mkdir(mode=0o700, parents=True, exist_ok=True)
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        argv = [*self._argv, "--mode", "rpc", "--no-session", "--no-builtin-tools", "--no-context-files",
                "--no-skills", "--no-prompt-templates", "--offline", "-e", str(GATE_FILE)]
        # not --no-extensions: that would turn off Pi's built-in MCP client too
        err_fd = os.open(self._log_path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        try:
            proc = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=err_fd, cwd=work,
                                    env=self._environment(), start_new_session=True)
        except OSError as exc:
            raise BrowseError(f"could not start the browser agent: {exc.strerror or type(exc).__name__}",
                              CANT_START) from None
        finally:
            os.close(err_fd)  # the child has its own copy
        return _Child(proc)

    def _ensure_child(self, budget: _Budget) -> _Child:
        with self._state_lock:
            if self._closed:
                raise BrowseError("the browser agent was closed")
            child = self._child
            if child is not None and not child.dead and child.proc.poll() is None:
                return child
            if child is not None:
                child.end()  # reaps it and what it left in its group
            if self._chrome_profile is not None:
                end_profile_processes(self._chrome_profile)  # the browser it started is in no group of Pi's
            child = self._child = self._spawn()
            if not self._registered:
                self._registered = True
                atexit.register(self.close)  # backstop; main()'s finally closes first
        log.info("browse: started the browser agent (pid %d)", child.proc.pid)
        try:
            reply = self._await_response(child, child.send("get_commands"), START_TIMEOUT, budget)
            names = {c.get("name") for c in (reply.get("data") or {}).get("commands", []) if isinstance(c, dict)}
            if GATE_COMMAND not in names:  # Pi ran on without the gate: its tools would be ungated
                raise BrowseError("the browser agent started without its tool gate", CANT_START)
        except BaseException:
            child.end()
            raise
        return child

    # -- talking to it

    def _await_response(self, child: _Child, rid: str, timeout: float, budget: _Budget) -> dict:
        """The response to command `rid`, skipping other records. A child that ends, or does not answer in
        `timeout`, is ended."""
        limit = time.monotonic() + timeout
        while True:
            budget.check()
            if time.monotonic() >= limit:
                child.end()
                raise BrowseError("the browser agent did not answer", FAILED)
            try:
                event = child.events.get(timeout=POLL_SECONDS)
            except queue.Empty:
                continue
            if event is _EOF:
                child.dead = True
                raise BrowseError("the browser agent exited", FAILED)
            if event.get("type") == "response" and event.get("id") == rid:
                if not event.get("success"):
                    raise BrowseError(f"{event.get('command')} failed: {self._scrub(str(event.get('error')))[:200]}")
                return event

    def _abort(self, child: _Child) -> None:
        """Abort the running task and wait for `agent_settled`; end the child when it does not come."""
        settle_by = time.monotonic() + ABORT_WAIT
        try:
            child.send("abort")
            while time.monotonic() < settle_by:
                try:
                    event = child.events.get(timeout=POLL_SECONDS)
                except queue.Empty:
                    continue
                kind = event.get("type")
                if kind == "agent_settled":
                    return
                if event is _EOF:
                    child.dead = True
                    return
                if kind == "extension_ui_request" and event.get("method") in _DIALOGS:
                    child.write({"type": "extension_ui_response", "id": event.get("id"), "cancelled": True})
        except BrowseError:
            return  # the pipe is gone: the child is
        log.warning("browse: no agent_settled %g seconds after abort; ending the browser agent", ABORT_WAIT)
        child.end()

    # -- a task

    def run(self, goal: str, *, stop: Callable[[], bool], on_progress: Callable[[str, dict], None],
            on_ask: Callable[[str, str, Callable[[], bool]], bool], deadline_s: float) -> str:
        """Do `goal` in the browser and return the answer text.

        `on_progress(tool_name, args)` is called as each tool starts (an exception from it is logged, nothing
        more). `on_ask(title, message, stop)` answers a gate question and may block: it runs on the task's own thread
        and nothing else moves meanwhile, so IT owns honoring `stop`, which it is given and which also turns true when
        the question has waited `ask_timeout_s`. A False (or an exception) is a no, and aborts the task.

        Raises `BrowseStopped` when `stop()` ends it, `BrowseDeclined` after a no, and a `BrowseError` otherwise
        (also when `deadline_s` passes: the task is aborted first)."""
        while not self._run_lock.acquire(timeout=POLL_SECONDS):
            if stop():
                raise BrowseStopped
        try:
            started = time.monotonic()
            budget = _Budget(stop, started + deadline_s, deadline_s)
            budget.check()
            child = self._ensure_child(budget)
            return self._task(child, goal, budget, on_progress, on_ask, started)
        finally:
            self._run_lock.release()

    def _task(self, child: _Child, goal: str, budget: _Budget, on_progress: Callable[[str, dict], None],
              on_ask: Callable[[str, str, Callable[[], bool]], bool], started: float) -> str:
        while True:  # whatever an earlier task left behind (late events, unread responses)
            try:
                if child.events.get_nowait() is _EOF:
                    child.dead = True
                    raise BrowseError("the browser agent exited", FAILED)
            except queue.Empty:
                break
        # the prompt goes out only once this is answered: see the module docstring
        reply = self._await_response(child, child.send("new_session"), NEW_SESSION_TIMEOUT, budget)
        if (reply.get("data") or {}).get("cancelled"):
            raise BrowseError("new_session was cancelled")
        prompt_id = child.send("prompt", message=PROMPT.format(goal=goal))

        final, last_stop, last_error, tools = "", None, "", 0
        while True:
            try:
                budget.check()
            except (BrowseStopped, BrowseError):
                self._abort(child)
                raise
            try:
                event = child.events.get(timeout=POLL_SECONDS)
            except queue.Empty:
                continue
            kind = event.get("type")
            if kind == "agent_settled":
                break
            if event is _EOF:
                child.dead = True
                raise BrowseError("the browser agent exited during the task", FAILED)
            if kind == "message_end":
                message = event.get("message") or {}
                if message.get("role") == "assistant":
                    last_stop, last_error = message.get("stopReason"), str(message.get("errorMessage") or "")
                    text = _text_of(message)
                    if text:
                        final = text
            elif kind == "tool_execution_start":
                tools += 1
                name = str(event.get("toolName"))
                log.info("browse: tool %s (%.1fs in)", name, time.monotonic() - started)
                self._progress(on_progress, name, event.get("args"))
            elif kind == "extension_ui_request":
                if self._ui_request(child, event, budget, on_ask):
                    self._abort(child)
                    raise BrowseDeclined
            elif kind == "extension_error":  # the gate throwing blocks the call, but anything else is not safe to trust
                log.warning("browse: extension error in %s (%s)", event.get("extensionPath"), event.get("event"))
                self._abort(child)
                raise BrowseError("an extension of the browser agent failed")
            elif kind == "response" and event.get("id") == prompt_id:
                if not event.get("success"):
                    raise BrowseError(f"prompt failed: {self._scrub(str(event.get('error')))[:200]}")
                if (event.get("data") or {}).get("disposition") == "handled":  # no run started: nothing will settle
                    raise BrowseError("the prompt was not run", FAILED)
            elif kind == "_unparsed":
                log.warning("browse: an unreadable record from the browser agent (%s bytes)", event.get("size"))

        log.info("browse: done in %.1fs, %d tool calls", time.monotonic() - started, tools)
        if last_stop == "error":
            raise BrowseError(f"the model failed: {self._scrub(last_error)[:200]}", FAILED)
        if not final:
            raise BrowseError("the browser agent finished without text", NO_ANSWER)
        return final

    @staticmethod
    def _progress(on_progress: Callable[[str, dict], None], name: str, args) -> None:
        try:
            on_progress(name, args if isinstance(args, dict) else {})
        except Exception:  # a callback must not kill the task
            log.warning("browse: the progress callback failed for %s", name, exc_info=True)

    def _ui_request(self, child: _Child, event: dict, budget: _Budget,
                    on_ask: Callable[[str, str, Callable[[], bool]], bool]) -> bool:
        """Answer one dialog. Returns True when it was a question answered no (the caller aborts the task). A stop
        or the task's deadline that ends the question is not a no: the loop raises what ended it (after `close()`,
        answering the dead child raises the `BrowseError`)."""
        method, rid = event.get("method"), event.get("id")
        if method == "confirm":
            asked_at = time.monotonic()

            def ended() -> bool:  # also when the child exited or close() ended it: nobody is left to ask for
                return budget.stop() or time.monotonic() >= budget.deadline or child.dead or self._closed

            def ask_stop() -> bool:
                return ended() or time.monotonic() - asked_at >= self._ask_timeout

            try:
                approved = bool(on_ask(str(event.get("title") or ""), str(event.get("message") or ""), ask_stop))
            except Exception:  # a failed question is a no
                log.warning("browse: the question callback failed", exc_info=True)
                approved = False
            if ask_stop():  # an expired question, a stopped task or one out of time authorizes nothing
                approved = False
            child.write({"type": "extension_ui_response", "id": rid, "confirmed": approved})
            # a stop or deadline that cut the question short is handled by the loop, not as a refusal
            return not approved and not ended()
        if method in _DIALOGS:  # nothing in vaice answers these: cancelled
            child.write({"type": "extension_ui_response", "id": rid, "cancelled": True})
        return False  # notify, setStatus, setWidget, setTitle, set_editor_text: nothing to answer

    # -- closing

    def close(self) -> None:
        """End the child: close its stdin (an orderly shutdown, which stops its MCP servers), then SIGTERM and
        SIGKILL its group. Idempotent; never waits for a running task, which then fails with a `BrowseError`."""
        atexit.unregister(self.close)
        with self._state_lock:
            self._closed = True
            child, self._child = self._child, None
        if child is None:
            return
        child.dead = True
        with contextlib.suppress(OSError, ValueError):
            if child.proc.stdin is not None:
                child.proc.stdin.close()
        try:
            child.proc.wait(timeout=CLOSE_WAIT)
        except subprocess.TimeoutExpired:
            log.warning("browse: the browser agent did not exit within %g seconds of its stdin closing", CLOSE_WAIT)
        child.end()  # whatever is left of its group
        if self._chrome_profile is not None:
            end_profile_processes(self._chrome_profile)


def _text_of(message: dict) -> str:
    """The text blocks of an assistant message (thinking and tool-call blocks left out), stripped."""
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ""
    return "".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text").strip()

