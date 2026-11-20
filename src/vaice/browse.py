"""The browser agent's shared pieces: the errors and the `BrowseAgent` protocol the rest of vaice depends on, the
settings of Pi's config directory, the two presets that build an `AcpAgent` (`acp_agent.py`, a generic ACP client), and
what is left to clean up when a browser agent was killed. The `browse` skill and its settings live in
`browse_task.py`, `assistant.py` and `app.py`.

How it fits together:

- **vaice speaks ACP** (Agent Client Protocol) to the browser agent: the official Python SDK in `acp_agent.py`, one
  long-lived agent process, one fresh session per task. Pi is one agent, reached through `pi-acp` (pinned, run with
  `npx`), and `VAICE_BROWSE=acp` takes any other ACP agent (`VAICE_BROWSE_AGENT`).
- **Pi preset** (`VAICE_BROWSE=pi`): isolated from the user's Pi. `write_pi_home` writes a config directory of its own
  (`models.json`, `settings.json`, `mcp.json`; the child gets it as `PI_CODING_AGENT_DIR`, with `HOME` inside it) and
  `write_launcher` a script that starts `pi` with the flags that keep the user's skills, extensions and context files
  out (`pi-acp` runs the executable named by `PI_ACP_PI_COMMAND`). The child sees an allowlisted environment, not
  vaice's (the other API keys stay out). The model's API key reaches Pi through that environment (`"$NAME"`
  interpolation in `models.json`), never a file.
- **The gate** (`browse_gate.ts`, a Pi extension) classifies each tool call by `GATE_POLICY`: the read-only browser
  tools run, the ones that change a page ask, anything else asks. An ask is a `confirm` dialog, which `pi-acp` turns
  into a `session/request_permission` that goes to `on_ask`. A gate that loads without hooking in (an edited or
  broken file) leaves the tools ungated (one that throws while loading stops Pi 1.5.2 from starting), so two checks
  keep an ungated Pi from running tasks: `check_gate` (before anything is offered) asks a real Pi for its commands
  and wants `vaice-gate` among them, and the tripwire in `AcpAgent` (every task) wants the gate's
  `notify("vaice-gate <nonce>")` before the first tool call.
- **Any other ACP agent** gets the pinned chrome-devtools server in `session/new` and every permission request goes to
  `on_ask`. vaice can only gate what the agent asks about: its own tools and its permission mode are set in its own
  config.

Pi stops its MCP servers when it shuts down, which is why `AcpAgent.close()` closes stdin and waits before it
signals. A killed Pi can leave its MCP server and the browser it started running: they are NOT in the agent's process
group (measured against Pi 1.5.2: `npm exec` and chrome-devtools-mcp each lead a group of their own, Chrome another),
so `killpg` does not reach them. With a `chrome_profile`, the processes that hold that profile (`--user-data-dir=` for
Chrome, `--userDataDir=` for chrome-devtools-mcp and the `npm exec` that started it) are ended before every new agent
process is started and after `close()`: only those, matched on the exact profile path, never another Chrome.

Failures are `BrowseError`s with a `spoken` sentence; `BrowseStopped` is the caller's own stop. No message, log line
or error holds an API key or the page's content: tool names and timings only.
"""
import contextlib
import json
import logging
import os
import re
import secrets
import shlex
import signal
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from vaice.conversation import write_json_atomically

if TYPE_CHECKING:
    from vaice.acp_agent import AcpAgent

log = logging.getLogger(__name__)

POLL_SECONDS = 0.05  # how often a running task is checked against `stop()` and the deadline
START_TIMEOUT = 30.0  # seconds a fresh agent gets to answer `initialize` (or the gate check to answer, see below)
NEW_SESSION_TIMEOUT = 10.0  # seconds `session/new` may take before the agent counts as wedged
ABORT_WAIT = 10.0  # seconds after `session/cancel` for the prompt to end, normally about 10 ms; then the agent is ended
CLOSE_WAIT = 5.0  # seconds a child gets to exit after its stdin is closed
TERM_WAIT = 3.0  # seconds between SIGTERM and SIGKILL
KILL_WAIT = 2.0  # seconds to reap a SIGKILLed child
ASK_GRACE = 5.0  # seconds the gate waits longer than `ask_timeout_s`, so vaice's own "no" (and abort) comes first
PS_TIMEOUT = 10.0  # seconds `ps` may take when looking for what still holds the browser profile
MIN_PI_VERSION = (1, 5, 2)  # the Pi this module was measured against (its RPC mode and extension API)
MIN_SECRET_CHARS = 8  # an environment value this short is not scrubbed from messages as a secret (see `secret_values`)

CHROME_DEVTOOLS_MCP = "chrome-devtools-mcp@1.13.0"  # pinned: its flags and tool names are what the policy names
PI_ACP = "pi-acp@0.0.34"  # pinned like chrome-devtools-mcp: its mapping of Pi's tools and dialogs is what vaice reads
API_KEY_ENV = "VAICE_PI_API_KEY"  # the usual name of the child's environment variable holding the model's key
GATE_FILE = Path(__file__).with_name("browse_gate.ts")
GATE_COMMAND = "vaice-gate"  # the command browse_gate.ts registers (and the start of its notify marker)
LAUNCHER = "launch-pi"  # the script `write_launcher` writes into Pi's config directory
PROVIDER = "vaice"  # the provider id in models.json

_MUTATING_TOOLS = ("click", "fill", "fill_form", "type_text", "press_key", "evaluate_script", "upload_file", "drag",
                   "handle_dialog")
GATE_POLICY = {
    "mcp__chrome__*": "run",
    **{f"mcp__chrome__{name}": "ask" for name in _MUTATING_TOOLS},
    "default": "ask",
}  # tools Pi knows by exact name win over the `*` pattern; anything not named asks
# VAICE_BROWSE_CONFIRM=off: every browser tool runs unasked (clicks, typing, forms); anything else still asks
UNGATED_POLICY = {"mcp__chrome__*": "run", "default": "ask"}

# what the child may inherit: enough to find node and reach the network, nothing else (not vaice's other keys)
_INHERITED_ENV = ("PATH", "TMPDIR", "LANG", "LC_ALL", "LC_CTYPE", "TERM", "USER", "LOGNAME", "SHELL", "SSL_CERT_FILE",
                  "SSL_CERT_DIR", "NODE_EXTRA_CA_CERTS", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY",
                  "http_proxy", "https_proxy", "all_proxy", "no_proxy", "npm_config_cache", "npm_config_registry")
_PI_FLAGS = ("--no-session", "--no-builtin-tools", "--no-context-files", "--no-skills", "--no-prompt-templates",
             "--offline")  # not --no-extensions: that would turn off Pi's built-in MCP client too
_SECRET_NAME = re.compile(r"KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL", re.IGNORECASE)
_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")

PROMPT = ("Use the chrome browser tools to do this task in the browser, then answer in plain text. To search the "
          "web use Bing or DuckDuckGo, never Google: Google shows an automated browser a bot check instead of "
          "results. Task: {goal}")
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
    """A browser agent: does a goal in the browser and returns the answer text. `AcpAgent` is the one backend."""

    def run(self, goal: str, *, stop: Callable[[], bool], on_progress: Callable[[str, dict], None],
            on_ask: Callable[[str, str, Callable[[], bool]], bool], deadline_s: float,
            on_text: Callable[[str], None] | None = None) -> str:
        """Do `goal`. `on_progress(tool, args)` is told of each tool as it starts; `on_ask(title, message, stop)`
        answers a question about a step that changes a page (True = go ahead) and may block, honoring its `stop`;
        `on_text(text)`, optional, is told of the agent's own words as each message ends (what `browser_status` can
        report; a backend without them never calls it).
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
    `mcp.json` (the pinned chrome-devtools server, `chrome_args(browser_args)`). `api_key_env` is the NAME of
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
    # quietStartup: no banner or "new version" text in the answer's message chunks
    write_json_atomically(path / "settings.json", {
        "defaultProvider": PROVIDER, "defaultModel": model, "defaultProjectTrust": "never", "quietStartup": True})
    write_json_atomically(path / "mcp.json", {"mcpServers": {"chrome": {
        "command": "npx", "args": chrome_args(browser_args), "exposure": "direct",
    }}})


def chrome_args(browser_args: Sequence[str]) -> list[str]:
    """The arguments of `npx` for the pinned chrome-devtools server, `browser_args` after the package: no usage
    statistics to Google, and no trace URLs to the CrUX API (no trace tool is wanted anyway)."""
    return ["-y", CHROME_DEVTOOLS_MCP, "--no-usage-statistics", "--no-performance-crux", *browser_args]


def launcher_path(pi_home: Path) -> Path:
    return pi_home / LAUNCHER


def write_launcher(path: Path) -> None:
    """Write the executable (0700) `pi-acp` runs in place of `pi` (`PI_ACP_PI_COMMAND`): `pi` with the flags that
    keep the user's skills, prompt templates and context files out, no session files, and the gate extension. It
    adds `"$@"`, which `pi-acp` fills with `--mode rpc --no-themes`. Raises OSError."""
    flags = " ".join(shlex.quote(flag) for flag in (*_PI_FLAGS, "-e", str(GATE_FILE)))
    tmp = path.with_name(f"{path.name}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o700)
    with os.fdopen(fd, "w") as handle:
        handle.write(f'#!/bin/sh\nexec pi {flags} "$@"\n')
    tmp.chmod(0o700)
    tmp.replace(path)


# --- the processes of an agent


def end_process_group(proc: subprocess.Popen) -> None:
    """End `proc` and what it left in its process group: SIGTERM to the group, then, after it exits or `TERM_WAIT`
    passes, SIGKILL to the group anyway (a member that ignores SIGTERM must not outlive it), then reap it. Bounded."""
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(proc.pid, signal.SIGTERM)
    with contextlib.suppress(subprocess.TimeoutExpired):
        proc.wait(timeout=TERM_WAIT)
    with contextlib.suppress(ProcessLookupError, PermissionError):  # the rest of the group, the leader exited or not
        os.killpg(proc.pid, signal.SIGKILL)
    try:
        proc.wait(timeout=KILL_WAIT)
    except subprocess.TimeoutExpired:
        log.warning("browse: the browser agent (pid %d) did not exit after SIGKILL", proc.pid)


def secret_values(env: Mapping[str, str]) -> list[str]:
    """The values in `env` that look like credentials (a name with KEY, TOKEN, SECRET, PASSWORD or CREDENTIAL, a value
    of at least `MIN_SECRET_CHARS`): what an agent's messages are scrubbed of."""
    return [value for name, value in env.items() if _SECRET_NAME.search(name) and len(value) >= MIN_SECRET_CHARS]


# --- the Pi preset


def pi_environment(pi_home: Path, *, ask_timeout_s: float, gate_policy: Mapping[str, str], nonce: str,
                   api_key: str | None = None) -> dict[str, str]:
    """The environment of the `pi-acp` process and the `pi` it starts: an allowlist of vaice's (not its other keys),
    Pi's config directory and `HOME` inside it, the gate's settings (policy, how long a question may wait, the nonce
    of its marker), the launcher, and the model's key under `API_KEY_ENV` when there is one."""
    env = {name: os.environ[name] for name in _INHERITED_ENV if name in os.environ}
    env.setdefault("npm_config_cache", str(Path.home() / ".npm"))  # HOME moves, the package cache must not
    env.update(PI_CODING_AGENT_DIR=str(pi_home), HOME=str(home_dir(pi_home)), PI_OFFLINE="1",
               PI_SKIP_VERSION_CHECK="1", PI_TELEMETRY="0", VAICE_GATE_POLICY=json.dumps(dict(gate_policy)),
               VAICE_GATE_TIMEOUT_MS=str(int((ask_timeout_s + ASK_GRACE) * 1000)), VAICE_GATE_NONCE=nonce,
               PI_ACP_PI_COMMAND=str(launcher_path(pi_home)))
    if api_key:
        env[API_KEY_ENV] = api_key
    return env


def new_nonce() -> str:
    """A fresh random string for the gate's marker: a page cannot make the agent say what it cannot guess."""
    return secrets.token_hex(8)


def gate_marker(nonce: str) -> str:
    """The text of the notification the gate sends at the start of every prompt."""
    return f"{GATE_COMMAND} {nonce}"


def check_gate(pi_home: Path, env: Mapping[str, str], *, log_path: Path | None = None) -> bool:
    """Whether a real Pi, started the way `pi-acp` starts it, has the gate extension loaded: runs the launcher in
    RPC mode, asks for its commands, closes its stdin (Pi answers and exits) and looks for `vaice-gate`. A gate that
    loads without hooking in would leave the tools ungated. False also when Pi cannot
    be run or does not answer in `START_TIMEOUT`. Its stderr goes to `log_path` (0600)."""
    work = work_dir(pi_home)
    work.mkdir(mode=0o700, parents=True, exist_ok=True)
    err_fd = os.open(log_path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600) if log_path else subprocess.DEVNULL
    try:
        proc = subprocess.Popen([str(launcher_path(pi_home)), "--mode", "rpc"], stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE, stderr=err_fd, cwd=work, env=dict(env),
                                start_new_session=True)
    except OSError as exc:
        log.warning("browse: could not run pi to check the tool gate: %s", exc.strerror or type(exc).__name__)
        return False
    finally:
        if log_path:
            os.close(err_fd)
    request = json.dumps({"id": "g1", "type": "get_commands"}) + "\n"
    try:
        out, _ = proc.communicate(request.encode(), timeout=START_TIMEOUT)
    except subprocess.TimeoutExpired:
        log.warning("browse: pi did not answer the tool gate check in %g seconds", START_TIMEOUT)
        end_process_group(proc)
        out = b""
    for line in out.splitlines():
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if isinstance(record, dict) and record.get("type") == "response" and record.get("id") == "g1":
            commands = (record.get("data") or {}).get("commands", [])
            return any(isinstance(c, dict) and c.get("name") == GATE_COMMAND for c in commands)
    return False


def pi_agent(pi_home: Path, *, env: Mapping[str, str], nonce: str, log_path: Path, ask_timeout_s: float,
             chrome_profile: Path | None = None) -> "AcpAgent":
    """The agent for `VAICE_BROWSE=pi`: `pi-acp` (pinned, run with `npx`) over `env` (`pi_environment`), working in
    Pi's work directory. Pi's own `mcp.json` holds the browser server (`pi-acp` ignores the ones in `session/new`);
    its gate's questions all come as permission requests; the gate's marker (`nonce`) is the tripwire."""
    from vaice.acp_agent import AcpAgent

    return AcpAgent(["npx", "-y", PI_ACP], env=env, cwd=work_dir(pi_home), log_path=log_path, permissions="ask",
                    ask_timeout_s=ask_timeout_s, marker=gate_marker(nonce), chrome_profile=chrome_profile,
                    secrets=[env[API_KEY_ENV]] if env.get(API_KEY_ENV) else [], pi_dialogs=True)


# --- the generic ACP preset


def acp_environment(environ: Mapping[str, str]) -> dict[str, str]:
    """What an agent of `VAICE_BROWSE=acp` inherits: `environ` without a `VAICE_*` variable (vaice's own keys and
    settings). The agent needs the rest (its HOME, its provider keys), so this is not the Pi allowlist."""
    return {name: value for name, value in environ.items() if not name.startswith("VAICE_")}


def acp_agent(argv: Sequence[str], *, work: Path, log_path: Path, browser_args: Sequence[str], ask_timeout_s: float,
              chrome_profile: Path | None = None, confirm: bool = True,
              environ: Mapping[str, str] | None = None) -> "AcpAgent":
    """The agent for `VAICE_BROWSE=acp`: `argv` (the user's command line) in the empty directory `work` (made, 0700),
    with the pinned chrome-devtools server in `session/new`. Every permission request goes to `on_ask`, unless
    `confirm` is false (`VAICE_BROWSE_CONFIRM=off`: each is allowed once). No tripwire: nothing of ours runs in
    the agent. Raises OSError (the directory)."""
    from acp.schema import McpServerStdio

    from vaice.acp_agent import AcpAgent

    work.mkdir(mode=0o700, parents=True, exist_ok=True)
    env = acp_environment(os.environ if environ is None else environ)
    chrome = McpServerStdio(name="chrome", command="npx", args=chrome_args(browser_args), env=[])
    return AcpAgent(argv, env=env, cwd=work, log_path=log_path, mcp_servers=[chrome],
                    permissions="ask" if confirm else "allow", ask_timeout_s=ask_timeout_s,
                    chrome_profile=chrome_profile, secrets=secret_values(env))
