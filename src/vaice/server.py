"""`vaice up`: reuse the llama-server that is already running, or start one and stop it again at exit."""
import atexit
import logging
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from urllib.parse import urlsplit

import requests

from vaice import config, lifecycle

log = logging.getLogger(__name__)

BINARY = "llama-server"
MIN_BUILD = 11960  # the first llama.cpp build with the /v1/realtime WebSocket (docs/installation.md, "Requirements")
INSTALL_HINT = "brew install llama.cpp (other platforms: https://github.com/ggml-org/llama.cpp#usage)"
LOCAL_HOSTS = ("localhost", "127.0.0.1", "::1")
HEALTH_TIMEOUT = 2  # seconds for one /health probe
VERSION_TIMEOUT = 10  # seconds for `llama-server --version`
POLL_SECONDS = 1.0  # how often the start-up wait asks /health
PROGRESS_SECONDS = 10  # how often the wait says it is still going
STOP_WAIT = 5  # seconds the child gets to exit after terminate() before it is killed
LOG_TAIL_LINES = 10  # lines of the server's log shown when it dies early


def _health(url: str) -> int | None:
    """The HTTP status of the server's /health, or None when nothing answers."""
    try:
        return requests.get(f"{url}/health", timeout=HEALTH_TIMEOUT).status_code
    except requests.RequestException as exc:
        log.debug("no answer from %s/health: %s", url, exc)
        return None


def _host_port(url: str) -> tuple[str, int]:
    parts = urlsplit(url)
    return parts.hostname or "", parts.port or (443 if parts.scheme == "https" else 80)


def _binary() -> str:
    path = shutil.which(BINARY)
    if path is None:
        raise SystemExit(
            f"vaice: `{BINARY}` was not found on PATH; vaice up does not install it. "
            f"Install llama.cpp: {INSTALL_HINT}"
        )
    return path


def _build_number(binary: str) -> int | None:
    """The llama.cpp build number from `llama-server --version`, or None when it cannot be read."""
    try:
        result = subprocess.run([binary, "--version"], capture_output=True, text=True, timeout=VERSION_TIMEOUT,
                                check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        log.warning("could not run %s --version (%s); not checking its build", binary, exc)
        return None
    match = re.search(r"\bbuild\s+(\d+)", f"{result.stdout}\n{result.stderr}")  # "version: 1.0.0 (build 12359, ...)"
    if match is None:
        log.warning("could not read the build number from %s --version; not checking it", binary)
        return None
    return int(match.group(1))


def _check_build(binary: str) -> None:
    build = _build_number(binary)
    if build is not None and build < MIN_BUILD:
        raise SystemExit(
            f"vaice: {binary} is build {build}, but vaice needs build {MIN_BUILD} or newer for the /v1/realtime "
            "WebSocket. Upgrade it: brew upgrade llama.cpp"
        )


def _log_path() -> Path:
    if config.DATA_DIR is None:  # VAICE_DATA_DIR="": nothing under the data dir, so the log goes to a temp dir
        return Path(tempfile.mkdtemp(prefix="vaice-")) / "llama-server.log"
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    return config.DATA_DIR / "llama-server.log"


def _log_tail(path: Path) -> str:
    try:
        lines = path.read_text(errors="replace").splitlines()
    except OSError as exc:
        return f"(could not read {path}: {exc})"
    return "\n".join(lines[-LOG_TAIL_LINES:])


class ManagedServer:
    """A llama-server child that vaice started. `stop()` can be called any number of times."""

    def __init__(self, proc: subprocess.Popen, log_path: Path):
        self.proc = proc
        self.log_path = log_path

    def stop(self) -> None:
        atexit.unregister(self.stop)
        if self.proc.poll() is not None:
            return
        log.info("stopping llama-server")
        self.proc.terminate()
        try:
            self.proc.wait(timeout=STOP_WAIT)
        except subprocess.TimeoutExpired:
            log.warning("llama-server did not exit within %ds; killing it", STOP_WAIT)
            self.proc.kill()
            self.proc.wait()


def _start(command: list[str], log_path: Path) -> ManagedServer:
    log.info("starting llama-server: %s", " ".join(command))
    log.info("llama-server output goes to %s", log_path)
    try:
        with open(log_path, "ab") as out:
            out.write(f"\n--- vaice up: {' '.join(command)}\n".encode())
            out.flush()
            # Its own session: the terminal's Ctrl+C reaches vaice only, which stops the server itself.
            proc = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT,
                                    start_new_session=True)
    except OSError as exc:
        raise SystemExit(f"vaice: could not start {command[0]} ({exc}); log: {log_path}") from exc
    server = ManagedServer(proc, log_path)
    atexit.register(server.stop)  # backstop; main()'s finally stops it first
    return server


def _wait_until_ready(server: ManagedServer, url: str) -> None:
    started = time.monotonic()
    next_progress = PROGRESS_SECONDS
    while True:
        if lifecycle.stopping.is_set():
            raise SystemExit("vaice: stopped while waiting for llama-server")
        code = server.proc.poll()
        if code is not None:
            raise SystemExit(
                f"vaice: llama-server exited with code {code} before it was ready. Last lines of "
                f"{server.log_path}:\n{_log_tail(server.log_path)}"
            )
        if _health(url) == 200:
            return
        elapsed = time.monotonic() - started
        if elapsed >= config.LLAMACPP_START_TIMEOUT:
            raise SystemExit(
                f"vaice: llama-server was not ready after {config.LLAMACPP_START_TIMEOUT:g}s; see {server.log_path} "
                "(raise VAICE_LLAMACPP_START_TIMEOUT if the model is still downloading)"
            )
        if elapsed >= next_progress:
            log.info("waiting for llama-server (downloading/loading the model, %ds)… see %s",
                     elapsed, server.log_path)
            next_progress += PROGRESS_SECONDS
        lifecycle.stopping.wait(POLL_SECONDS)


def ensure_server() -> ManagedServer | None:
    """Make sure a llama-server answers at `config.LLAMACPP_URL`.

    Returns the child it started (the caller stops it), or None when one was already running.
    Exits with an actionable message when none can be started; a child that was started is
    stopped before that exit.
    """
    url = config.LLAMACPP_URL
    status = _health(url)
    if status in (200, 503):  # 503: up, still loading its model
        log.info("using the llama-server already running at %s", url)
        return None

    host, port = _host_port(url)
    if host not in LOCAL_HOSTS:
        raise SystemExit(
            f"vaice: no llama-server answers at {url}, and vaice up only starts a local server "
            f"(host localhost, 127.0.0.1 or ::1). Start it where {host} is, or change VAICE_LLAMACPP_URL"
        )
    binary = _binary()
    _check_build(binary)

    command = [binary, "-hf", config.LLAMACPP_MODEL, "--temp", "0", "--host", host, "--port", str(port),
               *config.LLAMACPP_SERVER_ARGS]
    server = _start(command, _log_path())
    try:
        _wait_until_ready(server, url)
    except BaseException:  # SystemExit from the wait, Ctrl+C: the child must not outlive vaice
        server.stop()
        raise
    log.info("llama-server is ready at %s", url)
    return server
