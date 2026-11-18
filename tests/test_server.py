import logging
import subprocess
from pathlib import Path

import pytest
import requests

from vaice import lifecycle, server

VERSION_LINE = "version: 1.0.0 (build 12359, commit b1b770d90)\n"


class FakeProc:
    """A stand-in for subprocess.Popen: alive until it exits on its own or is terminated/killed."""

    def __init__(self, exit_code=None, ignores_terminate=False):
        self.returncode = exit_code
        self.ignores_terminate = ignores_terminate
        self.calls = []

    def poll(self):
        return self.returncode

    def terminate(self):
        self.calls.append("terminate")
        if not self.ignores_terminate:
            self.returncode = -15

    def kill(self):
        self.calls.append("kill")
        self.returncode = -9

    def wait(self, timeout=None):
        self.calls.append(("wait", timeout))
        if self.returncode is None:
            raise subprocess.TimeoutExpired("llama-server", timeout)
        return self.returncode


class Env:
    """The world `ensure_server` looks at: /health answers, the binary, its version, the child."""

    def __init__(self, monkeypatch, tmp_path):
        self.health = []  # statuses handed out one per probe (None = nothing answers); the last one repeats
        self.probes = 0
        self.popen_calls = []
        self.proc = FakeProc()
        self.version = VERSION_LINE
        self.binary = "/opt/bin/llama-server"
        monkeypatch.setattr(server.config, "LLAMACPP_URL", "http://localhost:8001")
        monkeypatch.setattr(server.config, "LLAMACPP_MODEL", "LiquidAI/LFM3-Omni-2B-GGUF:Q8_0")
        monkeypatch.setattr(server.config, "LLAMACPP_SERVER_ARGS", [])
        monkeypatch.setattr(server.config, "LLAMACPP_START_TIMEOUT", 5.0)
        monkeypatch.setattr(server.config, "DATA_DIR", tmp_path / "data")
        monkeypatch.setattr(server, "POLL_SECONDS", 0.001)
        monkeypatch.setattr(server, "_health", self._probe)
        monkeypatch.setattr(server.shutil, "which", lambda name: self.binary if name == "llama-server" else None)
        monkeypatch.setattr(server.subprocess, "run", self._run)
        monkeypatch.setattr(server.subprocess, "Popen", self._popen)

    def _probe(self, url):
        status = self.health[min(self.probes, len(self.health) - 1)] if self.health else None
        self.probes += 1
        return status

    def _run(self, command, **kwargs):
        assert command == [self.binary, "--version"]
        return subprocess.CompletedProcess(command, 0, stdout="", stderr=self.version)  # llama.cpp logs to stderr

    def _popen(self, command, **kwargs):
        self.popen_calls.append((command, kwargs))
        return self.proc


@pytest.fixture
def env(monkeypatch, tmp_path):
    return Env(monkeypatch, tmp_path)


@pytest.fixture
def exit_hooks(monkeypatch):
    """The functions registered with atexit, instead of really registering them."""
    registered = []
    monkeypatch.setattr(server.atexit, "register", registered.append)

    def unregister(func):
        if func in registered:
            registered.remove(func)

    monkeypatch.setattr(server.atexit, "unregister", unregister)
    return registered


@pytest.mark.parametrize("status", [200, 503])
def test_a_server_that_answers_is_reused_and_nothing_is_started(env, status, caplog):
    env.health = [status]
    with caplog.at_level(logging.INFO, logger="vaice.server"):
        assert server.ensure_server() is None

    assert "using the llama-server already running at http://localhost:8001" in caplog.text
    assert env.popen_calls == []


def test_health_is_the_status_code_or_none_when_nothing_answers(monkeypatch):
    class Response:
        status_code = 503

    seen = []
    monkeypatch.setattr(server.requests, "get", lambda url, timeout: seen.append((url, timeout)) or Response())
    assert server._health("http://localhost:8001") == 503
    assert seen == [("http://localhost:8001/health", 2)]

    def refuse(url, timeout):
        raise requests.ConnectionError("refused")

    monkeypatch.setattr(server.requests, "get", refuse)
    assert server._health("http://localhost:8001") is None


def test_a_remote_server_that_is_down_is_not_replaced_by_a_local_one(env, monkeypatch):
    monkeypatch.setattr(server.config, "LLAMACPP_URL", "http://gpu-box:8001")
    with pytest.raises(SystemExit, match=r"only starts a local server.*gpu-box"):
        server.ensure_server()

    assert env.popen_calls == []


def test_a_missing_binary_says_how_to_install_it_and_installs_nothing(env, monkeypatch):
    monkeypatch.setattr(server.shutil, "which", lambda name: None)
    with pytest.raises(SystemExit) as exc:
        server.ensure_server()

    assert "brew install llama.cpp" in str(exc.value)
    assert "github.com/ggml-org/llama.cpp" in str(exc.value)
    assert env.popen_calls == []


def test_the_build_number_is_read_from_the_version_line(env):
    assert server._build_number(env.binary) == 12359


def test_the_build_number_is_also_read_from_stdout(env, monkeypatch):
    monkeypatch.setattr(server.subprocess, "run", lambda command, **kw: subprocess.CompletedProcess(
        command, 0, stdout=VERSION_LINE, stderr=""))
    assert server._build_number(env.binary) == 12359


def test_a_build_older_than_the_realtime_endpoint_is_refused_with_both_numbers(env):
    env.version = "version: 0.8.0 (build 11959, commit abc)\n"
    with pytest.raises(SystemExit) as exc:
        server.ensure_server()

    message = str(exc.value)
    assert "11959" in message and "11960" in message and "brew upgrade llama.cpp" in message
    assert env.popen_calls == []


def test_the_minimum_build_itself_is_accepted(env, exit_hooks):
    env.version = "version: 0.8.0 (build 11960, commit abc)\n"
    env.health = [None, 200]
    assert server.ensure_server() is not None


def test_a_version_that_cannot_be_parsed_warns_and_continues(env, exit_hooks, caplog):
    env.version = "something else entirely\n"
    env.health = [None, 200]
    with caplog.at_level(logging.WARNING, logger="vaice.server"):
        assert server.ensure_server() is not None

    assert "could not read the build number" in caplog.text
    assert len(env.popen_calls) == 1


def test_a_version_command_that_cannot_run_warns_and_continues(env, exit_hooks, monkeypatch, caplog):
    def broken(command, **kwargs):
        raise subprocess.TimeoutExpired(command, 10)

    monkeypatch.setattr(server.subprocess, "run", broken)
    env.health = [None, 200]
    with caplog.at_level(logging.WARNING, logger="vaice.server"):
        assert server.ensure_server() is not None

    assert "could not run" in caplog.text
    assert len(env.popen_calls) == 1


def test_the_child_is_started_with_the_model_the_urls_host_and_port_and_the_extra_args(env, exit_hooks, monkeypatch):
    monkeypatch.setattr(server.config, "LLAMACPP_URL", "http://127.0.0.1:9100")
    monkeypatch.setattr(server.config, "LLAMACPP_MODEL", "me/other-GGUF:Q8_0")
    monkeypatch.setattr(server.config, "LLAMACPP_SERVER_ARGS", ["--tools", "get_info,read_file"])
    env.health = [None, 200]

    managed = server.ensure_server()

    ((command, kwargs),) = env.popen_calls
    assert command == ["/opt/bin/llama-server", "-hf", "me/other-GGUF:Q8_0", "--temp", "0",
                       "--host", "127.0.0.1", "--port", "9100", "--tools", "get_info,read_file"]
    assert kwargs["stdin"] is subprocess.DEVNULL
    assert kwargs["stderr"] is subprocess.STDOUT
    assert kwargs["start_new_session"] is True
    assert managed.proc is env.proc
    assert env.proc.poll() is None  # a ready server keeps running; the caller stops it


@pytest.mark.parametrize(("url", "host", "port"), [
    ("http://localhost", "localhost", 80),
    ("https://localhost", "localhost", 443),
    ("http://[::1]:9000", "::1", 9000),
])
def test_host_and_port_come_from_the_url_with_the_schemes_default_port(url, host, port):
    assert server._host_port(url) == (host, port)


def test_an_ipv6_loopback_url_is_local(env, exit_hooks, monkeypatch):
    monkeypatch.setattr(server.config, "LLAMACPP_URL", "http://[::1]:8001")
    env.health = [None, 200]
    server.ensure_server()

    command = env.popen_calls[0][0]
    assert command[command.index("--host") + 1] == "::1"


def test_the_servers_output_is_appended_to_a_log_in_the_data_dir(env, exit_hooks, tmp_path, caplog):
    env.health = [None, 200]
    log_file = tmp_path / "data" / "llama-server.log"
    log_file.parent.mkdir()
    log_file.write_text("earlier run\n")
    with caplog.at_level(logging.INFO, logger="vaice.server"):
        managed = server.ensure_server()

    assert managed.log_path == log_file
    text = log_file.read_text()
    assert text.startswith("earlier run\n")
    assert "--- vaice up: /opt/bin/llama-server -hf" in text
    assert "starting llama-server: /opt/bin/llama-server -hf" in caplog.text
    assert str(log_file) in caplog.text


def test_without_a_data_dir_the_log_goes_to_a_temp_directory(env, exit_hooks, monkeypatch, tmp_path):
    monkeypatch.setattr(server.config, "DATA_DIR", None)
    monkeypatch.setattr(server.tempfile, "mkdtemp", lambda prefix: str(tmp_path / "scratch"))
    (tmp_path / "scratch").mkdir()
    env.health = [None, 200]

    managed = server.ensure_server()

    assert managed.log_path == tmp_path / "scratch" / "llama-server.log"
    assert managed.log_path.exists()


def test_a_child_that_cannot_be_started_is_an_actionable_exit(env, monkeypatch):
    def fail(command, **kwargs):
        raise PermissionError("not executable")

    monkeypatch.setattr(server.subprocess, "Popen", fail)
    with pytest.raises(SystemExit, match=r"could not start /opt/bin/llama-server .*not executable"):
        server.ensure_server()


def test_a_child_that_exits_early_reports_its_exit_code_and_the_end_of_its_log(env, exit_hooks, tmp_path):
    env.proc = FakeProc(exit_code=1)
    env.health = [None]
    log_file = tmp_path / "data" / "llama-server.log"
    log_file.parent.mkdir()
    log_file.write_text("".join(f"line {n}\n" for n in range(30)))

    with pytest.raises(SystemExit) as exc:
        server.ensure_server()

    message = str(exc.value)
    assert "exited with code 1" in message
    # the last 10 lines of the log: line 22..29, the blank separator and the "vaice up" header of this start
    assert "line 29" in message and "line 22" in message and "--- vaice up:" in message
    assert "line 21" not in message
    assert str(log_file) in message


def test_the_log_tail_survives_an_unreadable_log(tmp_path):
    assert "could not read" in server._log_tail(tmp_path / "missing.log")


def test_a_server_that_is_not_ready_in_time_is_stopped_and_reported(env, exit_hooks, monkeypatch):
    monkeypatch.setattr(server.config, "LLAMACPP_START_TIMEOUT", 0.05)
    env.health = [None, 503]  # loading forever

    with pytest.raises(SystemExit, match=r"not ready after 0\.05s.*VAICE_LLAMACPP_START_TIMEOUT"):
        server.ensure_server()

    assert env.proc.calls[0] == "terminate"
    assert exit_hooks == []  # stop() ran, so nothing is left for atexit


def test_the_wait_says_it_is_still_going(env, exit_hooks, monkeypatch, caplog):
    monkeypatch.setattr(server, "PROGRESS_SECONDS", 0.005)
    env.health = [None] * 20 + [200]
    with caplog.at_level(logging.INFO, logger="vaice.server"):
        assert server.ensure_server() is not None

    assert "waiting for llama-server (downloading/loading the model," in caplog.text
    assert "llama-server.log" in caplog.text


def test_the_stop_signal_ends_the_wait_promptly_and_stops_the_child(env, exit_hooks, monkeypatch):
    probes = []

    def probe(url):
        probes.append(url)
        if len(probes) == 3:
            lifecycle.stopping.set()  # Ctrl+C while the model loads
        return

    monkeypatch.setattr(server, "_health", probe)
    with pytest.raises(SystemExit, match="stopped while waiting"):
        server.ensure_server()

    assert len(probes) == 3  # the reuse check, then two waits; the flag is seen before a fourth
    assert env.proc.calls[0] == "terminate"


def test_ctrl_c_during_the_wait_stops_the_child_and_propagates(env, exit_hooks, monkeypatch):
    probes = []

    def probe(url):
        probes.append(url)
        if len(probes) == 2:
            raise KeyboardInterrupt
        return

    monkeypatch.setattr(server, "_health", probe)
    with pytest.raises(KeyboardInterrupt):
        server.ensure_server()

    assert env.proc.calls[0] == "terminate"


def test_stop_terminates_and_waits():
    proc = FakeProc()
    server.ManagedServer(proc, Path("x.log")).stop()

    assert proc.calls == ["terminate", ("wait", server.STOP_WAIT)]


def test_a_child_that_ignores_terminate_is_killed_after_the_grace_period(caplog):
    proc = FakeProc(ignores_terminate=True)
    with caplog.at_level(logging.WARNING, logger="vaice.server"):
        server.ManagedServer(proc, Path("x.log")).stop()

    assert proc.calls == ["terminate", ("wait", server.STOP_WAIT), "kill", ("wait", None)]
    assert "killing it" in caplog.text


def test_stopping_twice_is_harmless():
    proc = FakeProc()
    managed = server.ManagedServer(proc, Path("x.log"))
    managed.stop()
    managed.stop()

    assert proc.calls.count("terminate") == 1


def test_stopping_a_child_that_already_exited_signals_nothing():
    proc = FakeProc(exit_code=0)
    server.ManagedServer(proc, Path("x.log")).stop()

    assert proc.calls == []


def test_a_started_child_is_registered_for_exit_and_unregistered_by_stop(env, exit_hooks):
    env.health = [None, 200]

    managed = server.ensure_server()

    assert exit_hooks == [managed.stop]
    managed.stop()
    assert exit_hooks == []
