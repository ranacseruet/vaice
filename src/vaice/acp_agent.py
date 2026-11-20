"""`AcpAgent`: a browser agent (`browse.BrowseAgent`) that speaks ACP, the Agent Client Protocol, to any ACP agent over
its stdin/stdout, with the official Python SDK (`acp`, the optional `browse` extra: this module needs it, so it is
imported only where a browser agent is built). Pi is one such agent (`pi-acp`, see `browse.pi_agent`);
`browse.acp_agent` is the generic preset.

How it works:

- **One long-lived agent process** (started on the first task, restarted after it dies): its own session
  (`start_new_session`), stderr appended to a 0600 log file, an `atexit` backstop that `close()` unregisters.
  `browse.end_profile_processes` runs before every new process and after `close()` (the browser an agent started is in
  none of its groups).
- **One asyncio loop on a daemon thread** of its own talks to the agent. `run()` is called on the browse task's thread:
  it hands the task coroutine to the loop and polls `stop()` and the deadline every `POLL_SECONDS`, like `shell.run`.
  One task at a time. The loop never blocks: `on_ask` waits for the Run button, so it runs on a daemon thread of its
  own per question (not the loop's executor, whose threads are joined at exit).
- **Every task is a fresh `session/new`** (the agent's context starts empty), `initialize` once per process, no file
  system or terminal capability offered. `session/close` is sent after a task only when the agent advertises it.
- **The answer is the text after the last tool call**, not every chunk: "I'll open the page" before a tool call is
  chatter (and, for Pi, `pi-acp`'s start-up text), which goes to `on_text` as one message per tool call.
- **A no also cancels.** A refused step is answered with the agent's reject option, then `session/cancel`, then the
  task ends as `BrowseDeclined`: a model told no routes around it (retries, another tool, navigates directly).
  `stop()` and the deadline answer a pending question `cancelled` first, then `session/cancel`, then wait up to
  `ABORT_WAIT` for the prompt to end; an agent that does not stop is ended and replaced by the next task. Each
  response goes out before the cancel that follows it (read from the connection's outgoing messages).
- **The tripwire** (with a `marker`, the Pi preset): the gate sends `notify(marker)` when each prompt starts, which
  arrives as a message chunk. A tool call or permission request before that chunk means the gate is not running: the
  request is answered `cancelled`, the agent is cancelled and ended, and the task fails. Text before the marker
  (start-up chatter) is dropped, and the marker is never part of the answer.

An error from the SDK, the agent or a callback is never raised as anything but `BrowseError` (`BrowseStopped` is the
caller's own stop). No message, log line or error holds the page's content, the goal or an environment value: tool
names and timings only, and what an agent reports is scrubbed of `secrets` first.
"""
import asyncio
import atexit
import concurrent.futures
import contextlib
import json
import logging
import os
import subprocess
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any, Literal

from acp import PROTOCOL_VERSION, RequestError, connect_to_agent
from acp.schema import (
    AgentMessageChunk,
    AllowedOutcome,
    ClientCapabilities,
    DeniedOutcome,
    Implementation,
    McpServerStdio,
    PermissionOption,
    RequestPermissionResponse,
    TextContentBlock,
    ToolCallStart,
    ToolCallUpdate,
)

from vaice import browse
from vaice.browse import BrowseDeclined, BrowseError, BrowseStopped

log = logging.getLogger(__name__)

RAW_INPUT_CHARS = 2000  # of a permission request's arguments a question carries when the agent gave no `message`
LOG_TOOL_CHARS = 60  # of a tool's title a log line carries (titles are free text across agents)
STREAM_LIMIT = 16 * 1024 * 1024  # the longest line an agent may write (a screenshot is one)
SESSION_CLOSE_WAIT = 5.0  # seconds `session/close` may take (the answer is in already: this is housekeeping)
ANSWER_WAIT = 2.0  # seconds a response to a permission request may take to go out before the cancel follows it
CANCELLED = RequestPermissionResponse(outcome=DeniedOutcome(outcome="cancelled"))


class _Interrupted(Exception):
    """Internal: the task coroutine stopped because `state.interrupted` was set (the caller knows why)."""


@dataclass(frozen=True)
class _Budget:
    """What ends a task early: the caller's `stop()` and the deadline (`seconds` after the task began)."""

    stop: Callable[[], bool]
    deadline: float
    seconds: float

    def check(self) -> None:
        """Raise what ends the task now, if anything (the caller aborts the agent first when one is running)."""
        if self.stop():
            raise BrowseStopped
        if time.monotonic() >= self.deadline:
            raise BrowseError(f"the task's {self.seconds:g} seconds passed",
                              f"I stopped the browser task after {self.seconds:g} seconds.")


class _Task:
    """One task's state. The loop thread writes it; `ending()` is also read by the thread asking a question."""

    def __init__(self, budget: _Budget, on_progress: Callable[[str, dict], None],
                 on_text: Callable[[str], None] | None, on_ask: Callable[[str, str, Callable[[], bool]], bool],
                 marker: str | None, started: float, gone: Callable[[], bool]):
        self.budget = budget
        self.on_progress, self.on_text, self.on_ask = on_progress, on_text, on_ask
        self.marker, self.started, self._gone = marker, started, gone
        self.interrupted = asyncio.Event()
        self.reason: str | None = None  # "abort" (the caller's stop or deadline), "declined" or "tripwire"
        self.active = False  # updates and permission requests count only from the prompt until the task's end
        self.finished = False  # `run()` is done: nothing asked for it waits any longer
        self.marker_seen = marker is None
        self.session_id: str | None = None
        self.chunks: list[str] = []
        self.tools = 0

    def interrupt(self, reason: str) -> None:
        if self.reason is None:
            self.reason = reason
            self.interrupted.set()

    def ending(self) -> bool:
        """Whether the task is over or ending, whatever the reason: a question for it need not wait any longer."""
        return (self.reason is not None or self.finished or self.budget.stop()
                or time.monotonic() >= self.budget.deadline or self._gone())

    def answer(self) -> str:
        return "".join(self.chunks).strip()

    def update(self, update: object) -> None:
        """One `session/update`. Never awaits, so updates are handled in the order they arrived."""
        if self.reason is not None:
            return
        if isinstance(update, AgentMessageChunk):
            content = update.content
            if not isinstance(content, TextContentBlock):
                return
            if not self.marker_seen and content.text.strip() == self.marker:
                self.marker_seen = True
                self.chunks.clear()  # whatever came before the gate's word is start-up chatter
            else:
                self.chunks.append(content.text)
        elif isinstance(update, ToolCallStart):
            if not self.marker_seen:
                self.interrupt("tripwire")
                return
            self.tools += 1
            said = self.answer()
            self.chunks.clear()  # a message ends at a tool call: what it said is chatter, the answer comes later
            if said:
                _call(self.on_text, said)
            log.info("browse: tool %s (%.1fs in)", _one_line(update.name or update.title, LOG_TOOL_CHARS),
                     time.monotonic() - self.started)
            _call(self.on_progress, update.title, update.raw_input if isinstance(update.raw_input, dict) else {})


def _call(callback: Callable | None, *args: Any) -> None:
    if callback is None:
        return
    try:
        callback(*args)
    except Exception:  # a callback must not kill the task
        log.warning("browse: a callback failed", exc_info=True)


def _one_line(text: str, limit: int) -> str:
    one = " ".join(text.split())
    return one if len(one) <= limit else f"{one[: limit - 3]}..."


def _option(options: Sequence[PermissionOption], *kinds: str) -> PermissionOption | None:
    """The first option of the first of `kinds` there is one of."""
    for kind in kinds:
        for option in options:
            if option.kind == kind:
                return option
    return None


def _consume(future: asyncio.Future) -> None:
    """Retrieve a finished future's outcome, so an abandoned one is not reported at exit."""
    if not future.cancelled():
        future.exception()


def _in_thread(fn: Callable[[], Any]) -> asyncio.Future:
    """`fn()` on a daemon thread of its own, as a future of the running loop (a blocking call must not hold the loop,
    and an executor's threads are joined at exit)."""
    loop = asyncio.get_running_loop()
    future: asyncio.Future = loop.create_future()

    def deliver(setter: Callable[[Any], None], value: Any) -> None:
        if not future.done():
            setter(value)

    def work() -> None:
        try:
            value = fn()
        except Exception as exc:  # noqa: BLE001 - handed to whoever awaits the future
            with contextlib.suppress(RuntimeError):  # the loop is closed: nobody waits
                loop.call_soon_threadsafe(deliver, future.set_exception, exc)
        else:
            with contextlib.suppress(RuntimeError):
                loop.call_soon_threadsafe(deliver, future.set_result, value)

    threading.Thread(target=work, name="vaice-browse-worker", daemon=True).start()
    return future


class _Client:
    """The ACP client side of one agent process (on the loop). Only what vaice answers is implemented: any other
    request is `method not found`."""

    def __init__(self, agent: "AcpAgent"):
        self._agent = agent
        self.state: _Task | None = None  # the running task, set from the prompt until the task ends
        self.asking = False  # a permission request is being decided
        self.answered = 0  # permission responses handed to the connection
        self.sent = 0  # ... and seen going out

    def observe(self, event: Any) -> None:
        """A stream observer: counts the permission responses that have been written (they carry an `outcome`)."""
        message = event.message
        if event.direction.value == "outgoing" and isinstance(message.get("result"), dict) \
                and "outcome" in message["result"]:
            self.sent += 1

    async def session_update(self, session_id: str, update: Any, **kwargs: Any) -> None:
        state = self.state
        if state is None or not state.active:
            return
        try:
            state.update(update)
        except Exception as exc:  # noqa: BLE001 - nothing raised here may reach the SDK's logging (it prints the text)
            log.warning("browse: could not handle an update from the browser agent (%s)", type(exc).__name__)

    async def request_permission(self, session_id: str, tool_call: ToolCallUpdate,
                                 options: list[PermissionOption], **kwargs: Any) -> RequestPermissionResponse:
        self.asking = True
        try:
            response = await self._decide(tool_call, options)
        except Exception as exc:  # noqa: BLE001 - a failed question is a no, and nothing raised reaches the SDK
            log.warning("browse: could not answer a permission request (%s)", type(exc).__name__)
            response = CANCELLED
        finally:
            self.answered += 1
            self.asking = False
        return response

    async def _decide(self, tool_call: ToolCallUpdate,
                      options: Sequence[PermissionOption]) -> RequestPermissionResponse:
        state = self.state
        if state is None or not state.active or state.reason is not None:
            return CANCELLED
        if not state.marker_seen:  # a request before the gate's word: the gate is not running
            state.interrupt("tripwire")
            return CANCELLED
        allow = _option(options, "allow_once", "allow_always")
        if self._agent.permissions == "allow":
            return _selected(allow)
        title = tool_call.title or ""
        raw = tool_call.raw_input if isinstance(tool_call.raw_input, dict) else {}
        message = raw["message"] if isinstance(raw.get("message"), str) else json.dumps(raw, default=str)
        decision = await self._ask(state, title, message[:RAW_INPUT_CHARS])
        if decision is None:  # stopped, out of time or closed: not a no; the task ends for what ended it
            state.interrupt("abort")
            return CANCELLED
        if decision:
            return _selected(_option(options, "allow_once") or allow)
        response = _selected(_option(options, "reject_once", "reject_always"))
        state.interrupt("declined")
        return response

    async def _ask(self, state: _Task, title: str, message: str) -> bool | None:
        """`on_ask` on a thread of its own. True or False is the user's answer; None when the task ended first."""
        asked_at = time.monotonic()
        timeout = self._agent.ask_timeout_s

        def ask_stop() -> bool:  # also turns true when the question has waited `ask_timeout_s`
            return state.ending() or time.monotonic() - asked_at >= timeout

        def ask() -> bool:
            try:
                return bool(state.on_ask(title, message, ask_stop))
            except Exception:  # a failed question is a no
                log.warning("browse: the question callback failed", exc_info=True)
                return False

        answer = _in_thread(ask)
        waiter = asyncio.ensure_future(state.interrupted.wait())
        try:
            await asyncio.wait({answer, waiter}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            waiter.cancel()
        if not answer.done() or state.ending():
            answer.cancel()
            return None
        return answer.result() and time.monotonic() - asked_at < timeout  # an expired question is a no

    async def ext_method(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        raise RequestError.method_not_found(method)

    async def ext_notification(self, method: str, params: dict[str, Any]) -> None:
        return None


def _selected(option: PermissionOption | None) -> RequestPermissionResponse:
    """The response choosing `option`; cancelled when the agent offered no such option."""
    if option is None:
        return CANCELLED
    return RequestPermissionResponse(outcome=AllowedOutcome(outcome="selected", optionId=option.option_id))


class _Proc:
    """One agent process and its connection. `end()` and `close_stdin()` are safe from any thread."""

    def __init__(self, proc: subprocess.Popen, loop: asyncio.AbstractEventLoop):
        self.proc = proc
        self.loop = loop
        self.dead = False  # set when it is known unusable (ended, wedged, gone): the next task starts a new one
        self.conn: Any = None  # the SDK's ClientSideConnection, made on the loop by `AcpAgent._connect`
        self.client: _Client | None = None
        self.can_close_session = False
        self._transports: list[asyncio.BaseTransport] = []  # the loop's ends of the pipes
        self._end_lock = threading.Lock()
        self._ended = False

    def attach(self, transports: Sequence[asyncio.BaseTransport]) -> None:
        self._transports = list(transports)

    def close_stdin(self) -> None:
        """Close the agent's stdin: an orderly shutdown (Pi then stops its MCP servers)."""
        with contextlib.suppress(RuntimeError):  # the loop is closed
            if self._transports:
                self.loop.call_soon_threadsafe(self._transports[-1].close)
            elif self.proc.stdin is not None:  # not connected yet
                self.proc.stdin.close()

    def end(self) -> None:
        """Mark it unusable and end it (`browse.end_process_group`): whatever it left running in its group ends with
        it, as in `shell.run`. Bounded, and once only; the connection's ends are closed on the loop."""
        with self._end_lock:
            self.dead = True
            if self._ended:
                return
            self._ended = True
        browse.end_process_group(self.proc)
        with contextlib.suppress(RuntimeError, OSError, ValueError):
            self.loop.call_soon_threadsafe(self._release)

    def _release(self) -> None:  # on the loop
        for transport in self._transports:
            transport.close()
        if self.conn is not None:
            future = asyncio.ensure_future(self.conn.close())
            future.add_done_callback(_consume)
        if not self._transports:  # never connected: the pipes are still ours
            for pipe in (self.proc.stdin, self.proc.stdout):
                if pipe is not None:
                    with contextlib.suppress(OSError, ValueError):
                        pipe.close()


class AcpAgent:
    """Drives one ACP agent process. `run()` is thread-safe (tasks are serialized); `close()` ends the agent."""

    def __init__(self, argv: Sequence[str], *, env: Mapping[str, str], cwd: Path, log_path: Path,
                 mcp_servers: Sequence[McpServerStdio] = (), permissions: Literal["ask", "allow"] = "ask",
                 ask_timeout_s: float = 60.0, marker: str | None = None, chrome_profile: Path | None = None,
                 secrets: Sequence[str] = ()):
        """`argv` starts the agent (it speaks ACP on stdin/stdout) with exactly `env` as its environment, in `cwd`,
        its stderr appended to `log_path` (0600). `mcp_servers` go to every `session/new`. `permissions`: "ask"
        sends every permission request to `on_ask`, "allow" answers each allow-once without asking.
        `ask_timeout_s` is how long a question gets before it counts as a no. `marker` is the text of the gate's
        notification, the tripwire (see the module docstring). `chrome_profile` is the browser profile the agent's
        browser uses: what still holds it is ended before a new process starts and after `close()`. `secrets` are
        values scrubbed from anything the agent reports."""
        self._argv = list(argv)
        self._env = dict(env)
        self._cwd = cwd
        self._log_path = log_path
        self._mcp_servers = list(mcp_servers)
        self.permissions = permissions
        self.ask_timeout_s = ask_timeout_s
        self._marker = marker
        self._chrome_profile = chrome_profile
        self._secrets = tuple(value for value in secrets if value)
        self._proc: _Proc | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_thread: threading.Thread | None = None
        self._closed = False
        self._registered = False
        self._state_lock = threading.Lock()  # the process, the loop, `_closed` (close() never waits for a task)
        self._run_lock = threading.Lock()  # one task at a time

    # -- the loop and the process

    def _scrub(self, text: str) -> str:
        """`text` for a log line or an error: the secrets blanked."""
        for value in self._secrets:
            text = text.replace(value, "***")
        return text

    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        """The loop's thread, started with the first task (under `_state_lock`)."""
        if self._loop is None:
            loop = asyncio.new_event_loop()
            self._loop = loop
            self._loop_thread = threading.Thread(target=self._loop_main, args=(loop,), name="vaice-browse-loop",
                                                 daemon=True)
            self._loop_thread.start()
        return self._loop

    @staticmethod
    def _loop_main(loop: asyncio.AbstractEventLoop) -> None:
        asyncio.set_event_loop(loop)
        try:
            loop.run_forever()
        finally:
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            with contextlib.suppress(Exception):  # a task that fails while it is cancelled: the loop ends anyway
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            loop.close()

    def _spawn(self, loop: asyncio.AbstractEventLoop) -> _Proc:
        self._cwd.mkdir(mode=0o700, parents=True, exist_ok=True)
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        err_fd = os.open(self._log_path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        try:
            popen = subprocess.Popen(self._argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=err_fd,
                                     cwd=self._cwd, env=self._env, start_new_session=True)
        except OSError as exc:
            raise BrowseError(f"could not start the browser agent: {exc.strerror or type(exc).__name__}",
                              browse.CANT_START) from None
        finally:
            os.close(err_fd)  # the child has its own copy
        return _Proc(popen, loop)

    def _ensure_process(self) -> tuple[asyncio.AbstractEventLoop, _Proc]:
        """The running agent process, a new one when there is none or it died (what it left holding the browser
        profile is ended first)."""
        with self._state_lock:
            if self._closed:
                raise BrowseError("the browser agent was closed")
            loop = self._ensure_loop()
            proc = self._proc
            if proc is not None and not proc.dead and proc.proc.poll() is None:
                return loop, proc
            if proc is not None:
                proc.end()  # reaps it and what it left in its group
            if self._chrome_profile is not None:
                browse.end_profile_processes(self._chrome_profile)  # the browser it started is in no group of its
            proc = self._proc = self._spawn(loop)
            if not self._registered:
                self._registered = True
                atexit.register(self.close)  # backstop; main()'s finally closes first
        log.info("browse: started the browser agent (pid %d)", proc.proc.pid)
        return loop, proc

    async def _connect(self, proc: _Proc, state: _Task) -> None:
        """Hand the process's pipes to the SDK and `initialize` (once per process)."""
        loop = asyncio.get_running_loop()
        reader = asyncio.StreamReader(limit=STREAM_LIMIT)
        read_transport, _ = await loop.connect_read_pipe(lambda: asyncio.StreamReaderProtocol(reader),
                                                         proc.proc.stdout)
        write_transport, write_protocol = await loop.connect_write_pipe(asyncio.streams.FlowControlMixin,
                                                                        proc.proc.stdin)
        proc.attach([read_transport, write_transport])
        writer = asyncio.StreamWriter(write_transport, write_protocol, None, loop)
        proc.client = client = _Client(self)
        proc.conn = connect_to_agent(client, writer, reader, observers=[client.observe])
        try:
            version = metadata.version("vaice")
        except metadata.PackageNotFoundError:
            version = "unknown"
        init = await self._guard(proc, state, proc.conn.initialize(
            protocol_version=PROTOCOL_VERSION, client_capabilities=ClientCapabilities(),
            client_info=Implementation(name="vaice", title="vaice", version=version)),
            browse.START_TIMEOUT, "initialize", browse.CANT_START)
        if init.protocol_version != PROTOCOL_VERSION:
            await self._end(proc)
            raise BrowseError(f"the browser agent speaks ACP version {init.protocol_version}, not {PROTOCOL_VERSION}",
                              browse.CANT_START)
        capabilities = init.agent_capabilities
        sessions = capabilities.session_capabilities if capabilities else None
        proc.can_close_session = sessions is not None and sessions.close is not None

    # -- a task

    def run(self, goal: str, *, stop: Callable[[], bool], on_progress: Callable[[str, dict], None],
            on_ask: Callable[[str, str, Callable[[], bool]], bool], deadline_s: float,
            on_text: Callable[[str], None] | None = None) -> str:
        """Do `goal` in the browser and return the answer text.

        `on_progress(title, args)` is called as each tool starts and `on_text(text)` with each message that ends
        at a tool call and with the final one (an exception from either is logged, nothing more).
        `on_ask(title, message, stop)` answers a permission request and may block: it runs on a thread of its own,
        and IT owns honoring `stop`, which it is given and which also turns true when the question has waited
        `ask_timeout_s` or the task ends. A False (or an exception) is a no, and ends the task.

        Raises `BrowseStopped` when `stop()` ends it, `BrowseDeclined` after a no, and a `BrowseError` otherwise
        (also when `deadline_s` passes: the agent is cancelled first)."""
        while not self._run_lock.acquire(timeout=browse.POLL_SECONDS):
            if stop():
                raise BrowseStopped
        try:
            started = time.monotonic()
            budget = _Budget(stop, started + deadline_s, deadline_s)
            budget.check()
            loop, proc = self._ensure_process()
            state = _Task(budget, on_progress, on_text, on_ask, self._marker, started,
                          lambda: proc.dead or self._closed)
            future = asyncio.run_coroutine_threadsafe(self._task(proc, state, goal), loop)
            try:
                return self._wait(loop, proc, state, future, budget)
            finally:
                state.finished = True
        except (BrowseError, BrowseStopped):
            raise
        except Exception as exc:  # noqa: BLE001 - nothing but a BrowseError or BrowseStopped leaves
            log.warning("browse: the task failed (%s)", type(exc).__name__)
            raise BrowseError(f"the task failed ({type(exc).__name__})") from None
        finally:
            self._run_lock.release()

    def _wait(self, loop: asyncio.AbstractEventLoop, proc: _Proc, state: _Task, future: concurrent.futures.Future,
              budget: _Budget) -> str:
        """The task's outcome; the deadline and `stop()` are checked every `POLL_SECONDS`."""
        while True:
            try:
                return future.result(timeout=browse.POLL_SECONDS)
            except concurrent.futures.TimeoutError:
                pass
            except concurrent.futures.CancelledError:  # close() ended the loop
                raise BrowseError("the browser agent was closed") from None
            except _Interrupted:  # a question was left unanswered because the task is ending: say why
                budget.check()
                raise BrowseError("the browser agent was closed") from None
            try:
                budget.check()
            except (BrowseStopped, BrowseError):
                self._abort(loop, proc, state, future)
                raise

    def _abort(self, loop: asyncio.AbstractEventLoop, proc: _Proc, state: _Task,
               future: concurrent.futures.Future) -> None:
        """Stop the running task: the coroutine answers a pending question, cancels the session and waits for the
        prompt to end, ending the agent when it does not. Waits for that, bounded."""
        with contextlib.suppress(RuntimeError):
            loop.call_soon_threadsafe(state.interrupt, "abort")
        try:
            future.result(timeout=browse.ABORT_WAIT + browse.TERM_WAIT + browse.KILL_WAIT + 1.0)
        except concurrent.futures.TimeoutError:  # the loop itself does not answer
            log.warning("browse: the browser task did not settle; ending the browser agent")
            future.cancel()
            proc.end()
        except (BrowseError, BrowseStopped, _Interrupted, concurrent.futures.CancelledError):
            pass  # how the task ended is not wanted: the caller raises why it stopped it

    async def _task(self, proc: _Proc, state: _Task, goal: str) -> str:
        client: _Client | None = None
        try:
            if proc.conn is None:
                await self._connect(proc, state)
            client = proc.client
            session = await self._guard(proc, state, proc.conn.new_session(
                cwd=str(self._cwd), mcp_servers=self._mcp_servers), browse.NEW_SESSION_TIMEOUT, "new_session")
            state.session_id = session.session_id
            return await self._prompt(proc, state, client, goal)
        except (BrowseError, _Interrupted):
            raise
        except Exception as exc:  # noqa: BLE001 - nothing but a BrowseError leaves
            raise self._failure(proc, exc, "the task") from None
        finally:
            state.active = False
            if client is not None:
                client.state = None

    async def _prompt(self, proc: _Proc, state: _Task, client: _Client | None, goal: str) -> str:
        assert client is not None
        client.state = state
        state.active = True
        request = proc.conn.prompt(session_id=state.session_id,
                                   prompt=[TextContentBlock(type="text", text=browse.PROMPT.format(goal=goal))])
        prompt = asyncio.ensure_future(request)
        prompt.add_done_callback(_consume)
        waiter = asyncio.ensure_future(state.interrupted.wait())
        try:
            await asyncio.wait({prompt, waiter}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            waiter.cancel()
        if state.reason is not None:
            await self._settle(proc, state, client, prompt)
            if state.reason == "declined":
                raise BrowseDeclined
            if state.reason == "tripwire":
                log.warning("browse: the browser agent's safety gate did not report in; the agent was ended")
                raise BrowseError("the browser agent's safety gate did not report in")
            raise _Interrupted
        try:
            response = prompt.result()
        except Exception as exc:  # noqa: BLE001 - becomes a BrowseError
            raise self._failure(proc, exc, "the prompt") from None
        await self._close_session(proc, state)
        reason = response.stop_reason
        log.info("browse: done in %.1fs, %d tool calls (%s)", time.monotonic() - state.started, state.tools, reason)
        if reason != "end_turn":
            raise BrowseError(f"the browser agent stopped without an answer ({reason})")
        answer = state.answer()
        if not answer:
            raise BrowseError("the browser agent finished without text", browse.NO_ANSWER)
        _call(state.on_text, answer)
        return answer

    async def _settle(self, proc: _Proc, state: _Task, client: _Client, prompt: asyncio.Future) -> None:
        """The task is ending early: let the answer to a pending question go out, `session/cancel`, then wait for the
        prompt to end. The agent is ended when it does not (`ABORT_WAIT`) and, always, after a tripwire."""
        end = state.reason == "tripwire"
        give_up = time.monotonic() + ANSWER_WAIT
        while (client.asking or client.sent < client.answered) and time.monotonic() < give_up:
            await asyncio.sleep(0.01)
        with contextlib.suppress(ConnectionError, OSError):
            await proc.conn.cancel(session_id=state.session_id)
        if not end:
            await asyncio.wait({prompt}, timeout=browse.ABORT_WAIT)
            if not prompt.done():
                log.warning("browse: the prompt did not end %g seconds after the cancel; ending the browser agent",
                            browse.ABORT_WAIT)
                end = True
        if end:
            await self._end(proc)
        if not prompt.done():
            prompt.cancel()

    async def _close_session(self, proc: _Proc, state: _Task) -> None:
        """`session/close`, only when the agent advertised it; a failure is a warning (the next task starts fresh)."""
        if not proc.can_close_session:
            return
        try:
            await asyncio.wait_for(proc.conn.close_session(session_id=state.session_id), SESSION_CLOSE_WAIT)
        except (TimeoutError, ConnectionError, RequestError, OSError) as exc:
            log.warning("browse: could not close the session (%s)", type(exc).__name__)

    async def _end(self, proc: _Proc) -> None:
        await _in_thread(proc.end)

    async def _guard(self, proc: _Proc, state: _Task, awaitable: Any, timeout: float, what: str,
                     spoken: str = browse.FAILED) -> Any:
        """The result of `awaitable`. A task that is interrupted meanwhile ends the agent (a half-made session is
        not worth reusing) and raises `_Interrupted`; one that does not answer in `timeout` ends it too."""
        work = asyncio.ensure_future(awaitable)
        work.add_done_callback(_consume)
        waiter = asyncio.ensure_future(state.interrupted.wait())
        try:
            await asyncio.wait({work, waiter}, timeout=timeout, return_when=asyncio.FIRST_COMPLETED)
        finally:
            waiter.cancel()
        if work.done() and not work.cancelled():
            try:
                return work.result()
            except Exception as exc:  # noqa: BLE001 - becomes a BrowseError
                raise self._failure(proc, exc, what, spoken) from None
        work.cancel()
        await self._end(proc)
        if state.interrupted.is_set():
            raise _Interrupted
        log.warning("browse: the browser agent did not answer %s in %g seconds; it was ended", what, timeout)
        raise BrowseError(f"the browser agent did not answer {what}", spoken)

    def _failure(self, proc: _Proc, exc: BaseException, what: str, spoken: str = browse.FAILED) -> BrowseError:
        """The `BrowseError` for an exception out of the SDK: what an agent said about its own failure (scrubbed,
        capped), or only the exception's type."""
        if isinstance(exc, BrowseError):
            return exc
        if isinstance(exc, RequestError):
            return BrowseError(f"{what} failed: {self._scrub(str(exc))[:200]}", spoken)
        if isinstance(exc, ConnectionError | EOFError | BrokenPipeError):
            proc.dead = True
            return BrowseError("the browser agent exited" if not self._closed else "the browser agent was closed",
                               spoken)
        log.warning("browse: %s failed (%s)", what, type(exc).__name__)
        return BrowseError(f"{what} failed ({type(exc).__name__})", spoken)

    # -- closing

    def close(self) -> None:
        """End the agent: close its stdin (an orderly shutdown, which stops its MCP servers), then SIGTERM and
        SIGKILL its group, then the loop's thread. Idempotent; never waits for a running task, which then fails with a
        `BrowseError`."""
        atexit.unregister(self.close)
        with self._state_lock:
            self._closed = True
            proc, self._proc = self._proc, None
            loop, thread = self._loop, self._loop_thread
            self._loop = self._loop_thread = None
        if proc is not None:
            proc.dead = True
            proc.close_stdin()
            try:
                proc.proc.wait(timeout=browse.CLOSE_WAIT)
            except subprocess.TimeoutExpired:
                log.warning("browse: the browser agent did not exit within %g seconds of its stdin closing",
                            browse.CLOSE_WAIT)
            proc.end()  # whatever is left of its group
        if loop is not None:
            with contextlib.suppress(RuntimeError):
                loop.call_soon_threadsafe(loop.stop)
        if thread is not None:
            thread.join(timeout=browse.KILL_WAIT)
        if proc is not None and self._chrome_profile is not None:
            browse.end_profile_processes(self._chrome_profile)
