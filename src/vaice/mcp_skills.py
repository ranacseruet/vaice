"""MCP (Model Context Protocol) tools as skills (`VAICE_MCP_CONFIG`, assistant mode only).

A JSON file in Claude Desktop's `mcpServers` shape names stdio servers for vaice to start and, per server, the
tools to offer and how each is approved:

    {"mcpServers": {"notes": {"command": "uvx", "args": ["some-notes-mcp"], "env": {"NOTES_DIR": "~/Notes"},
                              "tools": {"search_notes": "run", "add_note": "auto", "delete_note": "click"}}}}

Each listed tool becomes a skill called `<server>__<tool>`. Only listed tools are offered: a server can offer
anything, and vaice cannot tell what a tool does (a server's own annotations such as `readOnlyHint` are claims, not
facts, and change nothing). The policy is the user's: `run` (a `read` skill: runs at once), `voice` (a `confirm`
skill a spoken yes or the Run button approves), `auto` (a `confirm` skill that runs by itself
`VAICE_ASSISTANT_AUTO_SECONDS` after its announcement was spoken, unless the user presses Cancel or says no; a spoken
yes or the Run button runs it at once), `click` (a `confirm` skill only the Run button approves, the default). The
confirmation question names the tool and the server, never the arguments, which come from the model. An `auto` call
is asked as a `click` one when another skill ran earlier in the same reply (its output may be untrusted text), and
with the setting at 0 an `auto` tool is a `click` tool. What an earlier turn's tool said is not tracked.

What a server sends is untrusted, like any tool output: a tool's description (which the brain reads) is cleaned and
capped, and a result reaches the brain only as a `tool` message. `remember` refuses once such a skill has run in the
reply (see `memory.py`).

Servers run in one asyncio loop on a daemon thread (`vaice-mcp`): each server's client is entered inside a task of
its own that lives until `close()`, because the SDK's cancel scopes must be entered and left by the same task. A
skill call comes from a reply's thread (or a Gradio worker, for a click), submits the call to that loop and polls
`stop()` and the deadline every `POLL_SECONDS`, like `shell.py`. The `mcp` package is the optional `mcp` extra and
is imported only here, only when `VAICE_MCP_CONFIG` is set.
"""
import asyncio
import atexit
import concurrent.futures
import contextlib
import json
import logging
import os
import re
import tempfile
import threading
import time
from collections.abc import Callable, Coroutine, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TextIO

from vaice import config
from vaice.confirm import NO, YES
from vaice.skills import Question, Skill, SkillError, SkillResult

log = logging.getLogger(__name__)

# the config's word -> the skill's kind
POLICIES = {"run": "read", "voice": "confirm", "auto": "confirm", "click": "confirm"}
DEFAULT_POLICY = "click"  # a tool can do anything: the user opts into anything looser, tool by tool
MAX_NAME = 64  # a skill name is an OpenAI function name: [A-Za-z0-9_-], at most 64
MAX_SERVER_NAME = 32  # leaves room for the tool in `<server>__<tool>`
MAX_DESCRIPTION = 300  # characters of a server-supplied description the brain gets
MCP_OUTPUT_CHARS = 2000  # characters of a result the brain gets (the whole text stays in the panel's detail)
DETAIL_CHARS = 20000  # characters of a result kept for the panel
ERROR_CHARS = 300  # characters of a tool's own error message that are spoken
POLL_SECONDS = 0.05  # how often a running call is checked against `stop()` and the deadline
CALL_GRACE = 1.0  # seconds past the SDK's own timeout before vaice gives up on a call itself
CLOSE_WAIT = 10.0  # seconds `close()` waits for the servers to end (each one's shutdown is bounded too)
MAX_PAGES = 20  # pages of `tools/list` read from one server
MAX_NAMES_LOGGED = 20
NOTHING = "the tool returned nothing"
CUT = "(more on screen)"
INSTALL_HINT = ("vaice: VAICE_MCP_CONFIG is set, but the MCP package (mcp 2.3 or newer) is not installed. "
                "Install it with: uv sync --extra mcp")

ASK_VOICE = "Shall I use {tool} from {server}? The details are on the screen."
ASK_CLICK = "I would use {tool} from {server}. The details are on the screen; click Run to allow the call."
# Spoken when an `auto` call is announced: no word the user answers with, since speaker echo could say yes (or stop)
# to it. The countdown is the announcement itself, kept short because its playback time is added to the wait (a first
# live run, with a longer sentence, waited 12 s); the details and the Cancel button are on the screen
ASK_AUTO = "Using {tool} from {server} in {seconds}."
ASK_AUTO_GENERIC = "Using one of the connected tools in {seconds}."
# For a name that holds a word the user answers with ("stop_timer"): speaker echo could say yes to it
ASK_VOICE_GENERIC = "Shall I use one of the connected tools? The details are on the screen."
ASK_CLICK_GENERIC = ("I would use one of the connected tools. The details are on the screen; "
                     "click Run to allow the call.")

_NAME = re.compile(rf"[A-Za-z0-9_-]{{1,{MAX_SERVER_NAME}}}")
_SERVER_KEYS = ("type", "command", "args", "env", "cwd", "tools")
# every word of every yes and no phrase, as `confirm` compares them ("don't" is "dont")
_ANSWER_WORDS = frozenset(word for phrase in YES | NO for word in re.sub(r"[^\w\s]", "", phrase.casefold()).split())


def _bad(message: str) -> SystemExit:
    return SystemExit(f"vaice: VAICE_MCP_CONFIG: {message}")


# --- the config file


@dataclass(frozen=True)
class ServerConfig:
    name: str
    command: str
    args: tuple[str, ...] = ()
    env: dict[str, str] = field(default_factory=dict, repr=False)  # may hold tokens: in no repr, log or error
    cwd: str | None = None
    tools: dict[str, str] = field(default_factory=dict)  # tool name -> its policy, file order


def _expand(value: str) -> str:
    """A leading `~` as the home directory (a server is started without a shell, which would have done it)."""
    return os.path.expanduser(value) if value == "~" or value.startswith("~/") else value


def _text_list(server: str, key: str, value: Any) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) and "\0" not in item for item in value):
        raise _bad(f"server {server}: {key} must be a list of text")
    return value


def _server_config(name: str, entry: Any) -> ServerConfig:
    if not _NAME.fullmatch(name):
        raise _bad(f"server name {name!r} must be 1 to {MAX_SERVER_NAME} letters, digits, underscores or dashes")
    if not isinstance(entry, dict):
        raise _bad(f"server {name} must be an object with a command")
    if "url" in entry:
        raise _bad(f"server {name}: remote servers (url) are not supported yet; "
                   "only stdio servers (command, args, env)")
    unknown = sorted(set(entry) - set(_SERVER_KEYS))
    if unknown:
        raise _bad(f"server {name}: unknown key {', '.join(map(repr, unknown))} (known: {', '.join(_SERVER_KEYS)})")
    if entry.get("type", "stdio") != "stdio":
        raise _bad(f"server {name}: only stdio servers are supported")
    command = entry.get("command")
    if not isinstance(command, str) or not command.strip() or "\0" in command:
        raise _bad(f"server {name}: command is required and must be text")
    args = [_expand(arg) for arg in _text_list(name, "args", entry.get("args", []))]
    env = entry.get("env", {})
    if not isinstance(env, dict):
        raise _bad(f"server {name}: env must be an object of text values")
    for key, value in env.items():  # the values may be tokens: only the key is ever named
        if not key or "=" in key or "\0" in key or not isinstance(value, str) or "\0" in value:
            raise _bad(f"server {name}: env must map names to text values (check {key!r})")
    cwd = entry.get("cwd")
    if cwd is not None and (not isinstance(cwd, str) or not cwd.strip() or "\0" in cwd):
        raise _bad(f"server {name}: cwd must be text")
    return ServerConfig(name, command.strip(), tuple(args), {key: _expand(value) for key, value in env.items()},
                        None if cwd is None else _expand(cwd), _policies(name, entry.get("tools")))


def _policies(server: str, tools: Any) -> dict[str, str]:
    """Tool name -> policy from the `tools` value: an object, or a list of names (all `click`)."""
    if tools is None:
        return {}
    if isinstance(tools, list):
        if not all(isinstance(tool, str) and tool for tool in tools):
            raise _bad(f"server {server}: tools must be an object of policies or a list of tool names")
        return dict.fromkeys(tools, DEFAULT_POLICY)
    if not isinstance(tools, dict):
        raise _bad(f"server {server}: tools must be an object of policies or a list of tool names")
    for tool, policy in tools.items():
        if not tool or not isinstance(policy, str) or policy not in POLICIES:
            raise _bad(f"server {server}: the policy of tool {tool!r} must be one of {', '.join(POLICIES)}")
    return dict(tools)


def load_config(path: Path) -> list[ServerConfig]:
    """The servers in the file at `path`. Exits (`SystemExit`, naming the server and the problem, never an `env`
    value) for a file that cannot be used."""
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise _bad(f"cannot read {path}: {exc.strerror or type(exc).__name__}") from None
    except UnicodeDecodeError:
        raise _bad(f"{path} is not UTF-8 text") from None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:  # its message never holds the text, only what was expected
        raise _bad(f"{path} is not valid JSON (line {exc.lineno}, column {exc.colno}): {exc.msg}") from None
    servers = data.get("mcpServers") if isinstance(data, dict) else None
    if not isinstance(servers, dict):
        raise _bad(f'{path} must be an object with an "mcpServers" object of servers')
    if not servers:
        log.warning("VAICE_MCP_CONFIG: %s lists no servers", path)
    return [_server_config(name, entry) for name, entry in servers.items()]


# --- what the brain and the user see of a tool


def skill_name(server: str, tool: str) -> str:
    """`<server>__<tool>` restricted to [A-Za-z0-9_-] (anything else becomes `_`), at most `MAX_NAME` characters."""
    return re.sub(r"[^A-Za-z0-9_-]", "_", f"{server}__{tool}")[:MAX_NAME]


def _one_line(text: str, limit: int) -> str:
    """`text` as a single line: control characters gone, runs of whitespace one space, cut at `limit`."""
    printable = "".join(" " if char.isspace() else char for char in text if char.isspace() or char.isprintable())
    line = " ".join(printable.split())
    return line if len(line) <= limit else line[: limit - 3].rstrip() + "..."


def _describe(server: str, tool: Any) -> str:
    """The skill's description for the brain: the server's text (untrusted: tool-description poisoning is a real
    attack) cleaned to one capped line, behind the server's name."""
    text = ""
    for candidate in (tool.description, tool.title, tool.annotations.title if tool.annotations else None, tool.name):
        text = _one_line(candidate or "", MAX_DESCRIPTION)
        if text:
            break
    return f"(from {server}) {text}"


def _clean_schema(node: Any) -> Any:
    """`node` (a JSON schema) with every `description` string cleaned like a tool's: the schema is what the
    server validates against, so it is passed on, but its descriptions are text the brain reads too."""
    if isinstance(node, dict):
        return {key: _one_line(value, MAX_DESCRIPTION) if key == "description" and isinstance(value, str)
                else _clean_schema(value) for key, value in node.items()}
    if isinstance(node, list):
        return [_clean_schema(item) for item in node]
    return node


def _parameters(server: str, tool: Any) -> dict:
    schema = tool.input_schema
    if not isinstance(schema, dict) or schema.get("type") != "object":
        log.warning("mcp server %s: the input schema of %s is not an object schema; offering it without arguments",
                    server, tool.name)
        return {"type": "object", "properties": {}}
    schema = _clean_schema(schema)
    required = schema.get("required")
    if required is not None and not (isinstance(required, list) and all(isinstance(key, str) for key in required)):
        del schema["required"]  # `Skills` checks it key by key
    return schema


def _speakable(name: str) -> str:
    return " ".join(re.sub(r"[^A-Za-z0-9]+", " ", name).split())


def question(server: str, tool: str, *, voice: bool) -> Question:
    """What is asked before a `voice` or `click` tool runs: fixed sentences that name the tool and the server, never
    the arguments. A name that holds a word the user answers with gets a sentence without names (see `confirm.py`)."""
    tool_words, server_words = _speakable(tool), _speakable(server)
    named = tool_words and server_words and not _ANSWER_WORDS & set(f"{tool_words} {server_words}".casefold().split())
    if named:
        return Question((ASK_VOICE if voice else ASK_CLICK).format(tool=tool_words, server=server_words), voice=voice)
    return Question(ASK_VOICE_GENERIC if voice else ASK_CLICK_GENERIC, voice=voice)


def auto_question(server: str, tool: str, seconds: float) -> Question:
    """What is announced before an `auto` tool runs by itself in `seconds`: a fixed sentence like `question`'s (names
    only, no answer words), with the click-only question as the `fallback` for a reply where something ran before it."""
    tool_words, server_words = _speakable(tool), _speakable(server)
    named = tool_words and server_words and not _ANSWER_WORDS & set(f"{tool_words} {server_words}".casefold().split())
    wait = f"{seconds:g} second" + ("" if seconds == 1 else "s")
    text = (ASK_AUTO.format(tool=tool_words, server=server_words, seconds=wait) if named
            else ASK_AUTO_GENERIC.format(seconds=wait))
    return Question(text, auto_seconds=seconds, fallback=question(server, tool, voice=False))


def _result_text(content: list) -> str:
    """The text of a result's content blocks, one per line; what is not text is named (`[image]`)."""
    from mcp import types

    lines = []
    for block in content:
        if isinstance(block, types.TextContent):
            lines.append(block.text)
        elif isinstance(block, types.ImageContent):
            lines.append("[image]")
        elif isinstance(block, types.AudioContent):
            lines.append("[audio]")
        elif isinstance(block, types.EmbeddedResource):
            inner = block.resource
            lines.append(inner.text if isinstance(inner, types.TextResourceContents)
                         else f"[resource {_one_line(str(inner.uri), 100)}]")
        elif isinstance(block, types.ResourceLink):
            lines.append(f"[resource {_one_line(str(block.uri), 100)}]")
        else:
            lines.append("[unknown content]")
    return "\n".join(lines)


# --- the servers


class _Server:
    """One started server: its client lives in `task` on the loop until `stop` is set."""

    def __init__(self, config_: ServerConfig):
        self.config = config_
        self.client: Any = None
        self.tools: list = []
        self.stop = asyncio.Event()
        self.task: asyncio.Task | None = None


@contextlib.contextmanager
def _errlog(name: str) -> Iterator[TextIO]:
    """Where a server's stderr goes: `<DATA_DIR>/mcp-<name>.log` (a temp dir when DATA_DIR is empty), mode 0600
    because a server may print what it was given. Its logs would otherwise fill vaice's terminal."""
    if config.DATA_DIR is None:
        directory = Path(tempfile.mkdtemp(prefix="vaice-"))
    else:
        directory = config.DATA_DIR
        directory.mkdir(parents=True, exist_ok=True)
    fd = os.open(directory / f"mcp-{name}.log", os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
    with os.fdopen(fd, "a", encoding="utf-8") as out:
        yield out


def _why(exc: BaseException) -> str:
    """A failure as one short line (the first of an exception group's); the `env` values are not in any of these."""
    while isinstance(exc, BaseExceptionGroup) and exc.exceptions:
        exc = exc.exceptions[0]
    return _one_line(f"{type(exc).__name__}: {exc}", 200)


async def _list_tools(client: Any) -> list:
    tools: list = []
    cursor = None
    for _ in range(MAX_PAGES):
        page = await client.list_tools(cursor=cursor)
        tools += page.tools
        cursor = page.next_cursor
        if cursor is None:
            return tools
    log.warning("mcp: a server listed more than %d pages of tools; the rest are ignored", MAX_PAGES)
    return tools


class McpTools:
    """The MCP servers vaice started and the skills they offer (`skills`, in config order). `close()` ends the
    servers; it can be called any number of times."""

    def __init__(self, timeout: float, auto_seconds: float = 0.0):
        self.timeout = timeout
        self.auto_seconds = auto_seconds  # what an `auto` tool waits before it runs; 0: it is a `click` tool
        self.skills: list[Skill] = []
        self.summaries: list[tuple[str, list[str]]] = []  # per server that started: its name, the tools offered
        self._servers: list[_Server] = []
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._closed = False
        self._lock = threading.Lock()

    # -- starting

    def _start(self, configs: list[ServerConfig]) -> None:
        if not configs:
            return
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, name="vaice-mcp", daemon=True)
        self._thread.start()
        atexit.register(self.close)  # backstop; main()'s finally closes first
        asyncio.run_coroutine_threadsafe(self._start_all(configs), self._loop).result()
        taken: dict[str, str] = {}
        for server in self._servers:
            self._offer(server, taken)

    async def _start_all(self, configs: list[ServerConfig]) -> None:
        await asyncio.gather(*(self._start_one(cfg) for cfg in configs))

    async def _start_one(self, cfg: ServerConfig) -> None:
        server = _Server(cfg)
        ready: asyncio.Future = asyncio.get_running_loop().create_future()
        server.task = asyncio.create_task(self._serve(server, ready), name=f"vaice-mcp-{cfg.name}")
        self._servers.append(server)  # `close()` stops it even while it is still starting
        try:
            await asyncio.wait_for(ready, self.timeout)
        except TimeoutError:
            log.warning("mcp server %s did not start within %g seconds (VAICE_MCP_TIMEOUT); skipping it", cfg.name,
                        self.timeout)
        except Exception as exc:  # noqa: BLE001 - whatever the server's start failed with: said, then skipped
            log.warning("mcp server %s did not start: %s (its stderr: mcp-%s.log in the data directory)",
                        cfg.name, _why(exc), cfg.name)
        else:
            return
        self._servers.remove(server)
        server.task.cancel()  # a start in flight is cancelled, which also ends its child process
        await asyncio.gather(server.task, return_exceptions=True)

    async def _serve(self, server: _Server, ready: asyncio.Future) -> None:
        """The server's whole life, in one task (the SDK's cancel scopes need it): start, list, wait for `stop`."""
        from mcp import Client, StdioServerParameters, stdio_client

        cfg = server.config
        try:
            with _errlog(cfg.name) as errlog:
                params = StdioServerParameters(command=cfg.command, args=list(cfg.args), env=cfg.env or None,
                                               cwd=cfg.cwd)
                async with Client(stdio_client(params, errlog=errlog), read_timeout_seconds=self.timeout) as client:
                    server.client = client
                    server.tools = await _list_tools(client)
                    if not ready.done():
                        ready.set_result(None)
                    await server.stop.wait()
        except Exception as exc:  # noqa: BLE001 - delivered to `_start_one`, or said when the server was up already
            if ready.done():
                log.warning("mcp server %s stopped: %s", cfg.name, _why(exc))
            else:
                ready.set_exception(exc)

    def _offer(self, server: _Server, taken: dict[str, str]) -> None:
        """Turn the listed tools of `server` that it really offers into skills; say what does not match."""
        cfg = server.config
        by_name = {tool.name: tool for tool in server.tools}
        offered = []
        for tool_name, policy in cfg.tools.items():
            tool = by_name.get(tool_name)
            if tool is None:
                log.warning("mcp server %s: the config lists the tool %s, which the server does not offer",
                            cfg.name, tool_name)
                continue
            skill = self._skill(server, tool, policy)
            if skill.name in taken:
                raise _bad(f"the tools {taken[skill.name]} and {cfg.name}.{tool_name} both become the skill "
                           f"{skill.name}; rename a server")
            taken[skill.name] = f"{cfg.name}.{tool_name}"
            log.debug("mcp %s.%s annotations (claims by the server, not used): %s", cfg.name, tool_name,
                      tool.annotations)
            self.skills.append(skill)
            offered.append(tool_name)
        unlisted = [name for name in by_name if name not in cfg.tools]
        if unlisted:
            shown = ", ".join(unlisted[:MAX_NAMES_LOGGED]) + (" ..." if len(unlisted) > MAX_NAMES_LOGGED else "")
            log.info("mcp server %s also offers %d tool(s) the config does not list, so they are not offered "
                     "(list them under \"tools\" to use them): %s", cfg.name, len(unlisted), shown)
        self.summaries.append((cfg.name, offered))

    def _skill(self, server: _Server, tool: Any, policy: str) -> Skill:
        cfg = server.config
        kind = POLICIES[policy]
        if policy == "auto" and self.auto_seconds > 0:
            asked = auto_question(cfg.name, tool.name, self.auto_seconds)
        else:
            asked = question(cfg.name, tool.name, voice=policy == "voice")

        def run(args: dict, stop: Callable[[], bool]) -> SkillResult:
            return self._call(server, tool.name, args, stop)

        return Skill(skill_name(cfg.name, tool.name), _describe(cfg.name, tool), _parameters(cfg.name, tool),
                     lambda args: run(args, lambda: False), kind=kind,
                     ask=None if kind == "read" else lambda _args: asked, run_stoppable=run)

    # -- calling

    def _submit(self, coroutine: Coroutine) -> concurrent.futures.Future:
        with self._lock:
            if self._closed or self._loop is None:
                coroutine.close()
                raise SkillError("the MCP servers are no longer running")
            return asyncio.run_coroutine_threadsafe(coroutine, self._loop)

    def _call(self, server: _Server, tool: str, args: dict, stop: Callable[[], bool]) -> SkillResult:
        """Call `tool` on `server` and wait for it, polling `stop()` and the deadline; a stop or a timeout cancels
        the call. A tool's own failure, a protocol error and a timeout are spoken `SkillError`s."""
        from mcp import MCPError, types

        label = f"the {tool} tool from {server.config.name}"
        # The detail is the result alone: the tool event already keeps the call's name and arguments, and its output
        # is capped (800 characters, 300 in the next prompt), so arguments in front would push the result out
        if stop():  # stopped between the approval and here: nothing is sent
            return SkillResult(f"{label} was stopped")
        future = self._submit(server.client.call_tool(tool, args, read_timeout_seconds=self.timeout))
        deadline = time.monotonic() + self.timeout + CALL_GRACE
        while not future.done():
            concurrent.futures.wait([future], timeout=POLL_SECONDS)
            if future.done():
                break
            if stop():
                future.cancel()
                return SkillResult(f"{label} was stopped")
            if time.monotonic() >= deadline:
                future.cancel()
                raise SkillError(f"{label} did not answer within {self.timeout:g} seconds")
        if future.cancelled():  # `close()` ended the servers under it
            raise SkillError(f"{label} was cut off because vaice is closing")
        try:
            result = future.result()
        except MCPError as exc:
            if exc.code == types.REQUEST_TIMEOUT:
                raise SkillError(f"{label} did not answer within {self.timeout:g} seconds") from None
            raise SkillError(f"{label} failed: {_one_line(exc.message, ERROR_CHARS)}") from None
        text = _result_text(result.content)
        if result.is_error:
            raise SkillError(_one_line(text, ERROR_CHARS) or f"{label} reported an error")
        if not text.strip():
            return SkillResult(NOTHING)
        speech = text if len(text) <= MCP_OUTPUT_CHARS else f"{text[:MCP_OUTPUT_CHARS]} {CUT}"
        return SkillResult(speech, text[:DETAIL_CHARS])

    # -- closing

    async def _stop_all(self) -> None:
        for server in self._servers:
            server.stop.set()
        await asyncio.gather(*(server.task for server in self._servers if server.task), return_exceptions=True)

    def close(self) -> None:
        """End every server (each client leaves its context, which closes the child's stdin and, failing that,
        kills it), stop the loop and join its thread, all bounded. Idempotent."""
        atexit.unregister(self.close)
        with self._lock:
            if self._closed:
                return
            self._closed = True
        loop, thread = self._loop, self._thread
        if loop is None or thread is None:
            return
        if thread.is_alive():
            try:
                asyncio.run_coroutine_threadsafe(self._stop_all(), loop).result(timeout=CLOSE_WAIT)
            except concurrent.futures.TimeoutError:
                log.warning("mcp servers did not stop within %g seconds; leaving them", CLOSE_WAIT)
            loop.call_soon_threadsafe(loop.stop)
            thread.join(timeout=CLOSE_WAIT)
        if not thread.is_alive():
            loop.close()


def start(path: Path, timeout: float, auto_seconds: float = 0.0) -> McpTools:
    """Start the servers in the config file at `path` (a server that fails to start is a warning and is skipped) and
    return them with their skills (`auto_seconds`: see `McpTools`). Exits when the `mcp` package is missing or the
    file is unusable."""
    try:
        from mcp import Client, MCPError, StdioServerParameters, stdio_client  # noqa: F401 - checks the v2 SDK is there
    except ImportError:
        raise SystemExit(INSTALL_HINT) from None
    configs = load_config(path)
    tools = McpTools(timeout, auto_seconds)
    try:
        tools._start(configs)
    except BaseException:  # a clash, Ctrl+C: the servers started so far must not outlive vaice
        tools.close()
        raise
    return tools
