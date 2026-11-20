"""The pure pieces of `browse` (`VAICE_BROWSE`): the skills the model sees, how a step that needs an approval is put
into words, what a running task reports about itself, and how the answer becomes a tool output. The task itself
(a thread, the outbox, the confirmations) lives in `assistant.py`; the agent in `browse.py`.

`browse(goal)` is a `background` skill: the call returns `WORKING` at once and the reply ends, the browser agent works
on a thread of its own and says nothing more than a heads-up that a step needs the Run button until it has finished;
its answer is then spoken through the outbox. `browser_status` and `stop_browsing` are what the user can ask
meanwhile: how it is going, and to end it.
"""
import json
import re
import threading
import time
from collections.abc import Callable
from urllib.parse import urlsplit

from vaice.skills import Skill, SkillError, SkillResult

WORKING = "Working in the browser."  # the result of the `browse` call: the model says it
FAILED = "Sorry, the browser task failed."  # for an error nothing else explained
BUSY = "A browser task is already running. Ask for an update, or ask to end the task first."
NO_TASK = "no browser task is running"
STOPPED = "stopped the browser task"
GOAL_MAX_CHARS = 500
STATUS_SPEECH_CHARS = 600  # what `browser_status` hands the model; its detail (the panel) keeps more
STATUS_DETAIL_CHARS = 2500
STATUS_TEXT_CHARS = 200  # of the agent's latest words in the speech line (the detail keeps `STATUS_DETAIL_TEXT_CHARS`)
STATUS_DETAIL_TEXT_CHARS = 800
STATUS_TOOLS_SHOWN = 5  # the latest tool names in the speech line
STATUS_TOOLS_KEPT = 200  # the status record keeps this many tool calls, the latest
RESULT_CHARS = 2000  # characters of the browser agent's answer the model gets
CUT = "(more on screen)"

ASK_STEP = "The browser task wants to {action}{where}. The details are on the screen."
# what each tool that changes a page is called (the gate asks about exactly these, see `browse.GATE_POLICY`)
STEP_ACTIONS = {
    "click": "click something",
    "fill": "fill in a field",
    "fill_form": "fill in a form",
    "type_text": "type some text",
    "press_key": "press a key",
    "evaluate_script": "run a script",
    "upload_file": "upload a file",
    "drag": "drag something",
    "handle_dialog": "answer a dialog",
}
OTHER_STEP = "take a step"
_MARKDOWN = re.compile(r"\*\*|__|`")


def browse_skill(busy: Callable[[], bool] | None = None) -> Skill:
    """The definition the model sees. It is never `run`: `Skills.prepare` validates the call and the responder starts
    the task (see `assistant.py`). With `busy`, a call while it is true is refused with `BUSY` as its
    result."""

    def start(args: dict) -> str:
        goal = goal_of(args)
        if not goal:
            raise SkillError("no goal was given")
        if busy is not None and busy():
            raise SkillError(BUSY)
        return WORKING

    return Skill(
        "browse",
        "Do a task on the web in a browser window: look something up, research a product, find recent news or "
        "prices, or open a website and read something from it. It works in the background and takes a while, one "
        "task at a time. Not for questions you can answer yourself or about this computer.",
        {"type": "object",
         "properties": {"goal": {"type": "string",
                                 "description": "What to do in the browser, in one sentence, naming the site or "
                                                "address if the user gave one."}},
         "required": ["goal"]},
        lambda _args: SkillResult("the browser task runs in the background"),
        kind="background", ask=start)


def browse_skills(*, busy: Callable[[], bool] | None, status: Callable[[], SkillResult],
                  stop: Callable[[], SkillResult]) -> list[Skill]:
    """`browse` and the two skills that go with it, offered together: `browser_status` (what the running task is
    doing; `status` answers) and `stop_browsing` (`stop` ends it, the same way the Stop button does)."""
    nothing = {"type": "object", "properties": {}}
    return [
        browse_skill(busy),
        Skill("browser_status",
              "Report how the running browser task is going: what it is doing, the pages it has visited and whether "
              "it waits for approval. Use it when the user asks how the browser task is going, what it is doing or "
              "for an update on it. Not for other questions.", nothing, lambda _args: status()),
        Skill("stop_browsing", "Stop the running browser task. Use it when the user asks to stop, cancel or end the "
                               "browser task.", nothing, lambda _args: stop(), kind="action"),
    ]


def goal_of(args: dict | str) -> str:
    """The goal in a call's arguments (the parsed object, or the model's JSON string): one line, at most
    `GOAL_MAX_CHARS` characters; empty when there is none."""
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except ValueError:
            return ""
    goal = args.get("goal") if isinstance(args, dict) else None
    return " ".join(goal.split())[:GOAL_MAX_CHARS] if isinstance(goal, str) else ""


def tool_name(title: str) -> str:
    """A browser tool's own name: Pi names the MCP server's tools `mcp__chrome__click`."""
    return title.rsplit("__", 1)[-1]


def host_of(args: object) -> str | None:
    """The host of the `url` argument of a navigation, without `www.` (which the speech cleaner turns into "a link"),
    or None."""
    url = args.get("url") if isinstance(args, dict) else None
    if not isinstance(url, str):
        return None
    try:
        parts = urlsplit(url)
        host = parts.hostname
    except ValueError:
        return None
    if parts.scheme not in ("http", "https") or not host:
        return None
    return host.removeprefix("www.")


def step_question(title: str, host: str | None) -> str:
    """The sentence that says what a gated step is: the action by its tool name, and the page's host when it is
    known. Never the arguments (the model's, and a page can steer them): the screen shows them."""
    action = STEP_ACTIONS.get(tool_name(title), OTHER_STEP)
    return ASK_STEP.format(action=action, where=f" on {host}" if host else "")


def result_of(answer: str) -> SkillResult:
    """The tool output for the model: the agent's answer as one line without markdown emphasis (it writes
    `**Welcome to Python.org**`, which would be spoken), cut at `RESULT_CHARS`; the panel's detail keeps the answer."""
    text = " ".join(_MARKDOWN.sub("", answer).split())
    if len(text) > RESULT_CHARS:
        text = f"{text[:RESULT_CHARS].rstrip()} {CUT}"
    return SkillResult(f"the browser task finished: {text}", answer)


def _one_line(text: str, limit: int) -> str:
    """`text` as one printable line without markdown emphasis, cut at `limit` characters (`...` marks the cut)."""
    clean = " ".join(_MARKDOWN.sub("", "".join(c if c.isprintable() or c.isspace() else " " for c in text)).split())
    return clean if len(clean) <= limit else f"{clean[: limit - 3].rstrip()}..."


class BrowseStatus:
    """What a running browser task has reported about itself, for `browser_status`: the goal, how long it has run,
    the tool calls so far (the latest `STATUS_TOOLS_KEPT`) with the pages they opened, the agent's latest words and
    the step that waits for an approval. Written on the task's thread (the agent's callbacks), read on the reply's, so
    every access is under one lock and `result()` builds from a snapshot. What the agent wrote is page-steerable text:
    it is cleaned to one line and capped, and it only ever reaches the model as a tool output."""

    def __init__(self, goal: str, clock: Callable[[], float] = time.monotonic):
        self._goal = goal
        self._clock = clock
        self._started = clock()
        self._lock = threading.Lock()
        self._tools: list[tuple[str, str | None]] = []  # (tool name, the url it opened, if any), oldest first
        self._text = ""
        self._asking: str | None = None
        self._ended = False

    @property
    def ended(self) -> bool:
        with self._lock:
            return self._ended

    def end(self) -> None:
        """The agent is done (an answer, a failure, a stop): the task no longer runs."""
        with self._lock:
            self._ended = True
            self._asking = None

    def tool(self, title: str, args: object) -> None:
        url = args.get("url") if isinstance(args, dict) else None
        with self._lock:
            self._tools.append((tool_name(title), _one_line(url, 200) if isinstance(url, str) and url else None))
            del self._tools[:-STATUS_TOOLS_KEPT]

    def text(self, text: str) -> None:
        """The agent's latest words between tools (the last one wins)."""
        with self._lock:
            self._text = _one_line(text, STATUS_DETAIL_TEXT_CHARS)

    def asking(self, title: str | None) -> None:
        """A step waits for an approval (`title`: the gated tool), or no longer does (None)."""
        with self._lock:
            self._asking = None if title is None else STEP_ACTIONS.get(tool_name(title), OTHER_STEP)

    def result(self) -> SkillResult:
        """The report: one capped line for the model, a fuller one for the panel."""
        with self._lock:
            tools, text, asking, ended = list(self._tools), self._text, self._asking, self._ended
        if ended:
            return SkillResult(NO_TASK)
        seconds = int(self._clock() - self._started)
        urls = [url for _, url in tools if url]
        visits = [h for h in (host_of({"url": url}) for url in urls) if h]
        hosts = list(dict.fromkeys(visits))  # each page once, in order of first visit
        parts = [f"the browser task is running: {_one_line(self._goal, 160)}", f"it has run for {seconds} seconds"]
        if tools:
            latest = ", ".join(name for name, _ in tools[-STATUS_TOOLS_SHOWN:])
            parts.append(f"{len(tools)} tool calls so far, the latest: {latest}")
        else:
            parts.append("no tool calls yet")
        if visits:
            parts.append(f"pages visited: {', '.join(hosts[-3:])}; now on {visits[-1]}")
        if asking:
            parts.append(f"a step is waiting for approval: {asking}")
        if text:
            parts.append(f"the agent's latest words: {_one_line(text, STATUS_TEXT_CHARS)}")
        speech = _one_line("; ".join(parts), STATUS_SPEECH_CHARS)
        lines = [f"goal: {self._goal}", f"running for {seconds} seconds", f"{len(tools)} tool calls"]
        if asking:
            lines.append(f"waiting for approval: {asking}")
        if text:
            lines.append(f"latest words: {text}")
        lines += [f"  {n}. {name}{f' {url}' if url else ''}" for n, (name, url) in enumerate(tools, 1)]
        return SkillResult(speech, "\n".join(lines)[:STATUS_DETAIL_CHARS])
