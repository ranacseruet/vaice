"""The pure pieces of `browse` (assistant mode, `VAICE_BROWSE`): the skill the brain sees, what is said while the
browser works, how a step that needs a yes is put into words, and how the answer becomes a tool result. The task
itself (a thread, the outbox, the confirmations) lives in `responders/assistant.py`; the agent in `browse.py`.

`browse(goal)` is a `background` skill: the reply says `WORKING` and ends, the browser agent works on a thread of
its own, and when it has finished its answer goes back to the brain as the tool result, which phrases the outcome.
"""
import json
import re
import time
from collections.abc import Callable
from urllib.parse import urlsplit

from vaice.confirm import NO, YES
from vaice.skills import Skill, SkillError, SkillResult

WORKING = "Working in the browser."  # said when the task starts: no answer words (speaker echo, see confirm.py)
FAILED = "Sorry, the browser task failed."  # for an error nothing else explained
GOAL_MAX_CHARS = 500
PROGRESS_LINE_SECONDS = 8.0  # at most one spoken progress line in this time
RESULT_CHARS = 2000  # characters of the browser agent's answer the brain gets
CUT = "(more on screen)"

ASK_STEP = "Shall I {action}{where}? The details are on the screen."
# what each tool that changes a page is called aloud (the gate asks about exactly these, see `browse.GATE_POLICY`);
# none of these words is one the user answers with, which `tests/test_browse_task.py` checks per word
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
# every word of every yes and no phrase, as `confirm` compares them ("don't" is "dont")
_ANSWER_WORDS = frozenset(word for phrase in YES | NO for word in re.sub(r"[^\w\s]", "", phrase.casefold()).split())
_MARKDOWN = re.compile(r"\*\*|__|`")


def browse_skill() -> Skill:
    """The definition the brain sees. It is never `run`: `Skills.prepare` validates the call and the responder starts
    the task (see `responders/assistant.py`)."""
    return Skill(
        "browse",
        "Do a task on the web in a browser window, such as opening a website and reading something from it. It "
        "works in the background and takes a while. Not for questions you can answer yourself or about this computer.",
        {"type": "object",
         "properties": {"goal": {"type": "string",
                                 "description": "What to do in the browser, in one sentence, naming the site or "
                                                "address the user gave."}},
         "required": ["goal"]},
        lambda _args: SkillResult("the browser task runs in the background"),
        kind="background", ask=_start)


def _start(args: dict) -> str:
    goal = goal_of(args)
    if not goal:
        raise SkillError("no goal was given")
    return WORKING


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


def progress_phrase(tool: str, args: object) -> str | None:
    """What to say as a browser tool starts: only a page being opened (the host, from the URL); snapshots,
    screenshots and page lists are silent, and so is a step that needs a yes (its question speaks)."""
    if tool_name(tool) in ("navigate_page", "new_page"):
        host = host_of(args)
        if host:
            return f"Opening {host}."
    return None


def step_question(title: str, host: str | None) -> str:
    """The sentence that asks about a gated step: the action by its tool name, and the page's host when it is
    known and holds no word the user answers with (a host comes from a web page, so speaker echo could say yes to
    it). Never the arguments (the model's, and a page can steer them): the screen shows them."""
    action = STEP_ACTIONS.get(tool_name(title), OTHER_STEP)
    named = host is not None and not _ANSWER_WORDS & set(re.findall(r"[a-z0-9]+", host.casefold()))
    return ASK_STEP.format(action=action, where=f" on {host}" if named else "")


def result_of(answer: str) -> SkillResult:
    """The tool result for the brain: the agent's answer as one line without markdown emphasis (it writes
    `**Welcome to Python.org**`, which would be spoken), cut at `RESULT_CHARS`; the panel's detail keeps the answer."""
    text = " ".join(_MARKDOWN.sub("", answer).split())
    if len(text) > RESULT_CHARS:
        text = f"{text[:RESULT_CHARS].rstrip()} {CUT}"
    return SkillResult(f"the browser task finished: {text}", answer)


class ProgressLimiter:
    """At most one progress line per `seconds`; the first is allowed at once."""

    def __init__(self, seconds: float = PROGRESS_LINE_SECONDS, clock: Callable[[], float] | None = None):
        self._seconds = seconds
        self._clock = clock or time.monotonic
        self._last: float | None = None

    def allow(self) -> bool:
        now = self._clock()
        if self._last is not None and now - self._last < self._seconds:
            return False
        self._last = now
        return True
