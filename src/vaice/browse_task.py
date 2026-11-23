"""The pure pieces of `browse` (`VAICE_BROWSE`): the skills the model sees, how a step that needs an approval is put
into words, what a running task reports about itself, and how the answer becomes a tool output. The task itself
(a thread, the outbox, the confirmations) lives in `assistant.py`; the agent in `browse.py`.

`browse(goal)` is a `background` skill: the call returns `WORKING` at once and the reply ends, the browser agent works
on a thread of its own and says nothing more than a heads-up that a step needs the Run button until it has finished;
its answer is then spoken through the outbox. `browser_status` and `stop_browsing` are what the user can ask: how the
running task is going (or, once it has ended, how the last one ended and what it found, from memory: `BrowseStatus`
keeps the outcome), and to end it.
"""
import json
import re
import threading
import time
import unicodedata
from collections.abc import Callable
from urllib.parse import urlsplit

from vaice.memory import clean_line, strip_markers
from vaice.skills import Skill, SkillError, SkillResult

WORKING = "Working in the browser."  # the result of the `browse` call: the model is told, and the reply ends
# What vaice says when the task started (`Assistant`, a fixed sentence through `SAY`): the model's reply ended with the
# call, so nothing else announces it. The outcome follows it.
STARTED = "I'm working on it in the browser. I'll tell you when it's done."
FAILED = "Sorry, the browser task failed."  # for an error nothing else explained
BUSY = "A browser task is already running. Ask for an update, or ask to end the task first."
NO_TASK = "no browser task is running"
STOPPED = "stopped the browser task"
GOAL_MAX_CHARS = 500
STATUS_SPEECH_CHARS = 600  # what `browser_status` hands the model; its detail (the panel) keeps more
STATUS_DETAIL_CHARS = 2500
LAST_SPEECH_CHARS = 700  # the speech line of the last ended task: under the realtime layer's tool output cap (800)
LAST_GOAL_CHARS = 120  # of the goal in it
STATUS_TEXT_CHARS = 200  # of the agent's latest words in the speech line (the detail keeps `STATUS_DETAIL_TEXT_CHARS`)
STATUS_DETAIL_TEXT_CHARS = 800
STATUS_TOOLS_SHOWN = 5  # the latest tool names in the speech line
STATUS_TOOLS_KEPT = 200  # the status record keeps this many tool calls, the latest
RESULT_CHARS = 2000  # characters of the answer that are spoken when it has no `Summary:` line
SUMMARY_CHARS = 400  # ... and of a `Summary:` line
CUT = "(more on screen)"
# The system-block line for the last finished task (`last_result_line`), so that "What did you find?" is answered
# from it and does not start a new task (#136). Imperative like `realtime._clock_line` (a declarative line was
# ignored there) and with no clause that forbids anything (those broke the audio, #77).
LAST_RESULT = ('You know what the last browser task found. It was asked: "{goal}". It finished {ago} and found: '
               '{summary} Use this when asked what the browser task found.')
NARRATION_CHARS = 400  # a rule or a heading this early ends the agent's narration ("I now have ... Let me compile")

# how a task ended, for `BrowseStatus.end` (the last ended task is reported by it)
FINISHED = "finished"
STOPPED_KIND = "stopped"
DECLINED = "declined"
FAILED_KIND = "failed"
_ENDED_AS = {FINISHED: "finished", STOPPED_KIND: "was stopped", DECLINED: "was declined at a step",
             FAILED_KIND: "failed"}
STATUS_DESCRIPTION = (
    "Report on the browser task: how the running one is going (what it is doing, the pages it has visited, whether "
    "it waits for approval), or what the last one found once it has finished. Use it when the user asks how the "
    "browser task is going, for an update, or what it found. Not for other questions.")

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
# the first line of the answer the prompt asks for (`browse.PROMPT`), with markdown emphasis or a heading mark around it
_SUMMARY_LINE = re.compile(r"^[ \t>#]*[*_]{0,3}[ \t]*summary[ \t]*[*_]{0,3}[ \t]*:[ \t]*[*_]{0,3}[ \t]*(.*)$",
                           re.IGNORECASE | re.MULTILINE)
_RULE_OR_HEADING = re.compile(r"^(?:[ \t]{0,3}([-*_])(?:[ \t]*\1){2,}[ \t]*|[ \t]{0,3}#{1,6}(?:[ \t].*)?)$",
                              re.MULTILINE)
_RULE_LINE = re.compile(r"^[ \t]{0,3}([-*_])(?:[ \t]*\1){2,}[ \t]*$|^[ \t|:-]*$")  # a rule, or a table's separator row
_LINE_MARKS = re.compile(r"^[ \t]*(?:#{1,6}[ \t]*|>[ \t]*|[-*+\u2022][ \t]+|\d+[.)][ \t]+)+")
_LINK = re.compile(r"!?\[([^\]]*)\]\([^)\s]*(?:\s+\"[^\"]*\")?\)")
_EMPHASIS = re.compile(r"[*`~]+|(?<![A-Za-z0-9])_+|_+(?![A-Za-z0-9])")
_KEPT_SYMBOLS = "\u00b0"  # the degree sign is a "symbol, other" like an emoji, but it is spoken
# symbols other and modifier (emoji, skin tones), format, private use and unassigned characters
_PICTOGRAPH_CATEGORIES = ("So", "Sk", "Cf", "Co", "Cn")
_MARKUP_JOINERS = "\u200d\ufe0e\ufe0f"  # zero-width joiner and variation selectors: what makes an emoji one glyph


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
    doing, or how the last one ended and what it found; `status` answers) and `stop_browsing` (`stop` ends it, the same
    way the Stop button does)."""
    nothing = {"type": "object", "properties": {}}
    return [
        browse_skill(busy),
        Skill("browser_status", STATUS_DESCRIPTION, nothing, lambda _args: status()),
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


def _is_spoken(char: str) -> bool:
    """Whether a voice reads `char`: letters, digits, punctuation, currency signs and spaces, not emoji or other
    pictographs."""
    if char in _KEPT_SYMBOLS:
        return True
    if char in _MARKUP_JOINERS or unicodedata.category(char) in _PICTOGRAPH_CATEGORIES:
        return False
    return char.isprintable() or char.isspace()


def _speakable(text: str) -> str:
    """`text` as one line a voice can read: no headings, rules, quote, bullet or numbering marks, table pipes,
    emphasis, link targets, emoji or pictographs (letters, digits, punctuation and currency signs stay)."""
    lines = []
    for line in text.splitlines():
        if not _RULE_LINE.match(line):
            line = _EMPHASIS.sub("", _LINK.sub(r"\1", _LINE_MARKS.sub("", line))).replace("|", " ")
            lines.append("".join(c for c in line if _is_spoken(c)))
    return " ".join(" ".join(lines).split())


def _cut(text: str, limit: int) -> str:
    return text if len(text) <= limit else f"{text[:limit].rstrip()} {CUT}"


def _split_summary(answer: str) -> tuple[str, str]:
    """The speakable text of the answer's `Summary:` line (the next line when it stands alone; empty when there is
    none) and the answer without that line."""
    match = _SUMMARY_LINE.search(answer)
    if match is None:
        return "", answer
    after = answer[match.end(1):].splitlines(keepends=True)
    summary = _speakable(match.group(1))
    if summary:
        return summary, answer[:match.start()] + "".join(after[1:])
    following = next((n for n, line in enumerate(after) if line.strip()), None)  # `Summary:` alone: the next line
    if following is None:
        return "", answer
    summary = _speakable(after[following])
    return summary, (answer[:match.start()] + "".join(after[following + 1:]) if summary else answer)


def _report_body(answer: str) -> str:
    """The answer from its first horizontal rule or heading when that is within `NARRATION_CHARS` (every recorded Pi
    answer is its own narration, "I now have ... Let me compile ...", then a rule, then the report: the narration is
    not the finding), else all of it."""
    start = _RULE_OR_HEADING.search(answer)
    return answer[start.start():] if start is not None and start.start() < NARRATION_CHARS else answer


def spoken_summary(answer: str) -> str:
    """What of the agent's answer is spoken: the text of its `Summary:` line (`browse.PROMPT` asks for one; markdown
    emphasis around the word, any case, anywhere in the answer) cut at `SUMMARY_CHARS`. Without one, the answer from
    its first horizontal rule or heading when that is within `NARRATION_CHARS` (see `_report_body`), cut at
    `RESULT_CHARS`. Either way as one speakable line (`_speakable`). Deterministic: no model involved."""
    summary, _ = _split_summary(answer)
    if summary:
        return _cut(summary, SUMMARY_CHARS)
    return _cut(_speakable(_report_body(answer)), RESULT_CHARS)


def _ago(seconds: float) -> str:
    """"N seconds ago", or "N minutes ago" from a minute on (rounded)."""
    if seconds < 60:
        count, unit = int(seconds), "second"
    else:
        count, unit = round(seconds / 60), "minute"
    return f"{count} {unit}{'' if count == 1 else 's'} ago"


def _fit(text: str, limit: int) -> str:
    """`text` cut so that it, with `CUT` after it when it was cut, is at most `limit` characters."""
    if len(text) <= limit:
        return text
    return f"{text[:max(0, limit - len(CUT) - 1)].rstrip()} {CUT}"


def result_of(answer: str) -> SkillResult:
    """The outcome as `_browse` uses it: `speech` is the spoken summary of the answer (`spoken_summary`), the panel's
    `detail` keeps the whole answer."""
    return SkillResult(f"the browser task finished: {spoken_summary(answer)}", answer)


def _one_line(text: str, limit: int) -> str:
    """`text` as one printable line without markdown emphasis, cut at `limit` characters (`...` marks the cut)."""
    clean = " ".join(_MARKDOWN.sub("", "".join(c if c.isprintable() or c.isspace() else " " for c in text)).split())
    return clean if len(clean) <= limit else f"{clean[: limit - 3].rstrip()}..."


class BrowseStatus:
    """What a browser task has reported about itself, for `browser_status`: while it runs, the goal, how long it has
    run, the tool calls so far (the latest `STATUS_TOOLS_KEPT`) with the pages they opened, the agent's latest words and
    the step that waits for an approval; once it has ended (`end`), how it ended and, when it finished, its answer.
    Written on the task's thread (the agent's callbacks), read on the reply's, so every access is under one lock and
    `result()` builds from a snapshot. What the agent wrote is page-steerable text: it is cleaned to one line and
    capped, and it only ever reaches the model as a tool output."""

    def __init__(self, goal: str, clock: Callable[[], float] = time.monotonic):
        self._goal = goal
        self._clock = clock
        self._started = clock()
        self._lock = threading.Lock()
        self._tools: list[tuple[str, str | None]] = []  # (tool name, the url it opened, if any), oldest first
        self._text = ""
        self._asking: str | None = None
        self._ended = False
        self._outcome = STOPPED_KIND  # how it ended (`end`), the answer of a finished one, and when
        self._answer = ""
        self._ended_at = self._started

    @property
    def ended(self) -> bool:
        with self._lock:
            return self._ended

    def end(self, outcome: str, answer: str = "") -> None:
        """The agent is done: the task no longer runs. `outcome` is how it ended (`FINISHED` with its `answer`,
        `STOPPED_KIND`, `DECLINED` or `FAILED_KIND`); `result()` reports it from then on."""
        with self._lock:
            self._ended = True
            self._asking = None
            self._outcome, self._answer = outcome, answer if outcome == FINISHED else ""
            self._ended_at = self._clock()

    def finding(self) -> tuple[str, str, int] | None:
        """(goal, answer, seconds since it ended) of a task that ended by finishing, else None (running, stopped,
        declined or failed). One look under the lock."""
        with self._lock:
            if not self._ended or self._outcome != FINISHED:
                return None
            goal, answer, ended_at = self._goal, self._answer, self._ended_at
        return goal, answer, int(self._clock() - ended_at)

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

    def view(self) -> dict:
        """A plain snapshot for the UI's status line (not text for the model; `result()` is that): the goal, the
        seconds run (up to the end, once ended), the tool calls so far, the latest host a tool opened (None when
        none did), whether a step waits for an approval (`asking`) and whether the task `ended`. The goal and the host
        are untrusted text: the UI must escape them."""
        with self._lock:
            tools, asking, ended, ended_at = list(self._tools), self._asking, self._ended, self._ended_at
        hosts = [h for h in (host_of({"url": url}) for _, url in tools if url) if h]
        seconds = int((ended_at if ended else self._clock()) - self._started)
        return {"goal": self._goal, "seconds": seconds, "steps": len(tools), "host": hosts[-1] if hosts else None,
                "asking": asking is not None, "ended": ended}

    def result(self) -> SkillResult:
        """The report: one capped line for the model, a fuller one for the panel. For a task that has ended, how it
        ended and how long ago, and the finding of a finished one (`_last_result`)."""
        with self._lock:
            tools, text, asking, ended = list(self._tools), self._text, self._asking, self._ended
            outcome, answer, ended_at = self._outcome, self._answer, self._ended_at
        if ended:
            return self._last_result(outcome, answer, int(self._clock() - ended_at))
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

    def _last_result(self, outcome: str, answer: str, seconds: int) -> SkillResult:
        """The report of an ended task: its goal, how long ago it ended and how. A finished one adds its spoken summary
        and then more of the answer, one speakable line cut to `LAST_SPEECH_CHARS` as a whole (the summary comes first,
        so a cut takes the rest); the panel's detail is the goal and the whole answer, capped. A stop, a refusal and
        a failure are one sentence each: the reason is in the panel."""
        head = f"the last browser task, {_one_line(self._goal, LAST_GOAL_CHARS)}, {_ENDED_AS[outcome]} {_ago(seconds)}"
        if outcome != FINISHED:
            return SkillResult(head)
        _, rest = _split_summary(answer)
        more = _speakable(_report_body(rest)) if rest != answer else ""  # without a `Summary:` line the summary is it
        text = " ".join(part for part in (spoken_summary(answer), more) if part)
        speech = _fit(f"{head}: {text}" if text else head, LAST_SPEECH_CHARS)
        return SkillResult(speech, f"goal: {self._goal}\nended {_ago(seconds)}\n\n{answer}"[:STATUS_DETAIL_CHARS])


def last_result_line(status: BrowseStatus | None) -> str:
    """The `LAST_RESULT` line for the system block of a reply, built per reply so that the "ago" stays current;
    empty (nothing is added to the prompt) unless `status` is a task that finished with something to say. The goal
    and the summary are page-steerable text: each is cleaned to one line (`memory.clean_line`; the summary is
    `spoken_summary`) and capped (`LAST_GOAL_CHARS`, `SUMMARY_CHARS`, the cut mark included). The rest of the answer
    never goes in."""
    found = status.finding() if status is not None else None
    if found is None:
        return ""
    goal, answer, seconds = found
    goal = _fit(clean_line(goal).replace('"', "'"), LAST_GOAL_CHARS)  # it stands in quotes
    # markers first, on the answer as written: `_speakable` would turn `<|im_end|>` into `< im_end >`
    summary = _fit(clean_line(spoken_summary(strip_markers(answer))), SUMMARY_CHARS)
    return LAST_RESULT.format(goal=goal, ago=_ago(seconds), summary=summary) if goal and summary else ""
