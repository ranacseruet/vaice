"""Live eval: a browser task as the user experiences it, from the request to the follow-up question.

Run by hand against a real llama-server with an audio model (see evals/README.md); not imported by the app, not run
in CI. `run.py --browse` scores only whether `browse` is called and `browse_live.py` only the browser agent; this
scores what the user gets around them: the reply that starts the task, the spoken outcome, what the chat shows and what
a follow-up question does. The browser agent is a fake that replays real recorded answers (`browse_answers.json`), so
it measures vaice and the voice model, not the agent, and uses no browser, Pi or network.

The reply that starts a task ends with the call (it has no audio of its own) and vaice says a fixed sentence afterwards:
the run waits for it, as a listener would, before the follow-up. `--outcome report|say` is the wording the outcome is
spoken with (`REPORT`, the model's words, or `SAY`, as it is).

Every run is the app's own wiring: an `Assistant` with a browser agent (so `_start_browse`, `_browse`, the real `busy`
check, `browser_status` and `stop_browsing`, and the outcome speech are the app's code) over the stub skills of
`run.py`, a `Conversation` whose turns and history are made as `app.make_handler` makes them, and an `Outbox` drained
as fastrtc's idle `emit()` drains it. No audio is played: frames are counted.
"""
import argparse
import contextlib
import json
import platform
import re
import statistics
import sys
import tempfile
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import run as evals_run

from vaice import assistant as vaice_assistant
from vaice import browse_task, config
from vaice.assistant import Assistant
from vaice.browse import BrowseStopped
from vaice.conversation import Conversation, Turn
from vaice.interruptible import IDLE_WAIT_SECONDS, LEAD_SECONDS
from vaice.outbox import Marker, Outbox, run_hook

HERE = Path(__file__).parent
TASK_SECONDS = 8.0  # how long the fake task takes (its tool calls are spread over it)
POLL = 0.05
OUTCOME_WAIT = 60.0  # seconds to wait for the outcome to be spoken and recorded
GIVE_UP_AFTER = 1.0  # seconds with no task thread, nothing queued and no outcome: none is coming
PLAYBACK_WAIT = 30.0  # the user listens to what was said before asking the next question
SETTLE_SECONDS = 2.0  # after the follow-up reply: speech still in flight may land
JOIN_SECONDS = 30.0  # for a task thread to end once stopped
TRANSCRIPT_WAIT = 5.0  # as in run.py and app.make_handler: the transcript is reported by a parallel session
BROWSE_THREAD = "vaice-assistant-browse"  # the name `Assistant._start_browse` gives the task's thread
SPEECH_THREAD = "vaice-assistant-speech"  # the name `Assistant` gives the threads that speak (the announcement)
OUTCOME_WORDINGS = {"report": "REPORT", "say": "SAY"}  # --outcome: the name in `vaice.assistant` of what it speaks with
INVENTED_CHARS = 120  # a start reply longer than this has said more than "I'll look into it"
MIN_REPORT_LINE = 20  # characters of a report line that counts as "the report itself" in the chat
# What a spoken outcome must not sound like: the agent's own narration instead of the finding
NARRATION = re.compile(r"compil|let me|i now have|i have gathered|final answer", re.IGNORECASE)
# What the fake agent does while it works: tool names as Pi's MCP server shows them, over a search page and a site
STEPS = (
    ("mcp__chrome__navigate_page", {"url": "https://duckduckgo.com/?q=best+reviews"}),
    ("mcp__chrome__take_snapshot", {}),
    ("mcp__chrome__navigate_page", {"url": "https://www.example.com/reviews"}),
    ("mcp__chrome__take_snapshot", {}),
)
FLAGS = ("start_browse", "start_invented", "outcome_spoken", "outcome_named", "outcome_narration", "chat_report",
         "followup_rebrowse", "followup_status", "followup_named")
NUMBERS = ("start_extra_chars", "start_first_audio_s", "outcome_first_audio_s", "followup_first_audio_s")
HEADS = {"start_browse": "browse", "start_invented": "invent", "outcome_spoken": "spoken", "outcome_named": "named",
         "outcome_narration": "narrat", "chat_report": "chat", "followup_rebrowse": "rebrw",
         "followup_status": "status", "followup_named": "f-named"}


def load_cases() -> tuple[list[dict], dict[str, dict]]:
    """The cases, and the fixtures (the recorded agent answers) by id."""
    fixtures = {fixture["id"]: fixture for fixture in evals_run.load_json("browse_answers.json")}
    return evals_run.load_json("browse_flow_cases.json"), fixtures


# -- the fake browser agent


class FakeBrowseAgent:
    """A `BrowseAgent` that replays a recorded answer: it reports a few tools over `seconds`, honours `stop()` like the
    real one (`BrowseStopped`) and returns `answer`. `release` holds the task open after its tools until it is set (the
    harness keeps a task running while a follow-up is asked). Counts its runs (`goals`) and notes when it finished."""

    def __init__(self, answer: str, seconds: float = TASK_SECONDS, release: threading.Event | None = None):
        self.answer, self.seconds = answer, seconds
        self.release = release or threading.Event()
        if release is None:
            self.release.set()
        self.goals: list[str] = []
        self.stopped = 0
        self.closed = 0
        self.finished_at: float | None = None  # time.monotonic() when the answer was returned
        self._lock = threading.Lock()

    @property
    def runs(self) -> int:
        with self._lock:
            return len(self.goals)

    def run(self, goal, *, stop, on_progress, on_ask, deadline_s, on_text=None) -> str:
        with self._lock:
            self.goals.append(goal)
        started = time.monotonic()
        try:
            for index, (tool, arguments) in enumerate(STEPS, 1):
                self._wait_until(started + self.seconds * index / (len(STEPS) + 1), stop)
                on_progress(tool, arguments)
            self._wait_until(started + self.seconds, stop)
            while not self.release.is_set():
                self._wait_until(time.monotonic() + POLL, stop)
        except BrowseStopped:
            with self._lock:
                self.stopped += 1
            raise
        with self._lock:
            self.finished_at = time.monotonic()
        return self.answer

    @staticmethod
    def _wait_until(moment: float, stop) -> None:
        while True:
            if stop():
                raise BrowseStopped
            remaining = moment - time.monotonic()
            if remaining <= 0:
                return
            time.sleep(min(POLL, remaining))

    def close(self) -> None:
        self.closed += 1


# -- the outbox drainer


class Drainer:
    """What fastrtc's idle `emit()` does with the outbox (`InterruptSafeReplyOnPause`), without playing anything: it
    touches the outbox on every call (so it is never stale), serves nothing while a reply is in flight (`replying`),
    and otherwise runs the markers it reaches and serves one frame, only while the audio served so far (reply audio
    counted by `note_playing`) ends less than `LEAD_SECONDS` ahead. Frames are only counted: `frames` holds (when
    served, seconds of audio)."""

    def __init__(self, outbox: Outbox, clock=time.monotonic):
        self._thread = threading.Thread(target=self._loop, name="eval-outbox-drainer", daemon=True)
        self._outbox, self._clock = outbox, clock
        self._lock = threading.Lock()
        self._until = 0.0  # when the audio served so far is estimated to have played
        self._halt = threading.Event()
        self.replying = threading.Event()
        self._frames: list[tuple[float, float]] = []

    @property
    def frames(self) -> list[tuple[float, float]]:
        with self._lock:
            return list(self._frames)

    def step(self) -> None:
        """One `emit()`."""
        self._outbox.touch()
        if self.replying.is_set():
            return
        while True:
            now = self._clock()
            with self._lock:
                allowed = self._until - now < LEAD_SECONDS
            item = self._outbox.poll(allowed)
            if item is None:
                return
            if isinstance(item, Marker):
                run_hook(item.callback, "marker")
                continue
            rate, samples = item
            seconds = np.size(samples) / rate
            with self._lock:
                self._until = max(self._until, now) + seconds
                self._frames.append((now, seconds))
            return

    def _loop(self) -> None:
        while not self._halt.is_set():
            self.step()
            self._halt.wait(IDLE_WAIT_SECONDS)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        """End the thread (when it was started) and wait for it."""
        self._halt.set()
        if self._thread.is_alive():
            self._thread.join(5)

    def is_alive(self) -> bool:
        return self._thread.is_alive()

    def note_playing(self, seconds: float) -> None:
        """Reply audio handed out plays before anything served after it."""
        with self._lock:
            self._until = max(self._until, self._clock()) + seconds

    def speech_start(self) -> None:
        """The user starts talking with no reply in flight: what was queued is dropped (`receive()`)."""
        self._outbox.clear()
        with self._lock:
            self._until = 0.0

    def playing_for(self) -> float:
        with self._lock:
            return max(0.0, self._until - self._clock())


# -- scoring (pure: a record and its fixture in, flags out)


def normalized(text: str) -> str:
    """`text` as one line without markdown marks, case ignored."""
    return " ".join(re.sub(r"[*_`#]", "", text).split()).casefold()


def names_finding(text: str, keywords: list[str]) -> bool:
    """Whether `text` names any of the fixture's keywords."""
    lowered = text.casefold()
    return any(keyword.casefold() in lowered for keyword in keywords)


def is_narration(text: str) -> bool:
    return NARRATION.search(text) is not None


def is_invented(text: str, keywords: list[str]) -> bool:
    """A start reply that said more than it knew (heuristic): longer than `INVENTED_CHARS`, or it names a finding
    that only the task's answer holds."""
    return len(text.strip()) > INVENTED_CHARS or names_finding(text, keywords)


def report_lines(fixture: dict) -> list[str]:
    """The lines of the report body (after the first `---`: before it is the agent's narration) that name a keyword,
    normalized: any of them in the chat is "the report itself". Empty when the answer has no such line."""
    _, separator, body = fixture["answer"].partition("\n---")
    lines = [normalized(line) for line in (body if separator else fixture["answer"]).splitlines()]
    return [line for line in lines if len(line) >= MIN_REPORT_LINE and names_finding(line, fixture["keywords"])]


def chat_has_report(ui: list[dict], fixture: dict) -> bool:
    """Whether the assistant's messages in the chat hold a whole line of the report (heuristic: a summary that names
    the products is not the report)."""
    chat = normalized("\n".join(m["content"] for m in ui if m.get("role") == "assistant"))
    return any(line in chat for line in report_lines(fixture))


def called(calls: list, name: str) -> bool:
    return any(call[0] == name for call in calls)


def score_run(record: dict, fixture: dict) -> dict:
    """The flags and numbers of one run (None where the run never got that far)."""
    keywords = fixture["keywords"]
    score: dict = dict.fromkeys((*FLAGS, *NUMBERS))
    start, outcome, follow = record.get("start"), record.get("outcome"), record.get("followup")
    if start is not None:
        text = start["text"].strip()
        score.update(start_browse=called(start["calls"], "browse"), start_extra_chars=len(text),
                     start_invented=is_invented(text, keywords), start_first_audio_s=start["first_audio_s"])
        score["outcome_spoken"] = outcome is not None
        score["outcome_named"] = outcome is not None and names_finding(outcome["text"], keywords)
        if outcome is not None:
            score.update(outcome_narration=is_narration(outcome["text"]),
                         outcome_first_audio_s=outcome["first_audio_s"])
        score["chat_report"] = chat_has_report(record["ui"], fixture)
    if follow is not None:
        score.update(followup_rebrowse=called(follow["calls"], "browse") or record["agent_runs"] > 1,
                     followup_status=called(follow["calls"], "browser_status"),
                     followup_named=names_finding(follow["text"], keywords),
                     followup_first_audio_s=follow["first_audio_s"])
    return score


def tally(scores: list[dict]) -> dict:
    """k and N per flag (over the runs where it is defined) and the median of each number."""
    flags = {}
    for name in FLAGS:
        known = [s[name] for s in scores if s[name] is not None]
        flags[name] = [sum(known), len(known)]
    medians = {}
    for name in NUMBERS:
        known = [s[name] for s in scores if s[name] is not None]
        medians[name] = statistics.median(known) if known else None
    return {"runs": len(scores), "flags": flags, "medians": medians}


# -- one run


class FlowRun:
    """One run's pieces: the fake agent, the assistant that drives it, a conversation, the outbox and its drainer."""

    def __init__(self, url: str, prompt: str, fixture: dict, *, vision: bool, task_seconds: float, hold: bool):
        self.fixture = fixture
        self.agent = FakeBrowseAgent(fixture["answer"], task_seconds, threading.Event() if hold else None)
        # `browse`, `browser_status` and `stop_browsing` over the assistant's own busy check, status and stop (what
        # `Assistant.__init__` builds when it makes its skills); every other skill is a stub
        control = browse_task.browse_skills(busy=lambda: self.assistant.browsing(),
                                            status=lambda: self.assistant._browser_status(),
                                            stop=lambda: self.assistant._stop_browsing())
        skills = evals_run.stub_skills(vision=vision, browse=True, browse_control=control)
        self.assistant = Assistant(url=url, prompt=prompt, skills=skills, browse=self.agent)
        self.outbox = Outbox()
        self.assistant.attach_outbox(self.outbox)
        self.drainer = Drainer(self.outbox)
        self.conversation = Conversation(path=None)
        self.turns: list[Turn] = []
        # the speech recorded onto a turn after its reply ended: the announcement of the task, then its outcome
        self.appended: list[dict] = []
        self.outcome_recorded = threading.Event()  # the first task's outcome was recorded on the first turn
        self.began = time.monotonic()
        self.drainer.start()

    def _watch(self, turn: Turn) -> None:
        """Note what `append_assistant_text` records onto `turn` (a marker the drainer served)."""
        index = self.turns.index(turn)
        original = turn.append_assistant_text

        def append(text: str) -> None:
            now = time.monotonic()
            self.appended.append({"turn": index, "t": round(now - self.began, 2), "text": text.strip()})
            original(text)
            if index == 0 and self._is_outcome_recorded(now):
                self.outcome_recorded.set()

        turn.append_assistant_text = append  # type: ignore[method-assign]

    def _is_outcome_recorded(self, now: float) -> bool:
        """Whether what was recorded onto the first turn by `now` includes the outcome. The announcement is always
        recorded first (the outcome waits for it), so a second entry is the outcome; a single one is it only when it
        came after the agent had finished (a task shorter than its own announcement is not what this measures)."""
        spoken = [a for a in self.appended if a["turn"] == 0]
        finished = self.agent.finished_at
        return len(spoken) >= 2 or (len(spoken) == 1 and finished is not None and now >= finished)

    def reply(self, audio) -> dict:
        """One reply as `app.make_handler` makes it: a turn, the history before it, the frames counted, the turn
        finished. Records the calls and the text as they stood when the reply ended."""
        started = time.perf_counter()
        record: dict = {"error": None}
        turn = self.conversation.start_turn()
        self.turns.append(turn)
        self._watch(turn)
        history = self.conversation.history(before=turn)
        self.drainer.speech_start()
        self.drainer.replying.set()
        first, seconds = None, 0.0
        try:
            for rate, frame in self.assistant.respond(audio, history, turn):
                if first is None:
                    first = time.perf_counter() - started
                length = np.size(frame) / rate
                seconds += length
                turn.audio_handed_out(length)
                self.drainer.note_playing(length)
            turn.finish()
        except Exception as exc:  # noqa: BLE001 - a failed run is recorded with its error, never aborts the eval
            record["error"] = f"{type(exc).__name__}: {exc}"
            turn.fail(record["error"])
        finally:
            self.drainer.replying.clear()
        record.update(text=turn.assistant_text, calls=[[e["name"], e["arguments"]] for e in list(turn.tool_events)],
                      first_audio_s=first, audio_s=seconds)
        turn.transcribed.wait(TRANSCRIPT_WAIT)  # keeps the transcription session out of the next reply
        record["transcript"] = turn.user_text
        return record

    def task_threads(self) -> list[threading.Thread]:
        return [t for t in threading.enumerate() if t.name == BROWSE_THREAD]

    def wait_for_outcome(self, timeout: float = OUTCOME_WAIT) -> bool:
        """Wait until the first task's outcome was spoken and recorded; False when none is coming (no task thread and
        nothing queued for `GIVE_UP_AFTER`) or `timeout` passed."""
        deadline = time.monotonic() + timeout
        idle_since = None
        while time.monotonic() < deadline:
            if self.outcome_recorded.is_set():
                return True
            if self.task_threads() or len(self.outbox):
                idle_since = None
            elif idle_since is None:
                idle_since = time.monotonic()
            elif time.monotonic() - idle_since >= GIVE_UP_AFTER:
                return False
            time.sleep(POLL)
        return False

    def wait_for_announcement(self) -> None:
        """The listener hears "I'm working on it" before saying anything more: wait until it was recorded onto the first
        turn, or nothing is coming (no task, no speech thread, nothing queued for `GIVE_UP_AFTER`), or `PLAYBACK_WAIT`
        passed."""
        deadline = time.monotonic() + PLAYBACK_WAIT
        idle_since = None
        while time.monotonic() < deadline:
            if any(a["turn"] == 0 for a in self.appended):
                return
            if self.task_threads() or self.speech_threads() or len(self.outbox):
                idle_since = None
            elif idle_since is None:
                idle_since = time.monotonic()
            elif time.monotonic() - idle_since >= GIVE_UP_AFTER:
                return
            time.sleep(POLL)

    def speech_threads(self) -> list[threading.Thread]:
        return [t for t in threading.enumerate() if t.name == SPEECH_THREAD]

    def wait_for_playback(self) -> None:
        """The user listens before asking the next question: what was said so far has (nearly) played."""
        deadline = time.monotonic() + PLAYBACK_WAIT
        while self.drainer.playing_for() > 0 and time.monotonic() < deadline:
            time.sleep(POLL)

    def outcome(self) -> dict | None:
        """What the outcome speech recorded on the asking turn, and how soon after the task ended it was heard."""
        spoken = [a for a in self.appended if a["turn"] == 0]
        finished = self.agent.finished_at
        if not spoken or finished is None:
            return None
        if len(spoken) == 1 and spoken[0]["t"] + 0.01 < finished - self.began:
            return None  # the one entry is the announcement: the outcome was never recorded
        outcome = spoken[-1]  # the announcement came first
        frames = [(when, seconds) for when, seconds in self.drainer.frames if when >= finished]
        return {"text": outcome["text"], "recorded_s": outcome["t"],
                "first_audio_s": frames[0][0] - finished if frames else None,
                "audio_s": sum(seconds for _, seconds in frames)}

    def finish(self) -> list[str]:
        """End the run: the task and what it was saying stop, every thread is joined. The problems found, if any."""
        problems = []
        self.agent.release.set()
        self.assistant.stop_task()
        for thread in self.task_threads():
            thread.join(JOIN_SECONDS)
            if thread.is_alive():
                problems.append("a browser task thread did not end")
        self.drainer.stop()
        if self.drainer.is_alive():
            problems.append("the outbox drainer did not end")
        return problems


def run_flow(url: str, prompt: str, case: dict, fixture: dict, audio, followup_audio, *, vision: bool = False,
             task_seconds: float = TASK_SECONDS) -> dict:
    """One run of a case: the request, the task, the follow-up. Never raises: a failure is recorded in `errors`."""
    follow = case["followup"]
    flow = FlowRun(url, prompt, fixture, vision=vision, task_seconds=task_seconds, hold=follow["when"] == "running")
    record: dict = {"errors": [], "start": None, "outcome": None, "followup": None}
    try:
        record["start"] = flow.reply(audio)
        if record["start"]["error"]:
            record["errors"].append(f"start: {record['start']['error']}")
        else:
            flow.wait_for_announcement()
            if follow["when"] == "finished":
                flow.wait_for_outcome()
            flow.wait_for_playback()
            record["followup"] = {"say": follow["say"], "when": follow["when"],
                                  "task_running": flow.agent.finished_at is None and bool(flow.task_threads()),
                                  **flow.reply(followup_audio)}
            if record["followup"]["error"]:
                record["errors"].append(f"followup: {record['followup']['error']}")
            flow.agent.release.set()
            if follow["when"] == "running":
                flow.wait_for_outcome()  # the held task ends now: its outcome follows the follow-up's answer
            time.sleep(SETTLE_SECONDS)
        record["outcome"] = flow.outcome()
        record["agent_runs"] = flow.agent.runs
        record["agent_goals"] = list(flow.agent.goals)
        record["ui"] = flow.conversation.messages_for_ui()
        record["turns"] = [turn.to_dict() for turn in flow.turns]
        record["appended"] = list(flow.appended)
    except Exception as exc:  # noqa: BLE001 - whatever broke the harness is the run's error, never aborts the eval
        record["errors"].append(f"{type(exc).__name__}: {exc}")
        record.setdefault("agent_runs", flow.agent.runs)
        record.setdefault("ui", [])
    finally:
        record["errors"].extend(flow.finish())
    record["score"] = score_run(record, fixture)
    return record


# -- the command line


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Eval a browser task end to end (request, outcome, chat, follow-up) "
                                                 "against a live llama-server, with a fake browser agent.")
    parser.add_argument("--url", default=config.LLAMACPP_URL, help="llama-server URL (default: VAICE_LLAMACPP_URL)")
    parser.add_argument("--repeats", type=int, default=3, help="runs per case (default: 3)")
    parser.add_argument("--cases", help="comma-separated case ids (default: all)")
    parser.add_argument("--task-seconds", type=float, default=TASK_SECONDS,
                        help=f"seconds the fake browser task takes (default: {TASK_SECONDS:g})")
    parser.add_argument("--vision", action="store_true",
                        help="also offer the look_at_screen skill (stubbed: a fixed fake screen)")
    parser.add_argument("--prompt", help="reply instructions (default: VAICE_PROMPT)")
    parser.add_argument("--outcome", choices=sorted(OUTCOME_WORDINGS), default="report",
                        help="how the task's outcome is spoken: report = in the model's words (the app's default), "
                             "say = as it is (default: report)")
    parser.add_argument("--label", default="", help="free text stored in the results, e.g. the quantization")
    parser.add_argument("--out", type=Path, help="results file (default: evals/results/browse-flow-<UTC timestamp>"
                                                 ".json)")
    args = parser.parse_args(argv)
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    if not args.task_seconds > 0:
        parser.error("--task-seconds must be > 0")
    return args


def cell(counts: list[int] | None) -> str:
    return "-" if not counts or not counts[1] else f"{counts[0]}/{counts[1]}"


def flag_mark(value: bool | None) -> str:
    return "-" if value is None else str(int(value))


def number(value: float | None, digits: int = 1) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def print_summary(runs: list[dict], case_ids: list[str]) -> dict:
    by_case = {case_id: tally([r["score"] for r in runs if r["case"] == case_id]) for case_id in case_ids}
    overall = tally([r["score"] for r in runs])
    errors = sum(1 for r in runs if r["errors"])
    print(f"\n== {len(runs)} runs, {errors} with errors ==")
    print("k/N over the runs where the flag applies; invent = the start reply said too much or named a finding, "
          "spoken = an outcome was recorded,\nnarrat = the outcome sounds like the agent's narration, chat = the chat "
          "holds a line of the report, rebrw = the follow-up started a task again")
    width = max([len("case"), len("all"), *(len(c) for c in case_ids)])
    print(f"\n{'case':<{width}}  " + "  ".join(f"{HEADS[name]:>8}" for name in FLAGS))
    for label, row in (*by_case.items(), ("all", overall)):
        print(f"{label:<{width}}  " + "  ".join(f"{cell(row['flags'][name]):>8}" for name in FLAGS))
    print(f"\n{'medians':<{width}}  {'start chars':>11}  {'start s':>8}  {'outcome s':>9}  {'followup s':>10}")
    for label, row in (*by_case.items(), ("all", overall)):
        m = row["medians"]
        print(f"{label:<{width}}  {number(m['start_extra_chars'], 0):>11}  {number(m['start_first_audio_s'], 2):>8}  "
              f"{number(m['outcome_first_audio_s'], 2):>9}  {number(m['followup_first_audio_s'], 2):>10}")
    for r in runs:
        for problem in r["errors"]:
            print(f"  error: {r['case']} #{r['repeat']}: {problem}")
    return {"by_case": by_case, "overall": overall, "runs_with_errors": errors}


@contextlib.contextmanager
def outcome_wording(name: str):
    """Speak the tasks' outcomes with `vaice.assistant.REPORT` or `SAY` (`--outcome`) for the block, then restore."""
    original = vaice_assistant.OUTCOME
    vaice_assistant.OUTCOME = getattr(vaice_assistant, OUTCOME_WORDINGS[name])
    try:
        yield
    finally:
        vaice_assistant.OUTCOME = original


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    all_cases, fixtures = load_cases()
    cases = evals_run.select_cases(all_cases, args.cases)
    url = config.server_base(args.url)
    info = evals_run.server_info(url)
    prompt = args.prompt or config.PROMPT
    out = args.out or HERE / "results" / f"browse-flow-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.json"

    print(f"server {url}: {info.get('model_alias')} {info.get('build_info')} temperature={info.get('temperature')}")
    runs: list[dict] = []
    started = datetime.now(UTC).isoformat(timespec="seconds")
    total = len(cases) * args.repeats
    try:
        with tempfile.TemporaryDirectory(prefix="vaice-evals-") as scratch, outcome_wording(args.outcome):
            audio = {case["id"]: (evals_run.audio_for(case, Path(scratch)),
                                  evals_run.audio_for({"id": f"{case['id']}-followup", "say": case["followup"]["say"]},
                                                      Path(scratch)))
                     for case in cases}
            for repeat in range(1, args.repeats + 1):  # repeats outermost, as run.py
                for case in cases:
                    request, followup = audio[case["id"]]
                    record = run_flow(url, prompt, case, fixtures[case["fixture"]], request, followup,
                                      vision=args.vision, task_seconds=args.task_seconds)
                    record.update(case=case["id"], say=case["say"], fixture=case["fixture"], repeat=repeat)
                    runs.append(record)
                    flags = ", ".join(f"{HEADS[name]} {flag_mark(record['score'][name])}" for name in FLAGS)
                    print(f"[{len(runs)}/{total}] {case['id']} #{repeat}: {flags}"
                          + (f" (errors: {'; '.join(record['errors'])})" if record["errors"] else ""), file=sys.stderr)
    except KeyboardInterrupt:
        print("\ninterrupted: reporting the runs so far", file=sys.stderr)
    summary = print_summary(runs, [c["id"] for c in cases])

    config_used = {"repeats": args.repeats, "task_seconds": args.task_seconds, "vision": args.vision, "prompt": prompt,
                   "outcome": args.outcome, "cases": [c["id"] for c in cases], "platform": platform.platform()}
    document = {"label": args.label, "started": started, "config": config_used, "server": info,
                "summary": summary, "runs": runs}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(document, indent=2) + "\n")
    print(f"\nresults written to {out}")


if __name__ == "__main__":
    main()
