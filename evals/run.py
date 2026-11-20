"""Live eval: how reliably do spoken commands turn into the right skill call?

Run by hand against a real llama-server with an audio model (see evals/README.md); not imported by the app, not run
in CI. Every run is the app's own path: `Assistant` over /v1/realtime with the real skill definitions as
the session's tools and the real executor, over stub skills (canned results; nothing is executed on this machine).
The calls are read from the turn's tool events, so a confirm skill's request and `browse`'s start count as the call.
"""
import argparse
import json
import platform
import statistics
import subprocess
import sys
import tempfile
import time
import wave
from collections.abc import Sequence
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import requests
import scoring

from vaice import browse_task, config
from vaice.assistant import Assistant
from vaice.conversation import WAITING_FOR_CONFIRMATION, Turn
from vaice.memory import Memory
from vaice.skills import Skill, SkillResult, Skills, default_skills
from vaice.vision import VisionClient, vision_skill

HERE = Path(__file__).parent
HISTORY = [
    ("user", "Hey, are you there?"),
    ("assistant", "Yes, I'm here."),
    ("user", "What day is it?"),
    ("assistant", "It's Monday."),
]
CHECK_TIMEOUT = 5  # seconds for the health and props requests
SAY_TIMEOUT = 30
SAY_ATTEMPTS = 3  # `say` sometimes hangs; see synthesize()
SILENT_SECONDS = 0.5  # a reply with less audio than this is counted as silent
# What the stubbed skills answer, in the wording of the real ones; nothing is executed
SKILL_STUBS = {
    "disk_space": "160 gigabytes free of 460 gigabytes",
    "battery_status": "battery at 78 percent, running on battery power, about 4 hours and 12 minutes left",
    "system_info": "macOS 15.6 on Example-MacBook, up 3 days and 4 hours, 48 gigabytes of memory",
    "top_processes": "Google Chrome 12 percent of memory, Xcode 9 percent, Slack 4 percent",
    "find_files": "3 files: report.pdf, report-final.pdf, old report.docx",
}
# The actions answer with their own arguments, as the real ones do: a canned "opened Safari" for a call
# that asked for Calculator made the model call again (seen as duplicates and odd answers)
ACTION_STUBS = {
    "open_app": lambda args: f"opened {args.get('name')}",
    "quit_app": lambda args: f"quit {args.get('name')}",
    "set_volume": lambda args: f"volume set to {args.get('level', 40)} percent",
    "volume_up": lambda args: "volume set to 60 percent",
    "volume_down": lambda args: "volume set to 40 percent",
    "mute_sound": lambda args: "sound muted" if args.get("muted", True) else "sound unmuted",
}
# run_shell (--shell): it asks, and nothing is run; the result is only there if a case gets past the question
SHELL_STUBS = {"run_shell": lambda args: "the command finished. Output: ok"}
# look_at_screen (--vision): a fixed fake screen, described in the wording of the real result; nothing is captured.
# What the question is about picks what is "seen", so a call that asks about the dialog gets its text
VISION_SCREEN = "a Numbers window titled Quarterly report"
VISION_DIALOG = 'a dialog that says "Your disk is almost full"'


def vision_stub(args: dict) -> str:
    question = str(args.get("question", "")).casefold()
    if any(word in question for word in ("error", "dialog", "message", "alert", "popup", "say")):
        return f"the screen shows: {VISION_DIALOG} over {VISION_SCREEN}"
    return f"the screen shows: {VISION_SCREEN}"


VISION_STUBS = {"look_at_screen": vision_stub}
BROWSER_STATUS_STUB = "the browser task is running: open python.org; it has run for 20 seconds; now on python.org"
TRANSCRIPT_WAIT = 5.0  # the transcript is reported by a parallel session, maybe after the reply; wait this long


def fail(message: str) -> None:
    sys.exit(f"evals: {message}")


def load_json(name: str):
    return json.loads((HERE / name).read_text())


def read_wav(path: Path) -> tuple[int, np.ndarray]:
    """A 16-bit PCM WAV as the (sample_rate, int16 array shaped (1, n)) frame the responder takes."""
    with wave.open(str(path), "rb") as wav:
        if wav.getsampwidth() != 2:
            fail(f"{path} is not 16-bit PCM")
        rate, channels = wav.getframerate(), wav.getnchannels()
        raw = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2")
    if channels > 1:  # recordings should be mono; fold anything else rather than refuse it
        raw = raw.reshape(-1, channels).mean(axis=1).astype("<i2")
    return rate, raw.reshape(1, -1)


def _say(sentence: str, out: Path) -> bool:
    """One `say` run; False when it hung. Any other failure ends the eval with its reason."""
    try:
        subprocess.run(["say", "-o", str(out), "--data-format=LEI16@16000", sentence],
                       check=True, capture_output=True, stdin=subprocess.DEVNULL, timeout=SAY_TIMEOUT)
    except FileNotFoundError:
        fail(f"no recording and no macOS `say` for {sentence!r}: put a 16-bit PCM mono WAV in "
             f"{HERE / 'recordings'} named <case id>.wav")
    except subprocess.CalledProcessError as exc:
        fail(f"`say` failed for {sentence!r}: {exc.stderr.decode(errors='replace').strip()}")
    except subprocess.TimeoutExpired:
        return False
    return True


def synthesize(sentence: str, out: Path) -> None:
    # `say` occasionally hangs (seen in a full run, on a sentence that worked before and after)
    for attempt in range(1, SAY_ATTEMPTS + 1):
        if _say(sentence, out):
            return
        print(f"`say` did not finish within {SAY_TIMEOUT}s for {sentence!r} (attempt {attempt} of {SAY_ATTEMPTS})",
              file=sys.stderr)
    fail(f"`say` hung {SAY_ATTEMPTS} times for {sentence!r}: record it instead "
         f"({HERE / 'recordings'}/<case id>.wav) or run again")


def audio_for(case: dict, scratch: Path) -> tuple[int, np.ndarray]:
    recording = HERE / "recordings" / f"{case['id']}.wav"
    if recording.exists():
        return read_wav(recording)
    synthesized = scratch / f"{case['id']}.wav"
    synthesize(case["say"], synthesized)
    return read_wav(synthesized)


def stub_skills(shell: bool = False, memory: Memory | None = None, vision: bool = False,
                browse: bool = False, browse_control: Sequence[Skill] | None = None) -> Skills:
    """The real skills' definitions with canned results: the model sees what the app offers, and nothing
    runs on this machine (the stubs replace each skill's `run`, and `run_stoppable` is dropped so the real
    command runner cannot be reached; Skills.run still checks the arguments). A `confirm` skill's `ask` is
    stubbed to always ask, so no check of the machine (is the app running?) is made. `shell` adds `run_shell`,
    `vision` adds `look_at_screen` (its stub answers from `VISION_STUBS`: no screenshot is taken, nothing is sent),
    `browse` adds `browse` (the real definition: it is a background skill that only validates its goal here, so no
    browser task is started and nothing is run) and the two that ask about and end it, `browser_status` and
    `stop_browsing` (their stubs answer with a fixed text; `browse_control` replaces those three with the caller's
    own, as `browse_flow.py` does to wire in the real assistant's busy check, status and stop).

    `memory` adds the memory skills, which stay real: they only touch that store, which must be in memory only
    (no path), so an eval can never read or write the app's own `memory.json`."""
    if memory is not None and memory.path is not None:
        raise ValueError("the eval's memory must be in memory only: it must never touch a notes file")
    answers = {**ACTION_STUBS, **SHELL_STUBS, **VISION_STUBS}
    # the real definition, over a client that is never used: the stub replaces the skill's `run`
    screen = vision_skill(VisionClient("http://stub.invalid/v1", 1), "stub") if vision else None
    # the real definitions of `browse`, `browser_status` and `stop_browsing`; the last two answer with a fixed text
    control = (browse_control if browse_control is not None else
               browse_task.browse_skills(busy=None, status=lambda: SkillResult(BROWSER_STATUS_STUB),
                                         stop=lambda: SkillResult(browse_task.STOPPED))) if browse else ()
    return Skills([
        skill if skill.name not in answers and skill.name not in SKILL_STUBS else
        replace(skill, run_stoppable=None,
                run=(lambda args, answer=answers[skill.name]: SkillResult(answer(args)))
                if skill.name in answers else (lambda _args, text=SKILL_STUBS[skill.name]: SkillResult(text)),
                **({"ask": lambda _args, name=skill.name: f"Shall I run {name.replace('_', ' ')}?"}
                   if skill.kind == "confirm" else {}))
        for skill in default_skills(shell=shell, memory=memory, vision=screen, browse=browse,
                                    browse_skills=control).skills])


def case_memory(case: dict) -> Memory:
    """A fresh in-memory store holding the case's `notes` (none for most cases): one per run, so a note a run
    remembers or forgets never reaches the next one."""
    memory = Memory(None)
    for note in case.get("notes", []):
        memory.add(note)
    return memory


def run_case(url: str, prompt: str, skills: Skills, memory: Memory | None, audio, history) -> dict:
    """One reply: the audio through `Assistant`, timed to the first audio, with the calls read back from
    the turn's tool events. `pending`: the run ended in a request for the Run button or a browser task, whose spoken
    answer is the model's wording of that, not an answer to the question."""
    responder = Assistant(url=url, prompt=prompt, skills=skills, memory=memory)
    turn = Turn()
    record: dict = {"error": None}
    first = None
    seconds = 0.0
    start = time.perf_counter()
    try:
        for rate, frame in responder.respond(audio, history, turn):
            if first is None:
                first = time.perf_counter() - start
            seconds += len(frame) / rate
    except Exception as exc:  # noqa: BLE001 - a failed run is recorded with its error, never aborts the eval
        record["error"] = f"{type(exc).__name__}: {exc}"
    total = time.perf_counter() - start
    turn.finish()
    turn.transcribed.wait(TRANSCRIPT_WAIT)  # keeps the transcription session out of the next run
    events = list(turn.tool_events)
    record.update(
        calls=[[event["name"], event["arguments"]] for event in events],
        pending=any(event["output"] in (WAITING_FOR_CONFIRMATION, browse_task.WORKING) for event in events),
        first_audio_s=first, total_s=total, audio_s=seconds, latency_s=first, answer=turn.assistant_text,
        transcript=turn.user_text)
    return record


def server_info(url: str) -> dict:
    try:
        health = requests.get(f"{url}/health", timeout=CHECK_TIMEOUT)
    except requests.RequestException as exc:
        fail(f"llama-server is not reachable at {url}/health ({type(exc).__name__}: {exc})")
    if not health.ok:
        fail(f"llama-server at {url} answered /health with HTTP {health.status_code}: {health.text[:200]}")
    info: dict = {"url": url}
    try:
        r = requests.get(f"{url}/props", timeout=CHECK_TIMEOUT)
        r.raise_for_status()
        props = r.json()
    except (requests.RequestException, ValueError) as exc:
        info["props_error"] = f"{type(exc).__name__}: {exc}"
        return info
    params = (props.get("default_generation_settings") or {}).get("params") or {}
    model_path = props.get("model_path")
    info.update(build_info=props.get("build_info"), model_alias=props.get("model_alias"),
                model_file=Path(model_path).name if model_path else None, temperature=params.get("temperature"))
    return info


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Eval spoken-command skill calling against a live llama-server.")
    parser.add_argument("--url", default=config.LLAMACPP_URL, help="llama-server URL (default: VAICE_LLAMACPP_URL)")
    parser.add_argument("--repeats", type=int, default=5, help="runs per case (default: 5)")
    parser.add_argument("--cases", help="comma-separated case ids (default: all)")
    parser.add_argument("--cases-file", default="cases.json", help="the cases: a file in evals/ or a path "
                                                                   "(default: cases.json)")
    parser.add_argument("--prompt", help="reply instructions (default: VAICE_PROMPT)")
    parser.add_argument("--shell", action="store_true",
                        help="also offer the run_shell skill (stubbed: it asks, nothing is run)")
    parser.add_argument("--vision", action="store_true",
                        help="also offer the look_at_screen skill (stubbed: a fixed fake screen, "
                             "nothing is captured or sent)")
    parser.add_argument("--browse", action="store_true",
                        help="also offer the browse skill (a background skill: its call is scored, "
                             "no browser task is started)")
    parser.add_argument("--memory", action="store_true",
                        help="also offer the memory skills and load the case notes (default: off, as in the app); "
                             "the store is always in memory, never the app's memory.json")
    parser.add_argument("--history", action=argparse.BooleanOptionalAction, default=True,
                        help="send two fixed earlier exchanges (default: on)")
    parser.add_argument("--out", type=Path, help="results file (default: evals/results/<UTC timestamp>.json)")
    parser.add_argument("--label", default="", help="free text stored in the results, e.g. the quantization")
    args = parser.parse_args(argv)
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    return args


def case_history(entries: list) -> list[tuple[str, str]]:
    """A case's own `"history"` as the (role, text) turns the responder takes."""
    return [(role, text) for role, text in entries]


def select_cases(cases: list[dict], wanted: str | None) -> list[dict]:
    if not wanted:
        return cases
    ids = [i.strip() for i in wanted.split(",") if i.strip()]
    by_id = {c["id"]: c for c in cases}
    missing = [i for i in ids if i not in by_id]
    if missing:
        fail(f"unknown case id(s): {', '.join(missing)}")
    return [by_id[i] for i in ids]


def fmt(seconds: float | None) -> str:
    return "-" if seconds is None else f"{seconds:.2f}s"


def print_summary(runs: list[dict], case_ids: list[str]) -> dict:
    agg = scoring.aggregate(runs)
    audio = [r["audio_s"] for r in runs if not r["error"]]
    agg["audio_median_s"] = statistics.median(audio) if audio else None
    agg["silent"] = sum(1 for seconds in audio if seconds < SILENT_SECONDS)
    print(f"\n== {agg['runs']} runs ==")
    for outcome in scoring.OUTCOMES:
        print(f"  {outcome:<11}{agg['counts'][outcome]:>4}  {agg['percent'][outcome]:>5.1f}%")
    print(f"  {'duplicate':<11}{agg['duplicate']:>4}  {agg['duplicate_percent']:>5.1f}%  (same tool and arguments "
          "twice; also counted above)")
    print(f"  time to first audio: median {fmt(agg['latency_median_s'])}, max {fmt(agg['latency_max_s'])}")
    print(f"  audio per reply: median {fmt(agg['audio_median_s'])}; under {SILENT_SECONDS:g} s: {agg['silent']}")
    for label, key in (("answer faithful", "answer_faithful"), ("command ok", "command_ok")):
        if key in agg:  # only when a case asked for it; apart from the verdicts above
            print(f"  {label} {agg[key]['ok']}/{agg[key]['of']}")
    cells = scoring.per_case(runs)
    checks = {label: scoring.per_case_check(runs, key) for label, key in (("answer", "answer_ok"),
                                                                           ("command", "command_ok"))}
    checks = {label: cell for label, cell in checks.items() if cell}  # the columns some case asked for
    width = max([len("case"), *(len(c) for c in case_ids)])
    print(f"\n{'case':<{width}}  " + "  ".join(f"{name:>8}" for name in ["calls", *checks]))
    for case_id in case_ids:
        correct, total = cells.get(case_id, (0, 0))
        row = [f"{correct}/{total}" if total else "-"]
        for cell in checks.values():
            ok, of = cell.get(case_id, (0, 0))
            row.append(f"{ok}/{of}" if of else "-")
        print(f"{case_id:<{width}}  " + "  ".join(f"{cell:>8}" for cell in row))
    heard = [r for r in runs if r.get("transcript") is not None]
    wrong = [r for r in heard if scoring.misheard(r["say"], r["transcript"])]
    if heard:
        print(f"\ntranscripts differing from the sentence (case, punctuation ignored): {len(wrong)} of {len(heard)}")
        for r in wrong:
            print(f"  {r['case']} #{r['repeat']}: said {r['say']!r}, heard {r['transcript']!r}")
        agg["misheard"] = {"differing": len(wrong), "transcribed": len(heard)}
    return agg


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cases = select_cases(load_json(args.cases_file), args.cases)
    url = config.server_base(args.url)
    info = server_info(url)
    prompt = args.prompt or config.PROMPT
    history = HISTORY if args.history else []
    out = args.out or HERE / "results" / f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.json"

    print(f"server {url}: {info.get('model_alias')} {info.get('build_info')} temperature={info.get('temperature')}")
    runs: list[dict] = []
    started = datetime.now(UTC).isoformat(timespec="seconds")
    total = len(cases) * args.repeats
    try:
        with tempfile.TemporaryDirectory(prefix="vaice-evals-") as scratch:
            audio = {case["id"]: audio_for(case, Path(scratch)) for case in cases}
            # repeats outermost: a drifting server (thermals, cache) spreads over the cases instead of one
            for repeat in range(1, args.repeats + 1):
                for case in cases:
                    # a case may carry its own earlier turns (even with --no-history: that is what it tests)
                    turns = case_history(case["history"]) if "history" in case else history
                    memory = case_memory(case) if args.memory else None  # fresh for every run
                    skills = stub_skills(shell=args.shell, memory=memory, vision=args.vision, browse=args.browse)
                    record = run_case(url, prompt, skills, memory, audio[case["id"]], turns)
                    record.update(case=case["id"], say=case["say"], repeat=repeat)
                    if not record["error"]:
                        calls = [tuple(c) for c in record["calls"]]
                        record["score"] = scoring.score(case["expect"], calls)
                        # apart from the verdict: a right call can have a wrong answer
                        record["score"].update(
                            scoring.check_answer(case["expect"], calls, record["answer"], record["pending"]))
                    runs.append(record)
                    outcome = f"error: {record['error']}" if record["error"] else record["score"]["verdict"]
                    print(f"[{len(runs)}/{total}] {case['id']} #{repeat}: {outcome}", file=sys.stderr)
    except KeyboardInterrupt:
        print("\ninterrupted: reporting the runs so far", file=sys.stderr)
    summary = print_summary(runs, [c["id"] for c in cases])

    config_used = {"cases_file": args.cases_file, "repeats": args.repeats, "shell": args.shell,
                   "vision": args.vision, "browse": args.browse, "memory": args.memory, "history": history,
                   "prompt": prompt, "cases": [c["id"] for c in cases], "platform": platform.platform()}
    document = {"label": args.label, "started": started, "config": config_used, "server": info,
                "summary": summary, "runs": runs}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(document, indent=2) + "\n")
    print(f"\nresults written to {out}")


if __name__ == "__main__":
    main()
