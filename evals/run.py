"""Live eval: how reliably do spoken commands turn into the right tool call?

Run by hand against a real llama-server with an audio model (see evals/README.md); not imported by
the app, not run in CI. Three paths are measured with repeats:

- interleaved: the app's own path, LlamaCppResponder over /v1/realtime with the tools declared
  (the stub tools in tools.json; nothing is executed for real);
- text: transcribe the audio with a realtime `transcription` session, then route the transcript
  through POST /v1/chat/completions with the same tools (routing only: no tool result is fed back);
- assistant (opt-in, with its own cases and skills): assistant mode's real code up to the answer, without
  /tts: transcribe, then the brain/skills loop of AssistantResponder with the real skill definitions
  and canned results (nothing is executed on this machine).
"""
import argparse
import json
import platform
import subprocess
import sys
import tempfile
import threading
import time
import wave
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
import requests
import scoring

from vaice import config
from vaice.brain import ChatBrain
from vaice.conversation import Turn
from vaice.responders.assistant import NOT_HEARD, build_messages, decide
from vaice.responders.llamacpp import LlamaCppResponder, _transcribe
from vaice.skills import SkillResult, Skills, default_skills

HERE = Path(__file__).parent
PATHS = ("interleaved", "text", "assistant")
DEFAULT_PATHS = ("interleaved", "text")  # the assistant path has its own skills and cases: ask for it
TEXT_PROMPT = ("You are a voice assistant on the user's Mac. Your reply is spoken aloud: "
               "answer in one or two short plain sentences.")
HISTORY = [
    ("user", "Hey, are you there?"),
    ("assistant", "Yes, I'm here."),
    ("user", "What day is it?"),
    ("assistant", "It's Monday."),
]
CHECK_TIMEOUT = 5  # seconds for the health and props requests
SAY_TIMEOUT = 30
SAY_ATTEMPTS = 3  # `say` sometimes hangs; see synthesize()
ROUTE_TIMEOUT = 120
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
TRANSCRIPT_WAIT = 5.0  # the interleaved path's transcript is reported after the reply; wait this long for it


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


class StubTools:
    """Records the calls a run makes and answers with each tool's canned result; nothing is executed."""

    def __init__(self, results: dict[str, str]):
        self._results = results
        self.calls: list[tuple[str, str]] = []
        self._lock = threading.Lock()  # the responder calls this from its session thread

    def __call__(self, name: str, arguments: str) -> str:
        with self._lock:
            self.calls.append((name, arguments))
        return self._results.get(name, f"tool {name} failed: no such tool")

    def received(self) -> list[tuple[str, str]]:
        with self._lock:
            return list(self.calls)


def run_interleaved(url: str, prompt: str, tools: list[dict], results: dict[str, str], audio, history) -> dict:
    stub = StubTools(results)
    responder = LlamaCppResponder(url=url, prompt=prompt, tools=tools, executor=stub)
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
    record.update(calls=[list(c) for c in stub.received()], first_audio_s=first, total_s=total, audio_s=seconds,
                  assistant_text=turn.assistant_text, transcript=turn.user_text, latency_s=first)
    return record


def chat_tools(tools: list[dict]) -> list[dict]:
    """The flat definitions in the chat-completions nesting."""
    return [{"type": "function", "function": {k: v for k, v in tool.items() if k != "type"}} for tool in tools]


def route(url: str, system: str, history, transcript: str, tools: list[dict]) -> tuple[list[tuple[str, str]], str]:
    messages = [{"role": "system", "content": system}]
    messages += [{"role": role, "content": text} for role, text in history]
    messages.append({"role": "user", "content": transcript})
    body = {"messages": messages, "tools": chat_tools(tools), "temperature": 0, "max_tokens": 300}
    r = requests.post(f"{url}/v1/chat/completions", json=body, timeout=ROUTE_TIMEOUT)
    if not r.ok:
        raise RuntimeError(f"chat completions answered HTTP {r.status_code}: {r.text[:200]}")
    message = r.json()["choices"][0]["message"]
    calls = []
    for call in message.get("tool_calls") or []:
        arguments = call["function"].get("arguments", "")
        calls.append((call["function"]["name"], arguments if isinstance(arguments, str) else json.dumps(arguments)))
    return calls, message.get("content") or ""


def run_text(url: str, prompt: str, tools: list[dict], audio, history) -> dict:
    record: dict = {"error": None, "calls": [], "transcript": None, "content": None, "latency_s": None}
    start = time.perf_counter()
    try:
        transcript = _transcribe(url, config.LLAMACPP_TIMEOUT, audio)
        record["transcribe_s"] = time.perf_counter() - start
        record["transcript"] = transcript
        routed = time.perf_counter()
        calls, content = route(url, prompt, history, transcript, tools)
        record["route_s"] = time.perf_counter() - routed
        record["calls"] = [list(c) for c in calls]
        record["content"] = content
        record["latency_s"] = record["transcribe_s"] + record["route_s"]
    except Exception as exc:  # noqa: BLE001 - a failed run is recorded with its error, never aborts the eval
        record["error"] = f"{type(exc).__name__}: {exc}"
    return record


def stub_skills(shell: bool = False) -> Skills:
    """The real skills' definitions with canned results: the model sees what the app offers, and nothing
    runs on this machine (the stubs replace each skill's `run`, and `run_stoppable` is dropped so the real
    command runner cannot be reached; Skills.run still checks the arguments). A `confirm` skill's `ask` is
    stubbed to always ask, so no check of the machine (is the app running?) is made. `shell` adds `run_shell`."""
    answers = {**ACTION_STUBS, **SHELL_STUBS}
    return Skills([
        replace(skill, run_stoppable=None,
                run=(lambda args, answer=answers[skill.name]: SkillResult(answer(args)))
                if skill.name in answers else (lambda _args, text=SKILL_STUBS[skill.name]: SkillResult(text)),
                **({"ask": lambda _args, name=skill.name: f"Shall I run {name.replace('_', ' ')}?"}
                   if skill.kind == "confirm" else {}))
        for skill in default_skills(shell=shell).skills])


def run_assistant(url: str, prompt: str, skills: Skills, audio, history, brain: ChatBrain | None = None) -> dict:
    record: dict = {"error": None, "calls": [], "transcript": None, "answer": None, "pending": False,
                    "latency_s": None}
    brain = brain or ChatBrain.local(url, ROUTE_TIMEOUT)
    start = time.perf_counter()
    try:
        transcript = _transcribe(url, config.LLAMACPP_TIMEOUT, audio)
        record["transcribe_s"] = time.perf_counter() - start
        record["transcript"] = transcript
        if transcript:
            decided = time.perf_counter()
            decision = decide(brain, skills, build_messages(prompt, history, transcript), lambda: False,
                              turn=Turn())
            record["decide_s"] = time.perf_counter() - decided
            record["brain_s"] = decision.brain_seconds
            record["calls"] = [[call.name, call.arguments] for call in decision.calls]
            if decision.pending is not None:  # asked for a confirm skill: that is the call (it never runs here)
                record["calls"].append([decision.pending.name, decision.pending.arguments])
                record["pending"] = True  # its answer is a question, which the answer checks do not apply to
            record["answer"] = decision.answer
        else:  # as in the app: nothing is asked of the brain
            record["answer"] = NOT_HEARD
        record["latency_s"] = time.perf_counter() - start
    except Exception as exc:  # noqa: BLE001 - a failed run is recorded with its error, never aborts the eval
        record["error"] = f"{type(exc).__name__}: {exc}"
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
    parser = argparse.ArgumentParser(description="Eval spoken-command tool calling against a live llama-server.")
    parser.add_argument("--url", default=config.LLAMACPP_URL, help="llama-server URL (default: VAICE_LLAMACPP_URL)")
    parser.add_argument("--paths", default=",".join(DEFAULT_PATHS),
                        help="comma-separated: interleaved,text,assistant (default: interleaved,text)")
    parser.add_argument("--repeats", type=int, default=5, help="runs per case and path (default: 5)")
    parser.add_argument("--cases", help="comma-separated case ids (default: all)")
    parser.add_argument("--cases-file", help="the cases: a file in evals/ or a path (default: cases.json, or "
                                             "assistant_cases.json when --paths is assistant alone)")
    parser.add_argument("--prompt", help="reply instructions (interleaved) and system prompt (text, assistant); "
                                         "default: VAICE_LLAMACPP_PROMPT, a spoken-assistant prompt for text, "
                                         "VAICE_ASSISTANT_PROMPT for assistant")
    parser.add_argument("--shell", action="store_true",
                        help="assistant path: also offer the run_shell skill (stubbed: it asks, nothing is run)")
    parser.add_argument("--brain-url", default=config.BRAIN_URL,
                        help="assistant path: the brain's OpenAI-compatible base URL, /v1 included (default: "
                             "VAICE_BRAIN_URL; empty = the local llama-server). The key is read from "
                             "VAICE_BRAIN_API_KEY only. With a remote host the case sentences and the stub "
                             "results are sent there")
    parser.add_argument("--brain-model", default=config.BRAIN_MODEL,
                        help="assistant path: the model sent to --brain-url (default: VAICE_BRAIN_MODEL)")
    parser.add_argument("--history", action=argparse.BooleanOptionalAction, default=True,
                        help="send two fixed earlier exchanges (default: on)")
    parser.add_argument("--out", type=Path, help="results file (default: evals/results/<UTC timestamp>.json)")
    parser.add_argument("--label", default="", help="free text stored in the results, e.g. the quantization")
    args = parser.parse_args(argv)
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    args.paths = [p.strip() for p in args.paths.split(",") if p.strip()]
    unknown = [p for p in args.paths if p not in PATHS]
    if unknown or not args.paths:
        parser.error(f"--paths takes {', '.join(PATHS)}, got {unknown or 'nothing'}")
    if args.brain_url:
        try:
            args.brain_url = config.brain_base(args.brain_url)
        except ValueError as exc:
            parser.error(f"--brain-url: {exc}")
    if not args.cases_file:
        args.cases_file = "assistant_cases.json" if args.paths == ["assistant"] else "cases.json"
    return args


def make_brain(args: argparse.Namespace) -> ChatBrain | None:
    """The helper brain for the assistant path when `--brain-url` is set (None: the local llama-server). The key
    comes from VAICE_BRAIN_API_KEY (config), never from the command line."""
    if not args.brain_url:
        return None
    return ChatBrain.helper(args.brain_url, config.BRAIN_TIMEOUT, args.brain_model, config.BRAIN_API_KEY)


def describe_brain(args: argparse.Namespace) -> dict:
    """Which brain the assistant path used, for the report and the results: the host and the model, never the
    key or the URL's path."""
    if not args.brain_url:  # host None: the local llama-server
        return {"host": None, "model": None}
    return {"host": urlsplit(args.brain_url).netloc, "model": args.brain_model or None}


def brain_line(used: dict) -> str:
    return "the local llama-server" if used["host"] is None else f"{used['host']}, model {used['model'] or '(none)'}"


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


def print_summary(runs: list[dict], paths: list[str], case_ids: list[str], brain_used: dict | None = None) -> dict:
    summary = {}
    for path in paths:
        records = [r for r in runs if r["path"] == path]
        agg = scoring.aggregate(records)
        summary[path] = agg
        latency = {"interleaved": "time to first audio", "assistant": "time to answer text"}.get(
            path, "time to decision")
        brain = f", brain: {brain_line(brain_used)}" if path == "assistant" and brain_used is not None else ""
        print(f"\n== {path}: {agg['runs']} runs{brain} ==")
        for outcome in scoring.OUTCOMES:
            print(f"  {outcome:<11}{agg['counts'][outcome]:>4}  {agg['percent'][outcome]:>5.1f}%")
        print(f"  {'duplicate':<11}{agg['duplicate']:>4}  {agg['duplicate_percent']:>5.1f}%  (same tool and arguments "
              "twice; also counted above)")
        print(f"  {latency}: median {fmt(agg['latency_median_s'])}, max {fmt(agg['latency_max_s'])}")
        for label, key in (("answer faithful", "answer_faithful"), ("command ok", "command_ok")):
            if key in agg:  # only when a case asked for it; apart from the verdicts above
                print(f"  {label} {agg[key]['ok']}/{agg[key]['of']}")
    cells = {path: scoring.per_case([r for r in runs if r["path"] == path]) for path in paths}
    checks = {label: scoring.per_case_check(runs, key) for label, key in (("answer", "answer_ok"),
                                                                           ("command", "command_ok"))}
    checks = {label: cell for label, cell in checks.items() if cell}  # the columns some case asked for
    width = max([len("case"), *(len(c) for c in case_ids)])
    print(f"\n{'case':<{width}}  " + "  ".join(f"{name:>12}" for name in [*paths, *checks]))
    for case_id in case_ids:
        row = []
        for path in paths:
            correct, total = cells[path].get(case_id, (0, 0))
            row.append(f"{correct}/{total}" if total else "-")
        for cell in checks.values():
            ok, of = cell.get(case_id, (0, 0))
            row.append(f"{ok}/{of}" if of else "-")
        print(f"{case_id:<{width}}  " + "  ".join(f"{cell:>12}" for cell in row))
    heard = [r for r in runs if r["path"] in ("text", "assistant") and r.get("transcript") is not None]
    wrong = [r for r in heard if scoring.misheard(r["say"], r["transcript"])]
    if heard:
        print(f"\ntranscripts differing from the sentence (case, punctuation ignored): {len(wrong)} of {len(heard)}")
        for r in wrong:
            print(f"  {r['case']} #{r['repeat']}: said {r['say']!r}, heard {r['transcript']!r}")
        summary["misheard"] = {"differing": len(wrong), "transcribed": len(heard)}
    return summary


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cases = select_cases(load_json(args.cases_file), args.cases)
    tool_entries = load_json("tools.json")
    tools = [entry["definition"] for entry in tool_entries]
    results = {entry["definition"]["name"]: entry["result"] for entry in tool_entries}
    url = config.server_base(args.url)
    info = server_info(url)
    prompts = {"interleaved": args.prompt or config.LLAMACPP_PROMPT, "text": args.prompt or TEXT_PROMPT,
               "assistant": args.prompt or config.ASSISTANT_PROMPT}
    skills = stub_skills(shell=args.shell)
    history = HISTORY if args.history else []
    out = args.out or HERE / "results" / f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"

    print(f"server {url}: {info.get('model_alias')} {info.get('build_info')} temperature={info.get('temperature')}")
    brain = make_brain(args) if "assistant" in args.paths else None
    brain_used = describe_brain(args) if "assistant" in args.paths else None
    if brain_used is not None:
        print(f"assistant brain: {brain_line(brain_used)}")
        if brain is not None and urlsplit(args.brain_url).hostname not in config.LOCAL_HOSTS:
            print("  the case sentences and the stub skills' results are sent to that endpoint")
    runs: list[dict] = []
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    total = len(cases) * len(args.paths) * args.repeats
    try:
        with tempfile.TemporaryDirectory(prefix="vaice-evals-") as scratch:
            audio = {case["id"]: audio_for(case, Path(scratch)) for case in cases}
            # repeats outermost: a drifting server (thermals, cache) spreads over the cases instead of one
            for repeat in range(1, args.repeats + 1):
                for case in cases:
                    for path in args.paths:
                        if path == "interleaved":
                            record = run_interleaved(url, prompts[path], tools, results, audio[case["id"]], history)
                        elif path == "assistant":
                            record = run_assistant(url, prompts[path], skills, audio[case["id"]], history, brain)
                        else:
                            record = run_text(url, prompts[path], tools, audio[case["id"]], history)
                        record.update(case=case["id"], say=case["say"], path=path, repeat=repeat)
                        if not record["error"]:
                            calls = [tuple(c) for c in record["calls"]]
                            record["score"] = scoring.score(case["expect"], calls)
                            if path == "assistant":  # apart from the verdict: a right call can have a wrong answer
                                record["score"].update(
                                    scoring.check_answer(case["expect"], calls, record["answer"], record["pending"]))
                        runs.append(record)
                        outcome = f"error: {record['error']}" if record["error"] else record["score"]["verdict"]
                        print(f"[{len(runs)}/{total}] {path} {case['id']} #{repeat}: {outcome}", file=sys.stderr)
    except KeyboardInterrupt:
        print("\ninterrupted: reporting the runs so far", file=sys.stderr)
    summary = print_summary(runs, args.paths, [c["id"] for c in cases], brain_used)

    config_used = {"paths": args.paths, "cases_file": args.cases_file, "repeats": args.repeats,
                   "shell": args.shell, "history": history, "prompts": prompts, "cases": [c["id"] for c in cases],
                   "platform": platform.platform()}
    if brain_used is not None:
        config_used["brain"] = brain_used  # the host and the model only: never the key, never the URL's path
    document = {"label": args.label, "started": started, "config": config_used, "server": info,
                "summary": summary, "runs": runs}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(document, indent=2) + "\n")
    print(f"\nresults written to {out}")


if __name__ == "__main__":
    main()
