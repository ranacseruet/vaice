"""Live check: can the configured browser agent do a research task?

Run by hand (see evals/README.md); not imported by the app, not run in CI. `run.py --browse` only scores whether the
voice model calls `browse`; this starts the real browser agent (`app._open_browse()`: Pi through pi-acp, or
VAICE_BROWSE=acp) with vaice's own configuration, gives it research goals, approves every step it asks about, and
records the answer, the tools, the pages and the time. It contacts real sites and the browse model.
"""
import argparse
import json
import logging
import platform
import signal
import subprocess
import sys
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

from vaice import app, browse, config

HERE = Path(__file__).parent
DEFAULT_GOALS = [
    "Research the best drip coffee makers under 100 dollars and name three.",
    "Find out what is new with the James Webb Space Telescope this month.",
    "Look up the best noise cancelling headphones and name the top two.",
]
# A test browser must never play audio on the user's machine: even headless, Chrome runs an AudioService on macOS
MUTE_ARGS = ["--chromeArg=--mute-audio", "--chromeArg=--audio-muffling"]
LEFTOVER_WAIT = 1.0  # seconds after close() before the profile's holders are counted
PS_TIMEOUT = 5.0
SNIPPET_CHARS = 100


def fail(message: str) -> None:
    sys.exit(f"evals: {message}")


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live check of the browser agent doing research tasks (real sites, "
                                                 "the real browse model).")
    parser.add_argument("--goal", action="append", help="a research goal; repeatable (default: three built-in goals)")
    parser.add_argument("--repeats", type=int, default=1, help="runs per goal (default: 1)")
    parser.add_argument("--seconds", type=float, help="seconds a task may take (default: VAICE_BROWSE_SECONDS)")
    parser.add_argument("--visible", action="store_true", help="show Chrome's window (default: headless; muted always)")
    parser.add_argument("--out", type=Path, help="results file (default: evals/results/browse-live-<UTC timestamp>"
                                                 ".json)")
    parser.add_argument("--label", default="", help="free text stored in the results, e.g. the browse model")
    args = parser.parse_args(argv)
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    if args.seconds is not None and not args.seconds > 0:
        parser.error("--seconds must be > 0")
    return args


def prepare_browser(visible: bool) -> None:
    """Headless unless `visible`, and the browser always muted (the app builds the arguments in `app`)."""
    if not visible:
        config.BROWSE_HEADLESS = True
    original = browse.default_browser_args
    app.default_browser_args = lambda profile: [*original(profile), *MUTE_ARGS]


def urls_in(arguments: object) -> list[str]:
    """The page a navigation tool was given (its `url` argument), when there is one."""
    url = arguments.get("url") if isinstance(arguments, dict) else None
    return [url] if isinstance(url, str) and url else []


def run_goal(agent: browse.BrowseAgent, goal: str, repeat: int, seconds: float,
             interrupted: threading.Event) -> dict:
    """One task on `agent`. Never raises: a failure is the run's outcome."""
    tools: list[str] = []
    asks: list[str] = []
    urls: list[str] = []

    def on_progress(title: str, arguments: dict) -> None:
        tools.append(title)
        urls.extend(urls_in(arguments))

    def on_ask(title: str, message: str, stop) -> bool:
        asks.append(title)
        return True

    record: dict = {"goal": goal, "repeat": repeat, "outcome": None, "answer": None, "error": None, "spoken": None}
    started = time.monotonic()
    try:
        answer = agent.run(goal, stop=interrupted.is_set, on_progress=on_progress, on_ask=on_ask, deadline_s=seconds,
                           on_text=lambda text: None)
        record.update(outcome="answer" if answer.strip() else "empty", answer=answer)
    except Exception as exc:  # noqa: BLE001 - whatever ended the run is its result
        record.update(outcome=type(exc).__name__, error=str(exc), spoken=getattr(exc, "spoken", None))
    record.update(seconds=round(time.monotonic() - started, 1), tools=tools, asks=asks, urls=urls)
    return record


def ps_listing() -> str:
    try:
        return subprocess.run(["ps", "-axww", "-o", "pid=,command="], capture_output=True, text=True,
                              stdin=subprocess.DEVNULL, timeout=PS_TIMEOUT, check=True).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"evals: could not list processes ({type(exc).__name__})", file=sys.stderr)
        return ""


def agent_info() -> dict:
    info = {"browse": config.BROWSE, "model": config.BROWSE_MODEL or None, "confirm": config.BROWSE_CONFIRM}
    if config.BROWSE == "acp":
        info["agent_command"] = config.BROWSE_AGENT
    elif config.BROWSE == "pi":
        version = browse.pi_version()
        info["pi_version"] = ".".join(map(str, version)) if version else None
    return info


def print_summary(runs: list[dict], leftovers: int | None) -> None:
    print(f"\n== {len(runs)} runs ==")
    width = max([len("goal"), *(min(len(r["goal"]), 40) for r in runs)])
    print(f"{'goal':<{width}}  {'#':>2}  {'outcome':<14}{'seconds':>8}{'tools':>6}  answer")
    for r in runs:
        snippet = (r["answer"] or r["spoken"] or r["error"] or "").replace("\n", " ")[:SNIPPET_CHARS]
        print(f"{r['goal'][:40]:<{width}}  {r['repeat']:>2}  {r['outcome']:<14}{r['seconds']:>8.1f}"
              f"{len(r['tools']):>6}  {snippet}")
    answered = sum(1 for r in runs if r["outcome"] == "answer")
    print(f"\nanswers: {answered}/{len(runs)}")
    print(f"processes still holding the browser profile: {'unknown' if leftovers is None else leftovers}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(levelname)s %(name)s: %(message)s")
    goals = args.goal or DEFAULT_GOALS
    seconds = args.seconds or config.BROWSE_SECONDS
    out = args.out or HERE / "results" / f"browse-live-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.json"
    prepare_browser(args.visible)
    agent = app._open_browse()
    if agent is None:
        fail("the browser agent could not be started: the startup warnings above say why "
             "(VAICE_BROWSE, VAICE_BROWSE_URL / VAICE_BROWSE_AGENT, the `browse` extra, Node.js)")

    interrupted = threading.Event()
    previous = None
    if threading.current_thread() is threading.main_thread():
        previous = signal.signal(signal.SIGINT, lambda *_: interrupted.set())
    started = datetime.now(UTC).isoformat(timespec="seconds")
    runs: list[dict] = []
    leftovers: int | None = None
    profile = getattr(agent, "_chrome_profile", None)
    info: dict = {}
    try:
        info = agent_info()
        total = len(goals) * args.repeats
        for goal in goals:
            for repeat in range(1, args.repeats + 1):
                if interrupted.is_set():
                    break
                runs.append(run_goal(agent, goal, repeat, seconds, interrupted))
                print(f"[{len(runs)}/{total}] {goal[:60]} #{repeat}: {runs[-1]['outcome']} in {runs[-1]['seconds']} s",
                      file=sys.stderr)
        if interrupted.is_set():
            print("\ninterrupted: reporting the runs so far", file=sys.stderr)
    finally:
        agent.close()
        if previous is not None:
            signal.signal(signal.SIGINT, previous)
    if profile is not None:
        time.sleep(LEFTOVER_WAIT)
        leftovers = len(browse._profile_pids(profile, ps_listing()))
    print_summary(runs, leftovers)

    document = {"label": args.label, "started": started,
                "config": {**info, "repeats": args.repeats, "seconds": seconds, "headless": config.BROWSE_HEADLESS,
                           "goals": goals, "platform": platform.platform()},
                "leftover_processes": leftovers, "runs": runs}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(document, indent=2) + "\n")
    print(f"\nresults written to {out}")
    clean = bool(runs) and not interrupted.is_set() and all(r["outcome"] == "answer" for r in runs)
    return 0 if clean and not leftovers else 1


if __name__ == "__main__":
    sys.exit(main())
