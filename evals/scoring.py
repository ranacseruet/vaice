"""Scoring for the tool-calling eval: pure functions, no I/O, no network.

A run's tool calls arrive as `(name, arguments)` pairs, the arguments being the JSON string the
model produced. `score` judges them against a case's `expect`, `aggregate` and `per_case` turn many
scored runs into the numbers the report prints.
"""
import json
import re
import statistics
from collections import Counter

VERDICTS = ("correct", "missed", "overcall", "wrong_args")
OUTCOMES = (*VERDICTS, "error")  # `error`: the run itself failed, so it has no verdict


def _parse(arguments: str) -> dict | None:
    """The arguments as a dict, or None when they are not a JSON object (an empty string is `{}`)."""
    if not arguments.strip():
        return {}
    try:
        parsed = json.loads(arguments)
    except ValueError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _number(value) -> float | None:
    """`value` as a number if it is one (or a numeric string); a bool is not a number."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _matches(wanted, actual) -> bool:
    """One expected value against one actual value.

    A list holds accepted alternatives. Strings compare case-insensitively after stripping; numbers
    by value, so `20`, `20.0` and `"20"` agree.
    """
    if isinstance(wanted, list):
        return any(_matches(option, actual) for option in wanted)
    if isinstance(wanted, bool):
        return actual is wanted
    if isinstance(wanted, (int, float)):
        number = _number(actual)
        return number is not None and number == wanted
    if isinstance(wanted, str):
        return isinstance(actual, str) and actual.strip().casefold() == wanted.strip().casefold()
    return actual == wanted


def args_mismatch(wanted: dict, arguments: str) -> str | None:
    """Why `arguments` (a JSON string) does not satisfy the expected subset `wanted`; None when it does."""
    parsed = _parse(arguments)
    if parsed is None:
        return f"arguments are not a JSON object: {arguments!r}"
    for key, expected in wanted.items():
        if key not in parsed:
            return f"missing {key!r}"
        if not _matches(expected, parsed[key]):
            return f"{key}={parsed[key]!r}, expected {expected!r}"
    return None


def _canonical(arguments: str) -> str:
    parsed = _parse(arguments)
    return arguments.strip() if parsed is None else json.dumps(parsed, sort_keys=True)


def _has_duplicate(calls: list[tuple[str, str]]) -> bool:
    """The same tool with the same arguments called more than once."""
    return any(n > 1 for n in Counter((name, _canonical(args)) for name, args in calls).values())


def _verdict(expect: dict, calls: list[tuple[str, str]]) -> tuple[str, str]:
    called = {name for name, _ in calls}
    if "any_of" in expect:
        allowed = set(expect["any_of"])
        if not calls:
            return "missed", f"no tool called, expected one of {sorted(allowed)}"
        if called - allowed:
            return "missed", f"called {sorted(called - allowed)}, expected one of {sorted(allowed)}"
        return "correct", f"called {sorted(called)}"
    if "tools" not in expect:
        raise ValueError(f"expect needs 'tools' or 'any_of': {expect!r}")
    wanted = set(expect["tools"])
    if not wanted:
        if calls:
            return "overcall", f"called {sorted(called)}, expected no call"
        return "correct", "no tool called"
    if not calls:
        return "missed", f"no tool called, expected {sorted(wanted)}"
    if called != wanted:
        parts = []
        if wanted - called:
            parts.append(f"missing {sorted(wanted - called)}")
        if called - wanted:
            parts.append(f"extra {sorted(called - wanted)}")
        return "missed", ", ".join(parts)
    specs = expect.get("args", {})
    for name in sorted(wanted):
        reasons = [args_mismatch(specs.get(name, {}), args) for tool, args in calls if tool == name]
        if all(reasons):  # no call of this tool satisfied the arguments; duplicates may differ
            return "wrong_args", f"{name}: {reasons[0]}"
    return "correct", f"called {sorted(called)}"


def score(expect: dict, calls: list[tuple[str, str]]) -> dict:
    """Judge one run's tool calls against `expect`.

    `expect` is one of:
    - `{"tools": [...], "args": {tool: {key: value}}}`: exactly these tools are called (order and
      repeats do not matter); `args` is an optional per-tool subset match;
    - `{"tools": []}`: no tool may be called;
    - `{"any_of": [...]}`: at least one call, and every call is to one of these tools.

    Returns `verdict` (correct, missed, overcall or wrong_args), `reason`, and `duplicate`: the same
    tool called twice with the same arguments, reported in addition to the verdict.
    """
    verdict, reason = _verdict(expect, calls)
    return {"verdict": verdict, "reason": reason, "duplicate": _has_duplicate(calls)}


def _percent(count: int, total: int) -> float:
    return round(100 * count / total, 1) if total else 0.0


def aggregate(records: list[dict]) -> dict:
    """Summary numbers for the runs of one path.

    A record has `error` (text or None), `score` (the `score` result, absent for an errored run) and
    `latency_s` (seconds to the first audio or to the decision; may be None). Percentages are of all
    runs, errors included; latency covers the runs that produced one.
    """
    total = len(records)
    counts = dict.fromkeys(OUTCOMES, 0)
    for record in records:
        counts["error" if record.get("error") else record["score"]["verdict"]] += 1
    duplicates = sum(1 for r in records if not r.get("error") and r["score"]["duplicate"])
    latencies = [r["latency_s"] for r in records if not r.get("error") and r.get("latency_s") is not None]
    return {
        "runs": total,
        "counts": counts,
        "percent": {name: _percent(n, total) for name, n in counts.items()},
        "duplicate": duplicates,
        "duplicate_percent": _percent(duplicates, total),
        "latency_median_s": statistics.median(latencies) if latencies else None,
        "latency_max_s": max(latencies) if latencies else None,
    }


def per_case(records: list[dict]) -> dict[str, tuple[int, int]]:
    """`case id -> (correct runs, all runs)`; records carry `case` as well as the fields `aggregate` reads."""
    table: dict[str, list[int]] = {}
    for record in records:
        cell = table.setdefault(record["case"], [0, 0])
        cell[1] += 1
        if not record.get("error") and record["score"]["verdict"] == "correct":
            cell[0] += 1
    return {case: (correct, total) for case, (correct, total) in table.items()}


def normalize_transcript(text: str) -> str:
    """Case, punctuation and spacing ignored: what two transcripts of the same words share."""
    return " ".join(re.sub(r"[^\w\s]", "", text.casefold()).split())


def misheard(said: str, heard: str) -> bool:
    """Whether the transcript differs from the sentence that was spoken, ignoring case and punctuation."""
    return normalize_transcript(said) != normalize_transcript(heard)
