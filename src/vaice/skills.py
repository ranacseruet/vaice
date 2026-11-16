"""Skills: small read-only checks of this Mac that the assistant path runs itself.

The brain (a chat model) asks for a skill by name; vaice runs it with vaice's own permissions and hands
the model a short plain-text result to answer from. A skill never changes anything, never goes through a
shell (argv lists only), and never raises: whatever goes wrong comes back as a `SkillResult` that says so
in plain words, which is what the model then tells the user. macOS only, except `disk_space`.
"""
import json
import logging
import platform
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

COMMAND_TIMEOUT = 10  # seconds a command may run
GIB = 1024**3
MAX_FILES = 20  # find_files stops listing here
SPOKEN_FILES = 3  # file names put into the spoken answer
MAX_PROCESSES = 5


@dataclass(frozen=True)
class SkillResult:
    """`speech`: short plain text for the model to answer from (no paths, no raw tables).
    `detail`: what the UI's Tool activity panel shows (paths, raw output); empty means show `speech`."""

    speech: str
    detail: str = ""


@dataclass(frozen=True)
class Skill:
    name: str
    description: str  # one sentence: when to use it
    parameters: dict  # JSON schema of the arguments
    run: Callable[[dict], SkillResult]


class SkillError(Exception):
    """A skill could not do its job for a reason worth saying aloud (the message is spoken)."""


class Skills:
    """A set of skills: their chat-completions tool definitions, and a `run` that never raises."""

    def __init__(self, skills: Sequence[Skill]):
        self.skills = tuple(skills)
        self._by_name = {skill.name: skill for skill in self.skills}

    @property
    def names(self) -> list[str]:
        return [skill.name for skill in self.skills]

    def definitions(self) -> list[dict]:
        """The skills as `tools` for POST /v1/chat/completions."""
        return [{"type": "function",
                 "function": {"name": s.name, "description": s.description, "parameters": s.parameters}}
                for s in self.skills]

    def run(self, name: str, arguments: str) -> SkillResult:
        """Run skill `name` with `arguments` (the model's JSON string). Never raises."""
        skill = self._by_name.get(name)
        if skill is None:
            return SkillResult(f"there is no skill called {name}")
        label = f"the {name.replace('_', ' ')} skill"
        try:
            args = json.loads(arguments) if arguments.strip() else {}
        except ValueError:
            return SkillResult(f"{label} failed: its arguments are not valid JSON")
        if not isinstance(args, dict):
            return SkillResult(f"{label} failed: its arguments are not a JSON object")
        for key in skill.parameters.get("required", []):
            if key not in args:
                return SkillResult(f"{label} failed: the argument {key} is missing")
        try:
            return skill.run(args)
        except SkillError as exc:
            log.warning("skill %s failed: %s", name, exc)
            return SkillResult(f"{label} failed: {exc}")
        except Exception as exc:
            log.exception("skill %s failed", name)
            return SkillResult(f"{label} failed: {type(exc).__name__}: {exc}")


# --- helpers shared by the skills


def _require_macos(what: str) -> None:
    if sys.platform != "darwin":
        raise SkillError(f"{what} is only available on macOS")


def _command(argv: list[str]) -> str:
    """stdout of `argv` (never a shell, no stdin, `COMMAND_TIMEOUT`)."""
    try:
        done = subprocess.run(argv, capture_output=True, text=True, stdin=subprocess.DEVNULL,
                              timeout=COMMAND_TIMEOUT, check=False)
    except FileNotFoundError:
        raise SkillError(f"{argv[0]} is not available") from None
    except subprocess.TimeoutExpired:
        raise SkillError(f"{argv[0]} did not answer within {COMMAND_TIMEOUT} seconds") from None
    if done.returncode != 0:
        raise SkillError(f"{argv[0]} failed with status {done.returncode}: {done.stderr.strip()[:100]}")
    return done.stdout


def _plural(count: int, word: str) -> str:
    return f"{count} {word}" if count == 1 else f"{count} {word}s"


def spoken_duration(seconds: float) -> str:
    """`seconds` as its two largest units: "3 days and 4 hours", "4 hours and 12 minutes"."""
    minutes = int(seconds // 60)
    days, rest = divmod(minutes, 1440)
    hours, minutes = divmod(rest, 60)
    units = [(days, "day"), (hours, "hour"), (minutes, "minute")]
    parts = [_plural(count, word) for count, word in units if count][:2]
    return " and ".join(parts) or "less than a minute"


# --- disk_space


def describe_disk(total: int, free: int) -> str:
    return f"{round(free / GIB)} gigabytes free of {round(total / GIB)} gigabytes"


def _disk_space(_args: dict) -> SkillResult:
    usage = shutil.disk_usage("/")
    return SkillResult(describe_disk(usage.total, usage.free))


# --- battery_status

_BATTERY = re.compile(r"(\d+)%;([^\n]*?)\s*(?:present:|$)", re.M)
_REMAINING = re.compile(r"(\d+):(\d{2}) remaining")


def parse_battery(output: str) -> str:
    """`pmset -g batt` output as a spoken sentence; "no battery" for a desktop."""
    match = _BATTERY.search(output)
    on_power = "AC Power" in output
    if match is None:
        return "this Mac has no battery" if on_power else "no battery was found"
    percent, status = match.group(1), match.group(2).lower()  # "discharging; 4:12 remaining"
    charging = False
    if "discharging" in status:
        phrase = "running on battery power"
    elif "finishing charge" in status:
        phrase, charging = "almost fully charged", True
    elif "charged" in status:
        phrase = "fully charged and plugged in" if on_power else "fully charged"
    elif "not charging" in status:
        phrase = "plugged in but not charging"
    elif "charging" in status:
        phrase, charging = "charging", True
    else:
        phrase = "plugged in" if on_power else "running on battery power"
    speech = f"battery at {percent} percent, {phrase}"
    remaining = _REMAINING.search(status)
    if remaining and (int(remaining.group(1)) or int(remaining.group(2))):
        left = spoken_duration(int(remaining.group(1)) * 3600 + int(remaining.group(2)) * 60)
        speech += f", about {left} {'until full' if charging else 'left'}"
    return speech


def _battery_status(_args: dict) -> SkillResult:
    _require_macos("the battery check")
    output = _command(["pmset", "-g", "batt"])
    return SkillResult(parse_battery(output), output.strip())


# --- system_info

_BOOTTIME = re.compile(r"\bsec\s*=\s*(\d+)")


def parse_boot_time(output: str) -> int:
    """Epoch seconds of the boot, from `sysctl -n kern.boottime` (`{ sec = 1731000000, usec = 0 } ...`)."""
    match = _BOOTTIME.search(output)
    if match is None:
        raise SkillError(f"unexpected kern.boottime: {output.strip()[:100]!r}")
    return int(match.group(1))


def parse_memory(output: str) -> int:
    """Bytes of memory, from `sysctl -n hw.memsize`."""
    try:
        return int(output.strip())
    except ValueError:
        raise SkillError(f"unexpected hw.memsize: {output.strip()[:100]!r}") from None


def describe_system(version: str, name: str, boot: int, memory: int, now: float) -> str:
    name = name.removesuffix(".local")
    up = spoken_duration(max(0.0, now - boot))
    return f"macOS {version} on {name}, up {up}, {round(memory / GIB)} gigabytes of memory"


def _system_info(_args: dict) -> SkillResult:
    _require_macos("the system check")
    boot = parse_boot_time(_command(["sysctl", "-n", "kern.boottime"]))
    memory = parse_memory(_command(["sysctl", "-n", "hw.memsize"]))
    return SkillResult(describe_system(platform.mac_ver()[0], platform.node(), boot, memory, time.time()))


# --- top_processes


def parse_processes(output: str) -> list[tuple[float, float, str]]:
    """(cpu percent, memory percent, name) per row of `ps -Aceo pcpu,pmem,comm`; the header is skipped."""
    rows = []
    for line in output.splitlines()[1:]:
        parts = line.split(None, 2)  # the name may hold spaces ("Google Chrome")
        if len(parts) < 3:
            continue
        try:
            rows.append((float(parts[0]), float(parts[1]), parts[2].strip()))
        except ValueError:
            continue
    return rows


_TRAILING_PARENTHETICAL = re.compile(r"\s*\([^()]*\)\s*$")


def speakable_process_name(name: str) -> str:
    """A process name as it can be spoken: no trailing "(Renderer)", a helper counted with its app
    ("Google Chrome Helper" -> "Google Chrome"), a reverse-DNS name as its last part
    ("com.apple.Virtualization.VirtualMachine" -> "VirtualMachine"). The raw name stays in the detail."""
    cleaned = _TRAILING_PARENTHETICAL.sub("", name).strip()
    parts = cleaned.split(".")
    if len(parts) >= 3 and not any(" " in part or not part for part in parts):
        cleaned = parts[-1]
    cleaned = re.sub(r"\s+Helper$", "", cleaned)
    return cleaned or name


def describe_processes(rows: list[tuple[float, float, str]], by: str, count: int) -> str:
    """The top `count` names by memory or cpu. Helper processes share a name (a browser runs dozens),
    so rows are added up by their speakable name, then sorted by that total."""
    totals: dict[str, float] = {}
    for cpu, memory, raw_name in rows:
        name = speakable_process_name(raw_name)
        totals[name] = totals.get(name, 0.0) + (memory if by == "memory" else cpu)
    top = sorted(totals.items(), key=lambda item: item[1], reverse=True)[:count]
    if not top:
        return "no processes were listed"
    parts = [f"{name} {round(value)} percent" for name, value in top]
    parts[0] += " of memory" if by == "memory" else " of the processor"
    return ", ".join(parts)


def _top_processes(args: dict) -> SkillResult:
    _require_macos("the process list")
    by = str(args.get("by", "memory")).strip().lower()
    if by not in ("memory", "cpu"):
        raise SkillError(f"by must be memory or cpu, not {by!r}")
    try:
        count = int(args.get("count", 3))
    except (TypeError, ValueError):
        raise SkillError(f"count must be a whole number, not {args.get('count')!r}") from None
    count = max(1, min(MAX_PROCESSES, count))
    output = _command(["ps", "-Aceo", "pcpu,pmem,comm", "-m" if by == "memory" else "-r"])
    rows = parse_processes(output)
    raw = "\n".join(f"{cpu:.1f}% cpu  {memory:.1f}% mem  {name}" for cpu, memory, name in rows[:10])
    return SkillResult(describe_processes(rows, by, count), raw)


# --- find_files

FOLDERS = ("home", "desktop", "documents", "downloads")


def _folder_path(folder: str) -> Path:
    home = Path.home()
    return home if folder == "home" else home / folder.capitalize()


def describe_files(output: str) -> SkillResult:
    """`mdfind` output (one path per line) as a count and up to 3 file names; the paths go in `detail`."""
    paths = [line.strip() for line in output.splitlines() if line.strip()]
    if not paths:
        return SkillResult("no files found")
    more = len(paths) > MAX_FILES
    paths = paths[:MAX_FILES]
    names = [Path(path).name for path in paths[:SPOKEN_FILES]]
    count = f"more than {MAX_FILES} files" if more else _plural(len(paths), "file")
    joiner = ": " if len(paths) <= SPOKEN_FILES and not more else ", including "
    return SkillResult(f"{count}{joiner}{', '.join(names)}", "\n".join(paths))


def _find_files(args: dict) -> SkillResult:
    _require_macos("the file search")
    name = str(args.get("name", "")).strip()
    if not name:
        raise SkillError("no file name to look for")
    folder = str(args.get("folder", "home")).strip().lower()
    if folder not in FOLDERS:
        raise SkillError(f"folder must be one of {', '.join(FOLDERS)}, not {folder!r}")
    output = _command(["mdfind", "-name", name, "-onlyin", str(_folder_path(folder))])
    return describe_files(output)


def default_skills() -> Skills:
    """The read-only skills the assistant path offers."""
    nothing = {"type": "object", "properties": {}}
    return Skills([
        Skill("disk_space", "Report how much disk space is free on this Mac.", nothing, _disk_space),
        Skill("battery_status", "Report the battery level and whether this Mac is charging or plugged in.",
              nothing, _battery_status),
        Skill("system_info", "Report the macOS version, the computer's name, how long it has been on, and how much "
                             "memory it has.", nothing, _system_info),
        Skill("top_processes", "List the apps and processes using the most memory or processor time right now.",
              {"type": "object",
               "properties": {
                   "by": {"type": "string", "enum": ["memory", "cpu"], "description": "What to rank by."},
                   "count": {"type": "integer", "minimum": 1, "maximum": MAX_PROCESSES,
                             "description": "How many to list (1 to 5, default 3)."}}},
              _top_processes),
        Skill("find_files", "Find files by name in the user's home, desktop, documents or downloads folder.",
              {"type": "object",
               "properties": {
                   "name": {"type": "string", "description": "Part of the file name to look for."},
                   "folder": {"type": "string", "enum": list(FOLDERS),
                              "description": "Where to look (default home)."}},
               "required": ["name"]},
              _find_files),
    ])
