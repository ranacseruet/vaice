"""Skills: small checks of and actions on this Mac that the assistant path runs itself.

The brain (a chat model) asks for a skill by name; vaice runs it with vaice's own permissions and hands
the model a short plain-text result to answer from. A skill never goes through a shell (argv lists
only; `run_shell`, see `shell.py`, is the opt-in exception) and never raises: whatever goes wrong comes back as a
`SkillResult` that says so in plain words, which is what the model then tells the user. macOS only, except
`disk_space`.

A skill has a `kind`: `read` checks something, `action` changes something that is easy to undo (an app
opens, the volume moves) and runs at once, `confirm` changes something that is not (an app quits) and runs
only after the user approved the question its `ask` built, which vaice holds (see `confirm.py`). The result
of an action names its object ("opened Safari"): history replays past calls without their arguments, so a
follow-up such as "now close it" can only be answered from the result's text.
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
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vaice.memory import Memory

log = logging.getLogger(__name__)

COMMAND_TIMEOUT = 10  # seconds a command may run
GIB = 1024**3
MAX_FILES = 20  # find_files stops listing here
SPOKEN_FILES = 3  # file names put into the spoken answer
MAX_PROCESSES = 5
MAX_APP_NAME = 60
APP_NAME_PUNCTUATION = " .&+'-"  # besides letters and digits: enough for "Google Chrome", "AT&T", "Notes 2"
VOLUME_STEP = 10  # points for volume_up / volume_down


@dataclass(frozen=True)
class SkillResult:
    """`speech`: short plain text for the model to answer from (no paths, no raw tables).
    `detail`: what the UI's Tool activity panel shows (paths, raw output); empty means show `speech`.
    `answer`: when set, the reply ends with exactly this sentence and the model is not asked again (see
    `SkillError`)."""

    speech: str
    detail: str = ""
    answer: str = ""


@dataclass(frozen=True)
class Question:
    """What to ask the user before a `confirm` skill runs: built only from the skill's validated arguments.
    `voice`: whether a spoken yes may approve it; False means only the UI's Run button can."""

    text: str
    voice: bool = True


@dataclass(frozen=True)
class Skill:
    name: str
    description: str  # one sentence: when to use it
    parameters: dict  # JSON schema of the arguments
    run: Callable[[dict], SkillResult]
    kind: str = "read"  # "read" | "action" (runs at once, easy to undo) | "confirm" (asks first)
    # confirm only: validates the arguments and returns the question to speak, or raises SkillError. The
    # question must not hold a word the user answers with (see confirm.py): the speakers' echo can reach the mic.
    # A `str` is a question a spoken yes may approve; a `Question` says so itself.
    ask: Callable[[dict], str | Question] | None = None
    # A skill that can be stopped (barge-in, the app ending) takes the reply's `stop` callable: `Skills.run` calls
    # this in place of `run` when it has one. `run` stays the fallback that cannot be stopped.
    run_stoppable: Callable[[dict, Callable[[], bool]], SkillResult] | None = None
    # A skill that must not act on what a tool reported (it stores text the model chose, e.g. `remember`): the
    # reply loop refuses it once another skill has run in the same reply. See `responders.assistant.decide`.
    direct_only: bool = False


class SkillError(Exception):
    """A skill could not do its job for a reason worth saying aloud (the message is spoken).

    `answer`: a fixed sentence that ends the reply instead of the model's paraphrase of the failure, for a
    refusal the model was seen misreporting (a refused reminder announced as "I've noted it")."""

    def __init__(self, message: str, *, answer: str = ""):
        super().__init__(message)
        self.answer = answer


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

    def direct_only(self, name: str) -> bool:
        """Whether skill `name` is `direct_only` (an unknown name is not)."""
        skill = self._by_name.get(name)
        return skill is not None and skill.direct_only

    def _resolve(self, name: str, arguments: str) -> tuple[Skill, dict, str] | SkillResult:
        """The skill, its parsed arguments and its label; or the result that says why that is not possible."""
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
        return skill, args, label

    def prepare(self, name: str, arguments: str) -> Question | SkillResult | None:
        """The checks that come before a `confirm` skill runs: the `Question` to ask the user, or the
        `SkillResult` that says why it cannot be done. None for any other skill (and an unknown one): `run`
        it, which reports its failures the same way. Never raises."""
        skill = self._by_name.get(name)
        if skill is None or skill.kind != "confirm":
            return None
        resolved = self._resolve(name, arguments)
        if isinstance(resolved, SkillResult):
            return resolved
        skill, args, label = resolved
        try:
            if skill.ask is None:
                raise SkillError("it has no question to ask")
            asked = skill.ask(args)
            return asked if isinstance(asked, Question) else Question(asked)
        except SkillError as exc:
            log.warning("skill %s cannot be asked about: %s", name, exc)
            return SkillResult(f"{label} failed: {exc}")
        except Exception as exc:
            log.exception("skill %s could not be asked about", name)
            return SkillResult(f"{label} failed: {type(exc).__name__}: {exc}")

    def run(self, name: str, arguments: str, stop: Callable[[], bool] | None = None) -> SkillResult:
        """Run skill `name` with `arguments` (the model's JSON string). Never raises. A `confirm` skill runs
        here too: the caller holds the user's approval (see `prepare`). `stop` ends a skill that can be stopped."""
        resolved = self._resolve(name, arguments)
        if isinstance(resolved, SkillResult):
            return resolved
        skill, args, label = resolved
        try:
            if skill.run_stoppable is not None:
                return skill.run_stoppable(args, stop or (lambda: False))
            return skill.run(args)
        except SkillError as exc:
            log.warning("skill %s failed: %s", name, exc)
            return SkillResult(f"{label} failed: {exc}", answer=exc.answer)
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


# --- open_app, quit_app


def _app_name(args: dict) -> str:
    """The validated app name. It reaches `open -a` and (as an argument, never as source) osascript, so
    nothing that could read as an option, a path or AppleScript gets through."""
    name = str(args.get("name", "")).strip()
    if not name:
        raise SkillError("no app name was given")
    if len(name) > MAX_APP_NAME:
        raise SkillError(f"that app name is longer than {MAX_APP_NAME} characters")
    if name.startswith("-") or not all(c.isalnum() or c in APP_NAME_PUNCTUATION for c in name):
        raise SkillError(f"{name!r} is not an app name I can use")
    return name


def _open_app(args: dict) -> SkillResult:
    _require_macos("opening an app")
    name = _app_name(args)
    try:
        _command(["open", "-a", name])
    except SkillError as exc:
        if "Unable to find application" in str(exc):
            raise SkillError(f"there is no app called {name}") from None
        raise
    return SkillResult(f"opened {name}")


_APP_HEADER = re.compile(r'^\s*\d+\) "(.*)" ASN:')


def parse_foreground_apps(output: str) -> list[str]:
    """The names of the running apps with a Dock icon, from `lsappinfo list` (one block per process, its
    header line `38) "Google Chrome" ASN:0x0-...:`, a `type="Foreground"` line inside)."""
    names: list[str] = []
    current: str | None = None
    for line in output.splitlines():
        header = _APP_HEADER.match(line)
        if header:
            current = header.group(1)
        elif current is not None and 'type="Foreground"' in line:
            names.append(current)
            current = None
    return names


def _running_app(name: str) -> str:
    """The running app's own name for `name` (case ignored). `lsappinfo` rather than AppleScript's `is running`:
    asked about an app that does not exist, that hangs on a "Where is <name>?" chooser (checked on macOS 26)."""
    for running in parse_foreground_apps(_command(["lsappinfo", "list"])):
        if running.casefold() == name.casefold():
            return running
    raise SkillError(f"{name} is not running")


def _ask_quit(args: dict) -> str:
    _require_macos("quitting an app")
    return f"Shall I quit {_running_app(_app_name(args))}?"


def _quit_app(args: dict) -> SkillResult:
    _require_macos("quitting an app")
    name = _running_app(_app_name(args))  # `quit` on an app that is not running would launch it first
    _command(["osascript", "-e", "on run argv", "-e", "tell application (item 1 of argv) to quit",
              "-e", "end run", name])
    return SkillResult(f"quit {name}")


# --- set_volume, mute_sound


def _whole_number(value, what: str) -> int:
    """`value` as an int (a bool, a fraction or text that is not a whole number is refused)."""
    if isinstance(value, bool):
        raise SkillError(f"{what} must be a whole number")
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value)
    raise SkillError(f"{what} must be a whole number, not {value!r}")


def _output_volume() -> int:
    output = _command(["osascript", "-e", "output volume of (get volume settings)"]).strip()
    try:
        return int(output)
    except ValueError:  # "missing value": a display or other output with no volume of its own
        raise SkillError("the volume of this output cannot be read") from None


def _apply_volume(volume: int) -> SkillResult:
    _command(["osascript", "-e", f"set volume output volume {volume}"])  # an int: safe to interpolate
    return SkillResult(f"volume set to {volume} percent")


def _set_volume(args: dict) -> SkillResult:
    _require_macos("changing the volume")
    volume = _whole_number(args.get("level"), "the level")
    if not 0 <= volume <= 100:
        raise SkillError(f"the level must be from 0 to 100, not {volume}")
    return _apply_volume(volume)


def _step_volume(step: int) -> SkillResult:
    _require_macos("changing the volume")
    return _apply_volume(max(0, min(100, _output_volume() + step)))


def _mute_sound(args: dict) -> SkillResult:
    _require_macos("changing the sound")
    muted = args.get("muted", True)
    if isinstance(muted, str) and muted.strip().lower() in ("true", "false"):  # a small model may quote it
        muted = muted.strip().lower() == "true"
    if not isinstance(muted, bool):
        raise SkillError(f"muted must be true or false, not {muted!r}")
    _command(["osascript", "-e", f"set volume output muted {'true' if muted else 'false'}"])
    return SkillResult("sound muted" if muted else "sound unmuted")


def default_skills(shell: bool = False, memory: "Memory | None" = None) -> Skills:
    """The skills the assistant path offers: five read-only checks, then the actions, with `shell`
    (`VAICE_ASSISTANT_SHELL`) `run_shell`, which asks first, and with a `memory` (`VAICE_ASSISTANT_MEMORY`) the
    skills that keep and forget notes (see `memory.py`)."""
    nothing = {"type": "object", "properties": {}}
    app_name = {"type": "object",
                "properties": {"name": {"type": "string",
                                        "description": "The app's name, such as Safari, Notes or Calculator."}},
                "required": ["name"]}
    skills = [
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
        Skill("open_app", "Open an app on this Mac by name.", app_name, _open_app, kind="action"),
        Skill("quit_app", "Quit a running app by name. The user is asked to confirm first.", app_name, _quit_app,
              kind="confirm", ask=_ask_quit),
        Skill("set_volume", "Set the speaker volume to the number the user said, from 0 to 100.",
              {"type": "object",
               "properties": {"level": {"type": "integer", "minimum": 0, "maximum": 100,
                                        "description": "The new volume, 0 to 100."}},
               "required": ["level"]},
              _set_volume, kind="action"),
        # Up and down are skills of their own, without arguments: as a `direction` argument of set_volume the
        # 2B model set a guessed level instead, or asked how much (live eval, 0 of 5 either way)
        Skill("volume_up", "Turn the speaker volume up (louder).", nothing,
              lambda _args: _step_volume(VOLUME_STEP), kind="action"),
        Skill("volume_down", "Turn the speaker volume down (quieter).", nothing,
              lambda _args: _step_volume(-VOLUME_STEP), kind="action"),
        Skill("mute_sound", "Mute the sound, or unmute it.",
              {"type": "object",
               "properties": {"muted": {"type": "boolean", "description": "true to mute, false to unmute."}}},
              _mute_sound, kind="action"),
    ]
    if shell:
        from vaice.shell import shell_skill  # shell imports this module: not at the top

        skills.append(shell_skill())
    if memory is not None:
        from vaice.memory import memory_skills  # memory imports this module: not at the top

        skills += memory_skills(memory)
    return Skills(skills)
