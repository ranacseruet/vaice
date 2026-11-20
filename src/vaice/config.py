"""Settings, overridable through environment variables."""
import os
import shlex
from collections.abc import Mapping
from pathlib import Path
from urllib.parse import urlsplit

from dotenv import load_dotenv

# A .env in the working directory (where `vaice` is launched from) fills in unset variables at
# import; variables already in the environment (exported, or from `set -a; source .env`) win, and
# built-in defaults come last.
load_dotenv(dotenv_path=Path.cwd() / ".env")


def _env(name: str, *, default: str) -> str:
    raw = os.getenv(name)
    return default if raw is None else raw


def _number(name: str, *, default: str, kind: type[int] | type[float]) -> int | float:
    raw = os.getenv(name)
    if raw is None:
        return kind(default)
    try:
        return kind(raw)
    except ValueError:
        raise SystemExit(f"vaice: {name} must be a number ({kind.__name__}), got {raw!r}") from None


def _flag(name: str, *, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    if raw.lower() in ("1", "true", "yes", "on"):
        return True
    if raw.lower() in ("0", "false", "no", "off"):
        return False
    raise SystemExit(f"vaice: {name} must be a boolean (1/true/yes/on or 0/false/no/off), got {raw!r}")


def _choice(name: str, *, default: str, choices: tuple[str, ...]) -> str:
    raw = os.getenv(name)
    if raw is None:
        return default
    if raw not in choices:
        raise SystemExit(f"vaice: {name} must be {' or '.join(choices)}, got {raw!r}")
    return raw


def server_base(url: str) -> str:
    """scheme://host:port of `url`: any path suffix is dropped. Raises ValueError if there is no host."""
    parts = urlsplit(url if "://" in url else f"http://{url}")
    if parts.scheme not in ("http", "https") or not parts.netloc:
        raise ValueError("expected an http(s) URL like http://localhost:8001")
    return f"{parts.scheme}://{parts.netloc}"


def _server_url(name: str, *, default: str) -> str:
    raw = os.getenv(name)
    if raw is None:
        return server_base(default)
    try:
        return server_base(raw)
    except ValueError as exc:
        raise SystemExit(f"vaice: {name} must be a URL ({exc}), got {raw!r}") from None


# llama-server's /v1/realtime. The mode line that tells the model to write its text alongside the speech is added in
# code (responders/llamacpp.py), so VAICE_PROMPT only has to say how to answer.
LLAMACPP_URL = _server_url("VAICE_LLAMACPP_URL", default="http://localhost:8001")
LLAMACPP_TIMEOUT = _number("VAICE_LLAMACPP_TIMEOUT", default="120", kind=float)

# Reply cap sent as the realtime session's max_output_tokens; text tokens and audio frames count
# together (the LFM3-Audio codec emits 12 tokens per 80 ms of speech). The model card's own
# voice-chat example uses 1024; llama-server's session default would be "inf".
LLAMACPP_MAX_OUTPUT_TOKENS = _number("VAICE_LLAMACPP_MAX_OUTPUT_TOKENS", default="1024", kind=int)

# Tool results are fed back to the model as text, and it tends to read them aloud. Each string value
# and the whole compacted result are capped: too small and an answer-bearing field is cut off; too large and the
# model dictates the raw result. The defaults keep the answer while leaving the dictation short.
LLAMACPP_TOOL_VALUE_CHARS = _number("VAICE_LLAMACPP_TOOL_VALUE_CHARS", default="200", kind=int)
LLAMACPP_TOOL_OUTPUT_CHARS = _number("VAICE_LLAMACPP_TOOL_OUTPUT_CHARS", default="800", kind=int)
if LLAMACPP_TOOL_VALUE_CHARS <= 0 or LLAMACPP_TOOL_OUTPUT_CHARS <= 0:
    raise SystemExit("vaice: VAICE_LLAMACPP_TOOL_VALUE_CHARS and VAICE_LLAMACPP_TOOL_OUTPUT_CHARS must be > 0")

# `vaice up` starts llama-server when none answers at LLAMACPP_URL: the `-hf` model, extra command-line
# arguments (e.g. `--tools get_info,read_file`), and how long to wait for it (the first run downloads
# about 2.4 GB). Plain `vaice` never starts a server.
LLAMACPP_MODEL = _env("VAICE_LLAMACPP_MODEL", default="LiquidAI/LFM3-Omni-2B-GGUF:Q8_0")


def _arguments(name: str) -> list[str]:
    raw = os.getenv(name, "")
    try:
        return shlex.split(raw)
    except ValueError as exc:
        raise SystemExit(f"vaice: {name} must be a shell-style argument list ({exc}), got {raw!r}") from None


LLAMACPP_SERVER_ARGS = _arguments("VAICE_LLAMACPP_SERVER_ARGS")
LLAMACPP_START_TIMEOUT = _number("VAICE_LLAMACPP_START_TIMEOUT", default="900", kind=float)

# The system prompt of every reply. Its lines were measured with the live eval: do not add or reword one without a
# live A/B (see evals/README.md).
PROMPT = _env(
    "VAICE_PROMPT",
    default="You are a voice assistant on the user's Mac. Your reply is spoken aloud: answer in one or two short "
            "plain sentences, with no lists, symbols or file paths. Use a tool only when the question is about "
            "this computer; answer everything else yourself. The computer changes all the time: for each question "
            "about the screen, the battery, the disk or other live facts, use the tool again, even if you answered "
            "the same kind of question earlier.",
)
# Adds the `run_shell` skill (shell.py). Every command is approved first, by the UI's Run button. Off: the skill
# does not exist.
SHELL = _flag("VAICE_SHELL", default=False)
# The assistant keeps short notes you ask it to remember ("remember that I park on level three") in
# <VAICE_DATA_DIR>/memory.json, across conversations, and puts them into every prompt. Off: no memory skills and no
# notes in the prompt (an existing file is left alone).
# Off by default: the memory skills cost points on ordinary requests (see evals/README.md).
MEMORY = _flag("VAICE_MEMORY", default=False)


LOCAL_HOSTS = ("localhost", "127.0.0.1", "::1")  # an endpoint at one of these hosts does not leave this machine


def endpoint_base(raw: str, key_variable: str) -> str:
    """An OpenAI-compatible endpoint's base URL as OpenAI clients take it (the vision model's, the browser agent's),
    `/v1` included, without a trailing slash. `key_variable` is named in the error for credentials in the URL.

    Raises ValueError with a message that never repeats `raw`: a key pasted into a URL must not reach the log."""
    parts = urlsplit(raw if "://" in raw else f"http://{raw}")
    if parts.scheme not in ("http", "https") or not parts.netloc:
        raise ValueError("expected an http(s) URL like http://localhost:1234/v1")
    if "@" in parts.netloc or parts.query or parts.fragment:
        raise ValueError(f"it must not contain credentials, a query or a fragment (the API key goes in "
                         f"{key_variable})")
    path = parts.path.rstrip("/")
    if path.endswith("/chat/completions"):
        raise ValueError("give the base URL (e.g. http://localhost:1234/v1), not the endpoint: vaice adds "
                         "/chat/completions itself")
    return f"{parts.scheme}://{parts.netloc}{path}"


def _endpoint_url(name: str) -> str:
    """`endpoint_base` of the variable; "" (unset or empty) = not configured. Its key variable is the `_API_KEY`
    sibling."""
    raw = os.getenv(name, "").strip()
    if not raw:
        return ""
    try:
        return endpoint_base(raw, name.removesuffix("_URL") + "_API_KEY")
    except ValueError as exc:
        raise SystemExit(f"vaice: {name}: {exc}") from None


# The `look_at_screen` skill (vision.py) takes a screenshot when the user asks about their
# screen and asks this vision model about it: any OpenAI-compatible endpoint that accepts images (LFM3-Omni-2B on
# a llama-server of its own, a hosted model). Unset = no skill, no screenshot is ever taken. The screenshot is
# sent to that endpoint, so a host off this machine receives an image of the screen (see docs/privacy.md). The key
# is a secret: never logged.
VISION_URL = _endpoint_url("VAICE_VISION_URL")
VISION_MODEL = _env("VAICE_VISION_MODEL", default="").strip()
VISION_API_KEY = _env("VAICE_VISION_API_KEY", default="").strip()
VISION_TIMEOUT = _number("VAICE_VISION_TIMEOUT", default="30", kind=float)
if VISION_TIMEOUT <= 0:
    raise SystemExit("vaice: VAICE_VISION_TIMEOUT must be > 0")
# Upper bound on the vision model's answer, in tokens (0 = send none). Without one, LFM3-Omni-2B described a busy
# screen at 400 to 830 tokens (5 to 9 s) and sometimes ran on until VISION_TIMEOUT ("I can't reach my vision
# model"); only the first MAX_ANSWER_CHARS of the answer are used anyway. Some hosted models reject `max_tokens`
# (they want `max_completion_tokens`): set 0 for those.
VISION_MAX_TOKENS = _number("VAICE_VISION_MAX_TOKENS", default="200", kind=int)
if VISION_MAX_TOKENS < 0:
    raise SystemExit("vaice: VAICE_VISION_MAX_TOKENS must be >= 0")

# MCP (Model Context Protocol) tools as skills (mcp_skills.py). A JSON file in Claude Desktop's
# `mcpServers` shape names the stdio servers to start and, per server, the tools to offer and how each is approved
# (`run` or `click`; click by default). Unset = no MCP, and the optional `mcp` package is never imported.
# Needs `uv sync --extra mcp`. The file may hold tokens (a server's `env`): vaice never logs them. Each tool call
# may take at most MCP_TIMEOUT seconds; a server must also start within it.
_MCP_CONFIG = _env("VAICE_MCP_CONFIG", default="").strip()
MCP_CONFIG = Path(_MCP_CONFIG).expanduser() if _MCP_CONFIG else None
MCP_TIMEOUT = _number("VAICE_MCP_TIMEOUT", default="30", kind=float)
if not MCP_TIMEOUT > 0:
    raise SystemExit("vaice: VAICE_MCP_TIMEOUT must be > 0")

# The `browse` skill (browse.py, browse_task.py), a task in a browser run by the Pi coding agent
# in the background. `off` (default) or `pi` (needs the `pi` command on PATH, version 1.5.2 or newer, and
# VAICE_BROWSE_URL: Pi has its own OpenAI-compatible endpoint, model and key). The browser is Chrome with a profile
# of its own under VAICE_DATA_DIR, in a visible window unless VAICE_BROWSE_HEADLESS is on. A task is stopped after
# VAICE_BROWSE_SECONDS. Pages' text and what the agent does go to that endpoint (see docs/privacy.md). The key is a
# secret: it reaches Pi's environment only, never logged.
BROWSE = _choice("VAICE_BROWSE", default="off", choices=("off", "pi"))
BROWSE_URL = _endpoint_url("VAICE_BROWSE_URL")
BROWSE_MODEL = _env("VAICE_BROWSE_MODEL", default="").strip()
BROWSE_API_KEY = _env("VAICE_BROWSE_API_KEY", default="").strip()
BROWSE_SECONDS = _number("VAICE_BROWSE_SECONDS", default="180", kind=float)
if not BROWSE_SECONDS > 0:
    raise SystemExit("vaice: VAICE_BROWSE_SECONDS must be > 0 (seconds)")
BROWSE_HEADLESS = _flag("VAICE_BROWSE_HEADLESS", default=False)
# `ask` (default): a step that changes a page (click, type, fill, run a script) needs the Run button first; `off`:
# none does.
BROWSE_CONFIRM = _choice("VAICE_BROWSE_CONFIRM", default="ask", choices=("ask", "off"))

PORT = _number("VAICE_PORT", default="7860", kind=int)

# The conversation is kept here between runs; the last HISTORY_TURNS exchanges go into each prompt.
# Set VAICE_DATA_DIR="" to keep the conversation in memory only: nothing is written, loaded, or kept.
_DATA_DIR = os.getenv("VAICE_DATA_DIR", "~/.vaice")
DATA_DIR = None if _DATA_DIR.strip() == "" else Path(_DATA_DIR).expanduser()
CONVERSATION_FILE = None if DATA_DIR is None else DATA_DIR / "conversation.json"
MEMORY_FILE = None if DATA_DIR is None else DATA_DIR / "memory.json"  # the notes
HISTORY_TURNS = _number("VAICE_HISTORY_TURNS", default="10", kind=int)

# fastrtc's Silero VAD: what counts as speech, and when an utterance ends. A barge-in fires when
# the interrupting sound ends, so the threshold decides what can interrupt a playing reply —
# raise it if room noise or speaker feedback cuts replies short. The min-silence gap delays both
# the reply start and interrupts; the speech pad pads detected speech on both sides.
VAD_THRESHOLD = _number("VAICE_VAD_THRESHOLD", default="0.6", kind=float)
VAD_MIN_SILENCE_MS = _number("VAICE_VAD_MIN_SILENCE_MS", default="150", kind=int)
VAD_SPEECH_PAD_MS = _number("VAICE_VAD_SPEECH_PAD_MS", default="30", kind=int)
if not 0 < VAD_THRESHOLD < 1:
    raise SystemExit(f"vaice: VAICE_VAD_THRESHOLD must be between 0 and 1 (exclusive), got {VAD_THRESHOLD}")
if VAD_MIN_SILENCE_MS < 0 or VAD_SPEECH_PAD_MS < 0:
    raise SystemExit("vaice: VAICE_VAD_MIN_SILENCE_MS and VAICE_VAD_SPEECH_PAD_MS must be >= 0")

# Whether speech can interrupt a playing reply at all. Off = a reply always runs to the end.
CAN_INTERRUPT = _flag("VAICE_CAN_INTERRUPT", default=True)


# Settings an earlier version read and this one ignores: name -> why. `removed_settings` finds the ones still set (in
# the environment or the `.env` loaded above) so `app.main()` can say they do nothing, instead of leaving the user
# to wonder why a hosted brain is no longer used. Only names are ever reported: a value may be a key.
_HELPER = "the helper brain was removed; every reply is one speech-to-speech session on the local llama-server"
REMOVED = {
    "VAICE_BRAIN_URL": _HELPER,
    "VAICE_BRAIN_MODEL": _HELPER,
    "VAICE_BRAIN_API_KEY": _HELPER,
    "VAICE_BRAIN_TIMEOUT": _HELPER,
    "VAICE_ASSISTANT_AUTO_SECONDS": "the MCP auto tier was removed; use run or click",
    "VAICE_MODE": "there is one mode now: the assistant, a speech-to-speech session with skills",
    "VAICE_LLAMACPP_PROMPT": "renamed to VAICE_PROMPT",
    "VAICE_ASSISTANT_PROMPT": "renamed to VAICE_PROMPT",
    "VAICE_ASSISTANT_SHELL": "renamed to VAICE_SHELL",
    "VAICE_ASSISTANT_MEMORY": "renamed to VAICE_MEMORY",
    "VAICE_LLAMACPP_TOOLS": "llama-server's own tools are not used; the assistant's skills run in vaice",
    "VAICE_VOICE": "there is no separate text-to-speech step; the voice is the speech-to-speech model's own",
    "VAICE_ASSISTANT_PROGRESS_SECONDS": "no spoken progress lines any more",
    "VAICE_ASSISTANT_MAX_STEPS": "a reply makes at most two tool rounds; there is no step budget",
    "VAICE_ASSISTANT_TASK_SECONDS": "a reply makes at most two tool rounds; there is no time budget",
}


def removed_settings(environ: Mapping[str, str] = os.environ) -> list[tuple[str, str]]:
    """The `REMOVED` settings that are set in `environ`, as (name, reason), in `REMOVED` order. Never their values."""
    return [(name, reason) for name, reason in REMOVED.items() if name in environ]
