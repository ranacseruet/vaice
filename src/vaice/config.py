"""Settings, overridable through environment variables."""
import os
import shlex
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


# llama-server's /v1/realtime. The "Respond with interleaved text and audio." mode line
# is added in code (responders/llamacpp.py), so VAICE_LLAMACPP_PROMPT only has to say how to answer.
LLAMACPP_URL = _server_url("VAICE_LLAMACPP_URL", default="http://localhost:8001")
LLAMACPP_PROMPT = _env(
    "VAICE_LLAMACPP_PROMPT",
    default="Reply naturally and concisely in 1 to 2 spoken sentences.",
)
LLAMACPP_TIMEOUT = _number("VAICE_LLAMACPP_TIMEOUT", default="120", kind=float)

# Reply cap sent as the realtime session's max_output_tokens; text tokens and audio frames count
# together (the LFM3-Audio codec emits 12 tokens per 80 ms of speech). The model card's own
# voice-chat example uses 1024; llama-server's session default would be "inf".
LLAMACPP_MAX_OUTPUT_TOKENS = _number("VAICE_LLAMACPP_MAX_OUTPUT_TOKENS", default="1024", kind=int)

# Tool results are fed back to the model as text, and it tends to read them aloud. Each string value
# and the whole compacted result are capped: too small and an answer-bearing field (the timestamp in
# `get_info`, the end of a file) is cut off; too large and the model dictates the raw result. The
# defaults keep the answer while leaving the dictation short.
LLAMACPP_TOOL_VALUE_CHARS = _number("VAICE_LLAMACPP_TOOL_VALUE_CHARS", default="200", kind=int)
LLAMACPP_TOOL_OUTPUT_CHARS = _number("VAICE_LLAMACPP_TOOL_OUTPUT_CHARS", default="800", kind=int)
if LLAMACPP_TOOL_VALUE_CHARS <= 0 or LLAMACPP_TOOL_OUTPUT_CHARS <= 0:
    raise SystemExit("vaice: VAICE_LLAMACPP_TOOL_VALUE_CHARS and VAICE_LLAMACPP_TOOL_OUTPUT_CHARS must be > 0")


# The tools `llama-server` makes available to the model in `session.update`: off (unset) or on
# ("builtin" = every tool the server advertises on GET /tools). The set of tools is the server
# operator's decision (`llama-server --tools ...`), which also gates what POST /tools executes —
# vaice takes the list as it finds it and does not filter it.
def _tools(name: str) -> bool | None:
    raw = os.getenv(name)
    if raw is None:
        return None
    if raw.strip() == "builtin":
        return True
    raise SystemExit(f"vaice: {name} is 'builtin' (tools on) or unset (tools off); to pick the tools, "
                     f"start llama-server with --tools <names>, got {raw!r}")


LLAMACPP_TOOLS = _tools("VAICE_LLAMACPP_TOOLS")

# `vaice up` starts llama-server when none answers at LLAMACPP_URL: the `-hf` model, extra command-line
# arguments (e.g. `--tools get_info,read_file`), and how long to wait for it (the first run downloads
# about 1 GB). Plain `vaice` never starts a server.
LLAMACPP_MODEL = _env("VAICE_LLAMACPP_MODEL", default="LiquidAI/LFM3-Audio-2B-GGUF:Q4_K_M")


def _arguments(name: str) -> list[str]:
    raw = os.getenv(name, "")
    try:
        return shlex.split(raw)
    except ValueError as exc:
        raise SystemExit(f"vaice: {name} must be a shell-style argument list ({exc}), got {raw!r}") from None


LLAMACPP_SERVER_ARGS = _arguments("VAICE_LLAMACPP_SERVER_ARGS")
LLAMACPP_START_TIMEOUT = _number("VAICE_LLAMACPP_START_TIMEOUT", default="900", kind=float)

# "chat": one model hears, decides and speaks in one interleaved pass (responders/llamacpp.py).
# "assistant": separate steps (transcribe, a chat-completions brain, skills run by vaice, llama-server's
# /tts) — see responders/assistant.py. The prompt and voice below apply to assistant mode only.
MODE = _choice("VAICE_MODE", default="chat", choices=("chat", "assistant"))
ASSISTANT_PROMPT = _env(
    "VAICE_ASSISTANT_PROMPT",
    default="You are a voice assistant on the user's Mac. Your reply is spoken aloud: answer in one or two short "
            "plain sentences, with no lists, symbols or file paths. Use a tool only when the question is about "
            "this computer; answer everything else yourself.",
)
# Assistant mode only: adds the `run_shell` skill (shell.py). Every command is approved first; only a short
# list of harmless ones by voice, the rest only by the UI's Run button. Off: the skill does not exist.
ASSISTANT_SHELL = _flag("VAICE_ASSISTANT_SHELL", default=False)
# Assistant mode only: after this many seconds without a spoken answer the reply says "One moment." (then
# "Still working." every 12 s) until the answer starts. 0 turns it off. A speaker's own filler can reach the
# microphone and be taken for the user interrupting (#36), which is the other reason for the off switch.
ASSISTANT_PROGRESS_SECONDS = _number("VAICE_ASSISTANT_PROGRESS_SECONDS", default="2.5", kind=float)
if not ASSISTANT_PROGRESS_SECONDS >= 0:
    raise SystemExit("vaice: VAICE_ASSISTANT_PROGRESS_SECONDS must be >= 0 (0 = no spoken progress)")
# Assistant mode only: the assistant keeps short notes you ask it to remember ("remember that I park on level
# three") in <VAICE_DATA_DIR>/memory.json, across conversations, and puts them into every prompt (so a remote
# helper brain gets them too). Off: no memory skills and no notes in the prompt (an existing file is left alone).
# Off by default: on the local 2B brain the memory skills cost 3 of the 51 older eval cases (225 -> 210 of 255:
# remember("Goodbye"), a reminder sent to remember, and Calculator opened for "Who makes the calculator app?").
ASSISTANT_MEMORY = _flag("VAICE_ASSISTANT_MEMORY", default=False)
VOICE = _choice("VAICE_VOICE", default="female", choices=("female", "male"))


LOCAL_HOSTS = ("localhost", "127.0.0.1", "::1")  # a brain at one of these hosts does not leave this machine


def brain_base(raw: str) -> str:
    """The helper brain's base URL as OpenAI clients take it, `/v1` included, without a trailing slash.

    Raises ValueError with a message that never repeats `raw`: a key pasted into a URL must not reach the log."""
    parts = urlsplit(raw if "://" in raw else f"http://{raw}")
    if parts.scheme not in ("http", "https") or not parts.netloc:
        raise ValueError("expected an http(s) URL like http://localhost:1234/v1")
    if "@" in parts.netloc or parts.query or parts.fragment:
        raise ValueError("it must not contain credentials, a query or a fragment (the API key goes in "
                         "VAICE_BRAIN_API_KEY)")
    path = parts.path.rstrip("/")
    if path.endswith("/chat/completions"):
        raise ValueError("give the base URL (e.g. http://localhost:1234/v1), not the endpoint: vaice adds "
                         "/chat/completions itself")
    return f"{parts.scheme}://{parts.netloc}{path}"


def _brain_url(name: str) -> str:
    """`brain_base` of the variable; "" (unset or empty) = no helper."""
    raw = os.getenv(name, "").strip()
    if not raw:
        return ""
    try:
        return brain_base(raw)
    except ValueError as exc:
        raise SystemExit(f"vaice: {name}: {exc}") from None


# Assistant mode only: an optional helper for the brain step, any OpenAI-compatible chat-completions endpoint
# (a local server such as LM Studio or Ollama, or a hosted one); vaice does not tell them apart. Unset = the
# local llama-server answers. Only the brain moves: speech recognition and speech output stay on llama-server.
# Everything the brain sees is sent to that endpoint (see README, Privacy). The key is a secret: never logged.
BRAIN_URL = _brain_url("VAICE_BRAIN_URL")
BRAIN_MODEL = _env("VAICE_BRAIN_MODEL", default="").strip()
BRAIN_API_KEY = _env("VAICE_BRAIN_API_KEY", default="").strip()
BRAIN_TIMEOUT = _number("VAICE_BRAIN_TIMEOUT", default="60", kind=float)
if BRAIN_TIMEOUT <= 0:
    raise SystemExit("vaice: VAICE_BRAIN_TIMEOUT must be > 0")

PORT = _number("VAICE_PORT", default="7860", kind=int)

# The conversation is kept here between runs; the last HISTORY_TURNS exchanges go into each prompt.
# Set VAICE_DATA_DIR="" to keep the conversation in memory only: nothing is written, loaded, or kept.
_DATA_DIR = os.getenv("VAICE_DATA_DIR", "~/.vaice")
DATA_DIR = None if _DATA_DIR.strip() == "" else Path(_DATA_DIR).expanduser()
CONVERSATION_FILE = None if DATA_DIR is None else DATA_DIR / "conversation.json"
MEMORY_FILE = None if DATA_DIR is None else DATA_DIR / "memory.json"  # assistant mode's notes
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
