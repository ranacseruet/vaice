# Configuration

All settings are optional and configured via environment variables. To keep them in a file, copy `.env.example` to `.env` (gitignored), uncomment what you want to change, and start `vaice` from that directory — it loads `.env` from the working directory at startup:

```bash
.venv/bin/vaice
```

Precedence: variables already in the environment (e.g. `set -a; source .env; set +a; .venv/bin/vaice`, or `VAICE_FOO=1 .venv/bin/vaice`) win over the `.env` file, which wins over the built-in defaults.

| Variable | Default | Meaning |
|---|---|---|
| `VAICE_PORT` | `7860` | UI port |
| `VAICE_DATA_DIR` | `~/.vaice` | Directory where `conversation.json` (and, with memory on, `memory.json`) is stored (created `0755`; the files themselves are `0600`). Set to empty (`VAICE_DATA_DIR=`) to keep the conversation and the notes in memory only — nothing is persisted |
| `VAICE_HISTORY_TURNS` | `10` | Past exchanges included in each prompt (`0` = none); older ones are dropped in blocks |
| `VAICE_VAD_THRESHOLD` | `0.6` | Silero VAD speech probability threshold — what counts as speech, and so what can interrupt a playing reply. Raise (e.g. `0.7`) if room noise or speaker feedback cuts replies short |
| `VAICE_VAD_MIN_SILENCE_MS` | `750` | Trailing silence that must follow speech in streaming VAD before an utterance ends (default `750` ms avoids 300–600 ms mid-sentence pause splits while keeping median reply latency ~0.8 s) |
| `VAICE_VAD_SPEECH_PAD_MS` | `30` | Padding around detected speech used by chunked fallback VAD |
| `VAICE_CAN_INTERRUPT` | `1` | Set `0` to disable barge-in: a reply always runs to the end |
| `VAICE_LLAMACPP_URL` | `http://localhost:8001` | `llama-server` URL; only scheme, host and port are used (`/health`, `/v1/models` and `ws://…/v1/realtime` hang off it) |
| `VAICE_LLAMACPP_MODEL` | `LiquidAI/LFM3-Omni-2B-GGUF:Q8_0` | `vaice up` only: the `-hf` model of the `llama-server` it starts |
| `VAICE_LLAMACPP_SERVER_ARGS` | *(empty)* | `vaice up` only: extra arguments for the `llama-server` it starts, split like a shell would (e.g. `--ctx-size 16384`) |
| `VAICE_LLAMACPP_START_TIMEOUT` | `900` | `vaice up` only: seconds to wait for the started `llama-server` to become ready; the first run downloads about 2.4 GB |
| `VAICE_PROMPT` | `You are a voice assistant on the user's Mac. …` (answer in one or two short spoken sentences, use a tool only for questions about this computer, and use it again for every question about live facts such as the screen, the battery or the disk, even if you answered one earlier) | The reply instructions; vaice adds the realtime mode line, the earlier turns, the current date and time and the memory notes. If you replace it, keep the "use the tool again" sentence: without it the model answers "what do you see now?" from its earlier answer instead of taking a new screenshot |
| `VAICE_SHELL` | `false` | Adds the `run_shell` skill, which proposes a shell command that runs only after you click Run (see [Shell commands](assistant.md#shell-commands)). Off: the skill does not exist. A boolean: `1/true/yes/on` or `0/false/no/off` |
| `VAICE_MEMORY` | `false` | Long-term memory, the `remember`, `forget` and `forget_everything` skills and the notes block in the prompt (see [Memory](assistant.md#memory)). Notes live in `<VAICE_DATA_DIR>/memory.json` (`0600`), or in memory only when `VAICE_DATA_DIR` is empty. Off: the skills do not exist, the notes are neither read nor used, and the file is left alone. A boolean, like `VAICE_SHELL` |
| `VAICE_VISION_URL` | *(empty: no screen skill)* | turns on the `look_at_screen` skill (see [Screen vision](screen-vision.md)). A vision model, any OpenAI-compatible endpoint that accepts images, as the **base URL with `/v1`** (e.g. `http://127.0.0.1:8002/v1`); a `/chat/completions` suffix, credentials, a query or a fragment exit at startup. **A screenshot of your main display is sent to this endpoint** whenever you ask about your screen |
| `VAICE_VISION_MODEL` | *(empty)* | The `model` sent to the vision endpoint. Hosted endpoints need it (startup warns when it is empty); a `llama-server` serving one model does not |
| `VAICE_VISION_API_KEY` | *(empty)* | Sent to the vision endpoint as `Authorization: Bearer …`. A secret: never logged, never in an error message. Put it in `.env` or the environment, not on a command line |
| `VAICE_VISION_TIMEOUT` | `30` | Seconds the vision endpoint may take for one request (must be > 0) |
| `VAICE_VISION_MAX_TOKENS` | `200` | Upper bound on the vision model's answer, in tokens (`0` = send none; for a hosted model that rejects `max_tokens`) |
| `VAICE_MCP_CONFIG` | *(empty: no MCP tools)* | path to a JSON file naming MCP servers to start and the tools to offer from them (see [MCP tools](mcp-tools.md); `~` is expanded). Needs the optional extra: `uv sync --extra mcp`. **Each server is a program vaice starts with your permissions, and what its tools receive and return goes to the model** |
| `VAICE_MCP_TIMEOUT` | `30` | Seconds a server may take to start and, separately, to answer one tool call (must be > 0) |
| `VAICE_BROWSE` | `off` | `pi` or `acp` adds the `browse` skill (and `browser_status` / `stop_browsing`), which hands a goal to a browser agent over ACP to carry out in a Chrome window of its own in the background, quietly until the result (see [Browser tasks](browser-tasks.md)). `pi`: the Pi coding agent through `pi-acp`; `acp`: any ACP agent, named by `VAICE_BROWSE_AGENT`. `off`, `pi` or `acp`; anything else exits at startup. Both need `npx` on `PATH` and the optional extra `uv sync --extra browse`; `pi` also needs `pi` 1.5.2 or newer on `PATH` and `VAICE_BROWSE_URL` (without any of these, a warning and no skill). **Page text and your goal go to the agent's model endpoint (`VAICE_BROWSE_URL` for `pi`); the agent can open any site and fill in pages after you approve each step** |
| `VAICE_BROWSE_AGENT` | *(empty)* | The agent's command line for `VAICE_BROWSE=acp` (split like a shell would, e.g. `opencode acp`); required for `acp`, its program must be on `PATH`. The agent runs with your environment minus every `VAICE_*` variable, and its own tools and permission mode are set in its own config: vaice only gates what the agent asks about |
| `VAICE_BROWSE_URL` | _(unset)_ | The model Pi uses (`VAICE_BROWSE=pi` only; ignored with `acp`): an OpenAI-compatible base URL, `/v1` included, the same convention and validation as `VAICE_VISION_URL` (no `/chat/completions`, no credentials in the URL; a bad value exits at startup naming this variable). Required for `VAICE_BROWSE=pi`. Everything Pi sees on a page goes to this host (startup warns) |
| `VAICE_BROWSE_MODEL` | _(empty)_ | The model id sent to that endpoint (`pi` only; empty = a warning: most endpoints need one) |
| `VAICE_BROWSE_API_KEY` | _(empty)_ | Bearer token for that endpoint (`pi` only), a secret: it reaches Pi only through its process environment, never a file or a log line, and no other vaice key is used instead |
| `VAICE_BROWSE_SECONDS` | `600` | Seconds one browser task may run before it is stopped (must be > 0) |
| `VAICE_BROWSE_HEADLESS` | `false` | A boolean: `true` runs Chrome without a visible window |
| `VAICE_BROWSE_CONFIRM` | `ask` | `ask`: a browser step that changes a page (click, type, fill a form, run a script, upload, a dialog) needs the Run button first. `off`: none does; the agent acts on its own (startup warns). Anything else exits at startup. **With `off` the agent can submit forms and buy things on any site it reaches; the profile being new only means it is logged in nowhere** |
| `VAICE_LLAMACPP_TIMEOUT` | `120` | Seconds to open a connection or the longest silence from `llama-server` before a reply fails. The parallel transcription session's silence budget is this setting too, but its connection **handshake** uses a fixed 2 s (transcription is best-effort and degrades to `(inaudible)`), so raising this cannot extend a slow-handshake server |
| `VAICE_LLAMACPP_MAX_OUTPUT_TOKENS` | `1024` | Reply cap sent as the realtime session's `max_output_tokens`; text tokens and audio frames count together. Raise it for longer spoken replies |
| `VAICE_LLAMACPP_TOOL_VALUE_CHARS` | `200` | Longest string value kept from a tool result before it is fed back to the model. Too small cuts answer-bearing fields (a build timestamp, the end of a file); too large makes the model dictate the result |
| `VAICE_LLAMACPP_TOOL_OUTPUT_CHARS` | `800` | Cap on the whole compacted tool result fed back to the model |

## Removed and renamed settings

An earlier version read the settings below; this one does not. A name that is still set (in the environment or the
`.env` file) is ignored and startup logs one warning naming it and the reason, never the value.

| Setting | What happened |
|---|---|
| `VAICE_ASSISTANT_PROMPT`, `VAICE_LLAMACPP_PROMPT` | Renamed to `VAICE_PROMPT` (one prompt now) |
| `VAICE_ASSISTANT_SHELL` | Renamed to `VAICE_SHELL` |
| `VAICE_ASSISTANT_MEMORY` | Renamed to `VAICE_MEMORY` |
| `VAICE_MODE` | Removed: there is one mode, the assistant (one speech-to-speech session with skills as its tools) |
| `VAICE_LLAMACPP_TOOLS` | Removed: llama-server's own tools are not used; the skills run in vaice |
| `VAICE_VOICE` | Removed: there is no separate text-to-speech step; the voice is the model's own |
| `VAICE_ASSISTANT_PROGRESS_SECONDS` | Removed: no spoken progress lines |
| `VAICE_ASSISTANT_MAX_STEPS`, `VAICE_ASSISTANT_TASK_SECONDS` | Removed: a reply makes at most two tool rounds; there is no step or time budget |
| `VAICE_BRAIN_URL`, `VAICE_BRAIN_MODEL`, `VAICE_BRAIN_API_KEY`, `VAICE_BRAIN_TIMEOUT` | Removed: there is no helper brain; every reply is one session on the local `llama-server` |
| `VAICE_ASSISTANT_AUTO_SECONDS` | Removed: the MCP `auto` policy is gone; use `run` or `click` |

A `voice` or `auto` policy in an MCP config file exits at startup naming the server and the tool.
