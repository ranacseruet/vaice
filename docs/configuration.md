# Configuration

All settings are optional and configured via environment variables. To keep them in a file, copy `.env.example` to `.env` (gitignored), uncomment what you want to change, and start `vaice` from that directory — it loads `.env` from the working directory at startup:

```bash
.venv/bin/vaice
```

Precedence: variables already in the environment (e.g. `set -a; source .env; set +a; .venv/bin/vaice`, or `VAICE_FOO=1 .venv/bin/vaice`) win over the `.env` file, which wins over the built-in defaults.

| Variable | Default | Meaning |
|---|---|---|
| `VAICE_PORT` | `7860` | UI port |
| `VAICE_DATA_DIR` | `~/.vaice` | Directory where `conversation.json` (and, in assistant mode, `memory.json`) is stored (created `0755`; the files themselves are `0600`). Set to empty (`VAICE_DATA_DIR=`) to keep the conversation and the notes in memory only — nothing is persisted |
| `VAICE_HISTORY_TURNS` | `10` | Past exchanges included in each prompt (`0` = none); older ones are dropped in blocks |
| `VAICE_VAD_THRESHOLD` | `0.6` | Silero VAD speech probability threshold — what counts as speech, and so what can interrupt a playing reply. Raise (e.g. `0.7`) if room noise or speaker feedback cuts replies short |
| `VAICE_VAD_MIN_SILENCE_MS` | `150` | Silence that must follow speech before an utterance counts as ended — delays the reply start and barge-ins alike |
| `VAICE_VAD_SPEECH_PAD_MS` | `30` | Padding added around detected speech |
| `VAICE_CAN_INTERRUPT` | `1` | Set `0` to disable barge-in: a reply always runs to the end |
| `VAICE_LLAMACPP_URL` | `http://localhost:8001` | `llama-server` URL; only scheme, host and port are used (`/health`, `/v1/models` and `ws://…/v1/realtime` hang off it) |
| `VAICE_LLAMACPP_MODEL` | `LiquidAI/LFM3-Omni-2B-GGUF:Q8_0` | `vaice up` only: the `-hf` model of the `llama-server` it starts |
| `VAICE_LLAMACPP_SERVER_ARGS` | *(empty)* | `vaice up` only: extra arguments for the `llama-server` it starts, split like a shell would (e.g. `--tools get_info,read_file`) |
| `VAICE_LLAMACPP_START_TIMEOUT` | `900` | `vaice up` only: seconds to wait for the started `llama-server` to become ready; the first run downloads about 2.4 GB |
| `VAICE_MODE` | `chat` | `chat` = speech-to-speech replies from the model; `assistant` = transcribe, a brain that can call vaice's own skills, then `/tts` (see [Assistant mode](assistant-mode.md)). Anything else exits at startup |
| `VAICE_ASSISTANT_PROMPT` | `You are a voice assistant on the user's Mac. …` (answer in one or two short spoken sentences, use a tool only for questions about this computer, and use it again for every question about live facts such as the screen, the battery or the disk, even if you answered one earlier) | Assistant mode only: the brain's system prompt (the current date and time is added). If you replace it, keep the "use the tool again" sentence: without it the model answers "what do you see now?" from its earlier answer instead of taking a new screenshot |
| `VAICE_ASSISTANT_SHELL` | `false` | Assistant mode only: adds the `run_shell` skill, which proposes a shell command that runs only after you approve it (see [Shell commands](assistant-mode.md#shell-commands)). Off: the skill does not exist. A boolean: `1/true/yes/on` or `0/false/no/off` |
| `VAICE_ASSISTANT_MEMORY` | `false` | Assistant mode only: long-term memory, the `remember`, `forget` and `forget_everything` skills and the notes block in the prompt (see [Memory](assistant-mode.md#memory)). Notes live in `<VAICE_DATA_DIR>/memory.json` (`0600`), or in memory only when `VAICE_DATA_DIR` is empty. Off: the skills do not exist, the notes are neither read nor used, and the file is left alone. A boolean, like `VAICE_ASSISTANT_SHELL` |
| `VAICE_ASSISTANT_PROGRESS_SECONDS` | `2.5` | Assistant mode only: seconds without an answer before the assistant says "One moment." (then "Still working." every 12 s) until the answer starts; `0` turns spoken progress off. On speakers the microphone may hear it as an interruption (see [Assistant mode](assistant-mode.md)) |
| `VAICE_ASSISTANT_MAX_STEPS` | `10` | Assistant mode only: the most brain calls one task may take (a question, and what follows a spoken yes to a confirmation). The last call is made without tools so it must answer, so `2` is the least (`1` would never use a tool). Must be an integer `>= 2` |
| `VAICE_ASSISTANT_TASK_SECONDS` | `120` | Assistant mode only: seconds a task may run, counted from its start and checked between brain calls (a call in flight is not cut off; the brain has its own timeout). Once they have passed, the next call is the one without tools. Must be `> 0` |
| `VAICE_ASSISTANT_AUTO_SECONDS` | `5` | Assistant mode only: seconds after its announcement has played (estimated from the audio handed out, not measured) that an MCP tool with the policy `auto` runs by itself, unless you press **Stop** or say stop; speech still in progress when it fires has not been heard yet, the button always works (see [MCP tools](mcp-tools.md)). `0` turns the countdown off: an `auto` tool then waits for the **Run** button like `click`. Must be `>= 0` |
| `VAICE_VOICE` | `female` | Assistant mode only: the `/tts` voice, `female` or `male` |
| `VAICE_BRAIN_URL` | *(empty: the local `llama-server` is the brain)* | Assistant mode only: an optional helper model, any OpenAI-compatible endpoint, as the **base URL OpenAI clients take, `/v1` included** (e.g. `http://localhost:1234/v1`, `https://api.openai.com/v1`); vaice posts to `<base>/chat/completions`. A URL that already ends in `/chat/completions`, or that holds credentials, a query or a fragment, exits at startup. Ignored in chat mode (one warning). What it receives: see [Helper model](helper-model.md) |
| `VAICE_BRAIN_MODEL` | *(empty)* | The `model` sent to the helper. Most hosted endpoints require it: with a helper URL and no model, vaice warns at startup and goes on |
| `VAICE_BRAIN_API_KEY` | *(empty)* | Sent to the helper as `Authorization: Bearer …`. A secret: never logged, never in an error message or the eval's results. Put it in `.env` or the environment, not on a command line |
| `VAICE_BRAIN_TIMEOUT` | `60` | Seconds the helper may take for one request (the local brain keeps `VAICE_LLAMACPP_TIMEOUT`) |
| `VAICE_VISION_URL` | *(empty: no screen skill)* | Assistant mode only: turns on the `look_at_screen` skill (see [Screen vision](screen-vision.md)). A vision model, any OpenAI-compatible endpoint that accepts images, as the **base URL with `/v1`** (e.g. `http://127.0.0.1:8002/v1`); same rules as `VAICE_BRAIN_URL` (a `/chat/completions` suffix, credentials, a query or a fragment exit at startup). Ignored in chat mode (one warning). **A screenshot of your main display is sent to this endpoint** whenever you ask about your screen |
| `VAICE_VISION_MODEL` | *(empty)* | The `model` sent to the vision endpoint. Hosted endpoints need it (startup warns when it is empty); a `llama-server` serving one model does not |
| `VAICE_VISION_API_KEY` | *(empty)* | Sent to the vision endpoint as `Authorization: Bearer …`. A secret: never logged, never in an error message. Put it in `.env` or the environment, not on a command line |
| `VAICE_VISION_TIMEOUT` | `30` | Seconds the vision endpoint may take for one request (must be > 0) |
| `VAICE_VISION_MAX_TOKENS` | `200` | Upper bound on the vision model's answer, in tokens (`0` = send none; for a hosted model that rejects `max_tokens`) |
| `VAICE_MCP_CONFIG` | *(empty: no MCP tools)* | Assistant mode only: path to a JSON file naming MCP servers to start and the tools to offer from them (see [MCP tools](mcp-tools.md); `~` is expanded). Needs the optional extra: `uv sync --extra mcp`. Ignored in chat mode (one warning). **Each server is a program vaice starts with your permissions, and what its tools receive and return goes to the brain** |
| `VAICE_MCP_TIMEOUT` | `30` | Seconds a server may take to start and, separately, to answer one tool call (must be > 0) |
| `VAICE_BROWSE` | `off` | Assistant mode only: `pi` adds the `browse` skill, which hands a goal to the Pi coding agent to carry out in a Chrome window of its own in the background (see [Browser tasks](browser-tasks.md)). `off` or `pi`; anything else exits at startup. Needs `pi` 1.5.2 or newer on `PATH` and `VAICE_BRAIN_URL` (without either, a warning and no skill). Ignored in chat mode (one warning). **Page text and your goal go to the brain endpoint; the agent can open any site and fill in pages after you approve each step** |
| `VAICE_BROWSE_SECONDS` | `180` | Seconds one browser task may run before it is stopped (must be > 0) |
| `VAICE_BROWSE_HEADLESS` | `false` | A boolean: `true` runs Chrome without a visible window |
| `VAICE_LLAMACPP_PROMPT` | `Reply naturally and concisely in 1 to 2 spoken sentences.` | Reply instructions; `vaice` prepends `Respond with interleaved text and audio.` |
| `VAICE_LLAMACPP_TIMEOUT` | `120` | Seconds to open a connection or the longest silence from `llama-server` before a reply fails. It is also the total bound for a tool call: when it passes — or the reply is stopped — vaice stops waiting for the call and discards the result (the underlying request is not cancelled; the server may still run/finish the tool). The parallel transcription session's silence budget is this setting too, but its connection **handshake** uses a fixed 2 s (transcription is best-effort and degrades to `(inaudible)`), so raising this cannot extend a slow-handshake server |
| `VAICE_LLAMACPP_MAX_OUTPUT_TOKENS` | `1024` | Reply cap sent as the realtime session's `max_output_tokens`; text tokens and audio frames count together. Raise it for longer spoken replies |
| `VAICE_LLAMACPP_TOOL_VALUE_CHARS` | `200` | Longest string value kept from a tool result before it is fed back to the model. Too small cuts answer-bearing fields (a build timestamp, the end of a file); too large makes the model dictate the result |
| `VAICE_LLAMACPP_TOOL_OUTPUT_CHARS` | `800` | Cap on the whole compacted tool result fed back to the model |
| `VAICE_LLAMACPP_TOOLS` | *(unset)* | Voice access to llama-server's built-in tools (see [Chat mode](chat-mode.md#tools)). Unset = off; `builtin` = every tool the server advertises — the set is chosen on the server with `--tools` |
