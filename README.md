# vaice

Talk to a local model and hear it answer; the spoken reply streams back as it is generated.
The conversation carries over between questions and is kept as text history.

`vaice` uses a speech-to-speech model ([LFM3-Audio-2B](https://huggingface.co/LiquidAI/LFM3-Audio-2B-GGUF) or
another compatible model) running on a llama.cpp `llama-server`. The server hears your audio and streams its spoken
reply over the `/v1/realtime` WebSocket. Choose and run the model separately; `vaice` does not bundle a model.

```
browser mic ─WebRTC─▶ fastrtc (VAD: detects you stopped talking)
                         │  audio
                         ▼
                  LlamaCppResponder  llama-server :8001 /v1/realtime (WebSocket): speech streams back;
                   │                  a second session transcribes your audio in parallel
                   │ records the transcript and spoken text on `turn`
                   ▼
                  Conversation (~/.vaice/conversation.json) ──▶ transcript in the UI
```

## Quickstart (From Scratch in 3 Steps)

### 1. Build and start `llama-server` with an audio model

`vaice` requires llama.cpp `llama-server` **b11960 or newer** (v0.8.0+) for the `/v1/realtime` WebSocket.
At the time of writing, Homebrew's `llama.cpp` (0.7.0) is too old, so build `llama-server` from source:

```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j --target llama-server
build/bin/llama-server -hf LiquidAI/LFM3-Audio-2B-GGUF:Q4_K_M --port 8001
```

`llama-server` downloads `LiquidAI/LFM3-Audio-2B-GGUF` automatically on first run via `-hf` and listens on `http://localhost:8001`.

### 2. Install and run `vaice`

In a second terminal, clone this repository, install the locked environment with [uv](https://docs.astral.sh/uv/), and launch `vaice`:

```bash
git clone https://github.com/ranacseruet/vaice.git && cd vaice
uv sync --extra dev
.venv/bin/vaice
```

### 3. Talk in the browser

When you see `Running on local URL: http://127.0.0.1:7860`:

1. Open `http://127.0.0.1:7860` in Chrome.
2. Put on **headphones** (otherwise speaker audio feeds back into the microphone).
3. Allow microphone access, click **Record**, ask a question, and pause when done.

For each reply, the terminal logs `[llamacpp] time to first audio: N.NNs` and `[llamacpp] reply finished in N.NNs`.

---

## Requirements & Compatibility

- **Python 3.10–3.12** (tested in CI on macOS; dependencies also resolve on 3.13 and 3.14, but runtime behavior is not validated there).
- **[uv](https://docs.astral.sh/uv/)** for reproducible dependency management (`uv sync` installs the exact versions from `uv.lock`).
- **llama.cpp `llama-server` b11960+** running a model with audio output.
  - **Only llama.cpp `llama-server` is supported and tested.** `vaice` speaks its `/v1/realtime` WebSocket protocol and checks `/health` and `/v1/models` at startup to verify that the loaded model advertises audio output (`output_modalities` includes `"audio"`). Other servers (Ollama, LM Studio, oMLX, …) will not work.
  - Point `VAICE_LLAMACPP_URL` at a different host or port if `llama-server` isn't on `:8001`.
  - Current upstream `llama.cpp` documentation also describes `/v1/realtime` conversation support for LFM3-Omni and the `--reasoning off` server option. `vaice`'s integration has been exercised with LFM3-Audio; Omni behavior is not verified end to end here. On `llama.cpp` builds that provide the option, add `--reasoning off` to the server command if your selected model's chat template supports disabling reasoning. See the [llama.cpp server documentation](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md).
  - CI runs on macOS; Linux and Windows audio behavior has not been validated.

## Configuration

All settings are optional and configured via environment variables. To keep them in a file, copy `.env.example` to `.env` (gitignored), uncomment what you want to change, and start with:

```bash
set -a; source .env; set +a; .venv/bin/vaice
```

(`vaice` does not read `.env` itself.)

| Variable | Default | Meaning |
|---|---|---|
| `VAICE_PORT` | `7860` | UI port |
| `VAICE_DATA_DIR` | `~/.vaice` | Directory where `conversation.json` is stored |
| `VAICE_HISTORY_TURNS` | `10` | Past exchanges included in each prompt (`0` = none); older ones are dropped in blocks |
| `VAICE_LLAMACPP_URL` | `http://localhost:8001` | `llama-server` URL; only scheme, host and port are used (`/health`, `/v1/models` and `ws://…/v1/realtime` hang off it) |
| `VAICE_LLAMACPP_PROMPT` | `Reply naturally and concisely in 1 to 2 spoken sentences.` | Reply instructions; `vaice` prepends `Respond with interleaved text and audio.` |
| `VAICE_LLAMACPP_TIMEOUT` | `120` | Seconds to open a connection or the longest silence from `llama-server` before a reply fails |

## How It Works & Conversation History

### Startup & Shutdown
At startup, `vaice` checks `llama-server`:
- An unreachable or still-loading server (HTTP 503) logs a warning and starts the UI anyway, so the server can finish loading before your first question.
- A reachable server whose model does not advertise audio output is rejected immediately with an actionable error.
- Replies are capped at 512 tokens (text and audio frames together); truncation is logged as a warning.
- **Ctrl+C** stops the app cleanly: any reply in flight is abandoned (`the app is stopping; the reply is abandoned`) and the process exits within ~1 s when idle or ~5–6 s when a browser is connected mid-reply, even if the server never answers. Finished turns remain on disk.

### Multi-Turn Conversation & Barge-In
`vaice` persists your conversation to `~/.vaice/conversation.json` (see `VAICE_DATA_DIR`) after every update, so it survives page reloads and restarts.

- **New conversation:** Clicking **New conversation** (or the chatbot's trash icon) clears the stored history and immediately silences any reply playing or queued. Any utterance you were halfway through saying when clicking it is discarded; what you say after clicking starts a fresh conversation with no prior turns in the prompt.
- **Live transcript updates:** The chatbot view is checked once a second and re-rendered only when it changes, so a user transcript that finishes *after* the spoken reply ends still replaces the `"…"` placeholder automatically.
- **What the model sees:** Earlier turns are sent as plain text (`User: …` / `Assistant: …` lines in the session `instructions`), while the current question is sent as audio.
  - Because `llama-server`'s `/v1/realtime` endpoint rejects `conversation.item.create` with the `assistant` role, each reply opens a fresh session with the conversation history embedded in `instructions`. This keeps `Conversation` (`conversation.py`) the single owner of history windowing, barge-in cuts, and resets.
  - A second `/v1/realtime` session of type `transcription` runs in parallel on the same audio to produce the user transcript (with language auto-detection).
  - A turn whose transcript has not arrived yet is left out of the prompt rather than shown as `(inaudible)`, so the model never sees a dangling answer to an unreadable question. An empty completed transcript is recorded as `(inaudible)`.
  - Small speech-to-speech models use the last few turns well but may forget details from much earlier in a long conversation.
- **Barge-in (interrupting a reply):** If you speak over a reply, playback stops when you finish your interrupting question, and history keeps only the estimated portion of the reply that actually played, marked `(interrupted)`:
  - `fastrtc` queues audio chunks ahead of real-time playback, and LFM models emit text roughly ~2x ahead of audio. `vaice` splits the generated text by the share of audio handed out, and on barge-in further trims it to the wall-clock duration that could have played since the first audio chunk (keeping at least one word if any audio played).
  - `InterruptSafeReplyOnPause` (`interruptible.py`) fixes two `fastrtc` 0.0.34 barge-in races: it drops the stale audio chunk that was mid-generation when the interrupt happened, and it restores the interrupting utterance if the previous reply happened to finish at the exact moment of the barge-in.
- **History trimming:** Prompts include at most `VAICE_HISTORY_TURNS` exchanges. Oldest turns are dropped in blocks of `window // 2` rather than one by one.
- **Fault tolerance:**
  - A corrupt `conversation.json` is moved aside to `conversation.json.corrupt` with a warning, and `vaice` starts with an empty conversation.
  - Unwritable disk state logs a warning and continues the conversation in memory without crashing the session.
  - A failed reply (`llama-server` down, WebSocket closed, or server error event) is caught inside the reply generator so `fastrtc`'s WebRTC audio task stays alive; the partial reply and `(error: …)` note appear in the UI, and the next question works without reloading the page.

## Troubleshooting

- **`llama-server` has no `/v1/realtime` endpoint or returns 404 on WebSocket connect:**
  Your `llama.cpp` build is older than `b11960` (for example, Homebrew's `0.7.0` release). Build `llama-server` from a current `llama.cpp` checkout as shown in [Quickstart](#1-build-and-start-llama-server-with-an-audio-model).
- **`RuntimeError: the model on llama-server ... does not output audio`:**
  The model loaded in `llama-server` is text-only or audio-input-only. Start `llama-server` with an audio-output model such as `LiquidAI/LFM3-Audio-2B-GGUF:Q4_K_M`.
- **The assistant interrupts itself or hears its own voice:**
  Speaker audio is feeding back into your microphone and triggering `fastrtc`'s voice activity detection (VAD). Use headphones.
- **`concurrency_limit_reached` when opening a new tab or reconnecting:**
  `vaice` is a single-user demo (`concurrency_limit=1`). Normal page reloads close the old WebRTC connection immediately and reconnect within milliseconds. However, if a browser tab crashes or is force-closed without sending a WebRTC goodbye, the server holds the slot for ~30 seconds until ICE consent checks expire. Wait ~30 seconds and try again.
- **`AbortError: BodyStreamBuffer was aborted` / `Connection errored out` in the browser console on reload:**
  Harmless message from Gradio's frontend client when page navigation cuts its open `queue/data` SSE stream.

## Setup & Dependency Updates

```bash
uv sync --extra dev      # creates .venv from uv.lock: the exact versions CI tests
```

`uv sync` installs from the committed `uv.lock`, so every developer and CI run gets the same dependency versions (`.python-version` selects Python 3.12; pass `--python 3.10` for Python 3.10). A plain `uv pip install -e ".[dev]"` or `pip install -e ".[dev]"` also works and resolves against the version bounds in `pyproject.toml`.

To upgrade dependencies deliberately:

```bash
uv lock --upgrade-package fastrtc   # or --upgrade to refresh everything
uv sync --extra dev
.venv/bin/pytest && .venv/bin/ruff check src tests
```

Known compatibility bounds in `pyproject.toml`:
- **`fastrtc[vad]>=0.0.34,<0.1`** — `interruptible.py` and `tests/test_interruptible.py` rely on `fastrtc` 0.0.34's `ReplyOnPause` internals and verify `ReplyOnPause.__init__`'s signature on every test run.
- **`gradio>=5.50,<6`** — imported directly by `app.py`; capped below 6.0 to match `fastrtc` 0.0.34 and keep `gr.Chatbot(allow_tags=False)` behavior consistent.
- **`onnxruntime<1.24; python_version < '3.11'`** — `onnxruntime` 1.24+ declares `>=3.10` but publishes only `cp311+` wheels, so Python 3.10 stays on the 1.23 line.

## Tests & Code Quality

```bash
.venv/bin/pytest                                         # run all unit tests
.venv/bin/pytest --cov=vaice --cov-report=term-missing   # run with line coverage
.venv/bin/ruff check src tests                           # run linter
```

CI (`.github/workflows/ci.yml`) runs Ruff and `pytest --cov=vaice` with `uv sync --locked` across Python 3.10 and 3.12 on macOS for every push to `main` and every pull request.

See [CONTRIBUTING.md](CONTRIBUTING.md) for contributor guidelines and how the fake `/v1/realtime` server and multi-threaded `fastrtc` test harnesses work.

## Layout

```
src/vaice/
  app.py            CLI + Gradio/WebRTC UI; wraps a Responder, logs time to first audio, transcript + reset
  interruptible.py  ReplyOnPause subclass fixing fastrtc 0.0.34 barge-in races, reset silencing, and idle CPU polling
  lifecycle.py      Stop signal handler: Ctrl+C asks the app and its responders to abandon in-flight calls
  config.py         Environment-driven settings (.env.example lists them with defaults)
  conversation.py   Turn / Conversation: history windowing, barge-in trimming, atomic JSON persistence, error notes
  audio.py          Mic frame → mono PCM16 at the model's rate (resampled via scipy) and back to float32
  responders/
    base.py         Responder protocol (audio + history in → audio chunks out; reports transcript and spoken text)
    __init__.py     Responder package
    llamacpp.py     llama-server /v1/realtime: parallel reply and transcription sessions on daemon threads, startup check
tests/
  fake_realtime.py  Fake /v1/realtime WebSocket + HTTP server for testing LlamaCppResponder end to end
```

## License

Licensed under the [MIT License](LICENSE).
