# vaice

Talk to a local model and hear it answer; the spoken reply streams back as it is generated.
The conversation carries over between questions and is kept as text history. The default path uses a
llama.cpp realtime audio model. An optional text path transcribes with local Whisper Tiny, streams text
from an OpenAI-compatible chat endpoint, and speaks each sentence with local Kokoro.

```
browser mic ─WebRTC─▶ fastrtc (VAD: detects you stopped talking)
                         │  audio
                         ▼
                  Responder
                   ├── realtime (default): llama-server /v1/realtime (WebSocket) speaks and transcribes
                   └── text (explicit): local Whisper → /v1/chat/completions → local Kokoro
                    records the transcript and audio-covered text on `turn`
                   ▼
                  Conversation (~/.vaice/conversation.json) ──▶ transcript in the UI
```

## Requirements

- Python 3.10–3.12 (what CI tests; the dependencies also resolve on 3.13 and 3.14, but nothing was run there)
- [uv](https://docs.astral.sh/uv/)
- For the default `realtime` path: a llama.cpp `llama-server` **b11960 or newer** (v0.8.0) running a model
  with audio input and output. It needs the `/v1/realtime` WebSocket. At the time of writing, Homebrew's
  llama.cpp (0.7.0) is too old, so build it from source:

  ```bash
  git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
  cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j --target llama-server
  build/bin/llama-server -hf LiquidAI/LFM3-Audio-2B-GGUF:Q4_K_M --port 8001
  ```

  **Only llama.cpp `llama-server` is supported and tested for the realtime path.** vaice speaks its
  `/v1/realtime` events and checks `/health` and `/v1/models`. Point `VAICE_LLAMACPP_URL` at a different host
  or port if llama-server isn't on `:8001`. The CI runner is macOS; Linux and Windows audio behavior has not
  been validated.

- For the optional `text` path: an OpenAI-compatible server with `/v1/chat/completions` streaming and
  `/v1/models`. Its API base defaults to `VAICE_LLAMACPP_URL` plus `/v1`; use `VAICE_TEXT_URL` for a different
  host or a path prefix. Chat endpoint support must be selected explicitly because model modality metadata
  alone does not establish that this endpoint works.
- Python 3.10–3.12 and the optional sherpa-onnx CPU package plus model files if you choose `text`. Those
  large speech models are downloaded separately and loaded only when the first text turn needs them.

  Current upstream llama.cpp documentation describes `/v1/realtime` conversation support for LFM3-Omni and
  the `--reasoning off` server option. vaice's integration has been exercised with LFM3-Audio; Omni behavior
  is not verified end to end here. On llama.cpp builds that provide the option, add `--reasoning off` to the
  server command if your selected model's chat template supports disabling reasoning. This is a server startup
  option, not a vaice setting.
  See the [llama.cpp server documentation](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md).

## Setup

```bash
uv venv .venv                       # picks Python 3.12 from .python-version
uv pip install -e ".[dev]"          # default realtime path
# For the optional local Whisper + Kokoro path:
uv pip install -e ".[dev,local-voice]"
```

### Local speech models for text mode

The `local-voice` extra installs sherpa-onnx; it does not download model weights. These official
sherpa-onnx archives match vaice's default paths and filenames:

```bash
mkdir -p ~/.vaice/models
cd ~/.vaice/models
curl -L -o whisper.tar.bz2 https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-whisper-tiny.tar.bz2
tar xf whisper.tar.bz2
rm whisper.tar.bz2
curl -L -o kokoro.tar.bz2 https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/kokoro-en-v0_19.tar.bz2
tar xf kokoro.tar.bz2
rm kokoro.tar.bz2
curl -L -o kokoro-en-v0_19/lexicon-en-us.txt https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/lexicon-en-us.txt
```

This provides Whisper's `tiny-encoder.int8.onnx`, `tiny-decoder.int8.onnx`, and `tiny-tokens.txt`, plus
Kokoro's `model.onnx`, `voices.bin`, `tokens.txt`, `espeak-ng-data/`, and the shared English lexicon.
These filenames and the `OfflineRecognizer.from_whisper` / `OfflineTtsKokoroModelConfig` setup follow
sherpa-onnx's [Whisper Tiny](https://k2-fsa.github.io/sherpa/onnx/pretrained_models/whisper/tiny.html),
[Kokoro](https://k2-fsa.github.io/sherpa/onnx/tts/pretrained_models/kokoro.html), and
[English lexicon](https://k2-fsa.github.io/sherpa/onnx/c-api/html/tts.html) documentation. The paths can
be changed individually with the `VAICE_WHISPER_*` and `VAICE_KOKORO_*` settings below.
The combined model process peaked at about 1.0 GiB RSS in a local smoke run; memory use varies by OS and
runtime build.

## Run

```bash
.venv/bin/vaice                      # realtime default; connect to llama-server at VAICE_LLAMACPP_URL
# VAICE_PIPELINE=text .venv/bin/vaice # local Whisper → text chat → local Kokoro
```

For realtime mode, llama-server downloads its model with `-hf`. For text mode, run the selected
OpenAI-compatible chat server and set `VAICE_TEXT_URL` if its API base differs from the default.
When vaice prints `Running on local URL: http://127.0.0.1:7860`, open it in Chrome, allow the
microphone, click record, ask a question, then stop talking. Use headphones, otherwise the speakers
feed back into the mic.

The page identifies the active voice path. In realtime mode, vaice checks llama-server at startup: an
unreachable or still-loading server is logged and the UI starts, so it can become available before
your first question. Text mode uses `VAICE_TEXT_URL`; without `VAICE_TEXT_MODEL`, its model ID is
discovered from `/models` on the first turn. Realtime replies are capped at 512 tokens (text and audio
frames together); the log reports truncation.

`VAICE_PIPELINE=realtime` keeps the default behavior. `VAICE_PIPELINE=text` selects local Whisper,
OpenAI-compatible chat streaming, and local Kokoro. `VAICE_PIPELINE=auto` only selects realtime when
`/v1/models` advertises both audio input and audio output and vaice verifies the `/v1/realtime` session
handshake. Text-only metadata does not prove a chat-completions endpoint, so auto mode asks you to set
`VAICE_PIPELINE=text` explicitly. Missing metadata is unknown, not a rejection or a capability claim.

`--backend` has been removed. If `VAICE_BACKEND` is set, startup shows a migration error; remove it from
your shell or `.env` file. `VAICE_NATIVE_MODEL` and `VAICE_NATIVE_PROMPT` are also removed and have no effect.
See [Configuration](#configuration) for the remaining settings and migration details.

Ctrl+C stops the app: the reply that is in flight is abandoned and the process exits,
even if the model or the server never answers, or never completes the connection (the log says
`the app is stopping; the reply is abandoned`). Measured: about 1 s with no browser connected, about 5–6 s
with one connected mid-reply (gradio closing the server). The turns that were already finished are on disk.

## Conversation history

vaice remembers the conversation. It is saved to `~/.vaice/conversation.json` (see
`VAICE_DATA_DIR`) after every change, so it survives page reloads and restarts, and the page shows
it as a transcript. The **New conversation** button (and the chatbot's trash icon) clears it and
silences a reply that is still playing or queued; a question you were half-way through saying when
you click it is discarded, so say it again. What you say *after* the click is the first turn of the
new conversation: it is answered as usual, but the model gets no earlier turns, because they are
gone.

- **The page keeps up:** the chatbot view is re-checked once a second and re-rendered only when it
  actually changed, so a transcript that arrives *after* the reply ended replaces the "…" without
  waiting for the next exchange or a page reload.
- **What the model sees:** earlier turns as plain text. A turn whose transcript has not arrived
  yet (it is still being transcribed) is left out of the prompt entirely, so the model never sees
  a reply to a question it cannot read and never reads it as `(inaudible)`. A transcript that
  arrived but is empty shows `(inaudible)`.
  - In realtime mode, the current question is sent as audio; earlier turns are text in the session
    instructions (`User: …` / `Assistant: …` lines after the prompt). llama-server's realtime sessions
    refuse earlier assistant turns as conversation items, so every reply opens a fresh session and vaice
    stays the single owner of history. A second, transcription session processes the same audio in parallel.
    In text mode, Whisper transcribes the current question locally before the chat request; earlier turns
    and the transcript are sent as ordinary user/assistant messages.
  - The small models use the last few turns well but often forget things said much earlier
    (e.g. your name from the first turn), even though they are in the prompt.
- **Barge-in:** if you talk over a reply, it stops when you stop talking, and history keeps the
  part of the reply that had played, marked `(interrupted)`. That part is estimated, not measured:
  the words of the audio handed out, cut to the share that the time since the first audio could
  have played (fastrtc queues audio ahead of playback). At least one word is kept, so an
  interrupted reply is never recorded as empty. The LFM models write their text well
  ahead of their audio, so the text is also split by the share of audio received so far. In text mode,
  complete sentences are synthesized and queued in order; the visible cumulative text is limited to the
  sentence samples handed out so far. Expect interrupted text cutoffs to be approximate at word boundaries.
  The chunk that was being generated when you interrupted is dropped instead of played;
  fastrtc would queue it after clearing its
  playback queue ([#2](https://github.com/ranacseruet/vaice/issues/2)).
- **Trimming:** each prompt includes at most the last `VAICE_HISTORY_TURNS` exchanges. The oldest
  are dropped in blocks of half that, not one at a time, so the start of the prompt stays the same
  for several turns. Each reply uses a fresh session, so the server does not reuse a prefix today.
- A corrupt `conversation.json` is renamed to `conversation.json.corrupt` (with a warning in the log;
  if it cannot even be renamed, vaice starts empty and leaves the file where it is)
  and a fresh conversation starts.
- **A conversation that can't be written to disk** (full disk, read-only file) is only a warning in
  the log: the reply goes on and the conversation stays in memory, but it is lost when the app stops.
  The file that was there before is left untouched.
- **A failed reply** (llama-server down, an `error` event from it, or a model error) does not end the session: the log gets the
  traceback, the transcript shows what was heard of the reply plus `(error: …)`, and the next
  question works. The error note is kept on disk with the turn but is never sent to the model;
  a question the failed reply never transcribed counts as `(inaudible)`.

This is a single-user demo: fastrtc accepts one WebRTC connection at a time, so a second tab is
refused (`concurrency_limit_reached`) while the first is connected.

- **Reloading the page** is fine: the browser closes the old connection as it navigates, and the
  reloaded page connects again straight away (measured, also mid-reply).
- **A browser that disappears without closing its connection** (measured in headless Chromium: tab closed
  via Playwright, with and without `beforeunload`, or the whole browser closed) sends no goodbye, so the
  server keeps the connection until its keep-alive checks give up, about 30 s. A connect attempt in that
  window is refused with `concurrency_limit_reached`; try again after it. Presumably the same after a crash
  or a network drop; not measured with a real, headed browser.
- Reloading while connected can log `AbortError: BodyStreamBuffer was aborted` and
  `Unexpected error Connection errored out.` in the browser console. That is Gradio's JS client reporting
  the event stream that carries transcript updates being cut by the navigation. It has no server-side effect
  and nothing breaks; the page reloads the stored conversation and reconnects.

## Configuration

All optional, via environment variables. To keep them in a file, copy `.env.example` to `.env` (gitignored),
uncomment what you want to change, and start with `set -a; source .env; set +a; .venv/bin/vaice`;
vaice does not read `.env` itself.

| Variable | Default | Meaning |
|---|---|---|
| `VAICE_PORT` | `7860` | UI port |
| `VAICE_DATA_DIR` | `~/.vaice` | where `conversation.json` is kept |
| `VAICE_HISTORY_TURNS` | `10` | past exchanges included in each prompt (`0` = none); older ones are dropped in blocks |
| `VAICE_PIPELINE` | `realtime` | `realtime`, `text`, or `auto`. Text mode is explicit because modality metadata does not prove a chat-completions endpoint exists |
| `VAICE_LLAMACPP_URL` | `http://localhost:8001` | llama-server URL; only scheme, host and port are used (`/health`, `/v1/models`, and `/v1/realtime` hang off it). The old `VAICE_LLM_URL` alias, including URLs with paths, still works |
| `VAICE_TEXT_URL` | `VAICE_LLAMACPP_URL` + `/v1` | OpenAI-compatible API base for text mode; its optional path prefix is preserved, and `/models` and `/chat/completions` are added |
| `VAICE_LLAMACPP_PROMPT` | `Reply naturally and concisely in 1 to 2 spoken sentences.` | realtime instructions; vaice adds `Respond with interleaved text and audio.` in front. The old `VAICE_LLM_PROMPT` and `VAICE_PROMPT` aliases still work |
| `VAICE_LLAMACPP_TIMEOUT` | `120` | seconds to open a connection or the longest silence from llama-server before a reply fails; the old `VAICE_LLM_TIMEOUT` alias still works |
| `VAICE_TEXT_MODEL` | first model from `/models` | optional model ID for `/chat/completions` |
| `VAICE_TEXT_PROMPT` | same default as realtime | system prompt sent to the text chat model; `VAICE_LLM_PROMPT` and `VAICE_PROMPT` remain aliases |
| `VAICE_WHISPER_MODEL_DIR` | `~/.vaice/models/sherpa-onnx-whisper-tiny` | Whisper folder; defaults to `tiny-encoder.int8.onnx`, `tiny-decoder.int8.onnx`, and `tiny-tokens.txt` |
| `VAICE_WHISPER_ENCODER` | under Whisper model dir | override the Whisper encoder file |
| `VAICE_WHISPER_DECODER` | under Whisper model dir | override the Whisper decoder file |
| `VAICE_WHISPER_TOKENS` | under Whisper model dir | override the Whisper tokens file |
| `VAICE_WHISPER_LANGUAGE` | empty | Whisper language hint; empty means automatic language detection |
| `VAICE_KOKORO_MODEL_DIR` | `~/.vaice/models/kokoro-en-v0_19` | Kokoro model folder |
| `VAICE_KOKORO_MODEL` | `model.onnx` under model dir | override the ONNX model |
| `VAICE_KOKORO_VOICES` | `voices.bin` under model dir | override Kokoro voices |
| `VAICE_KOKORO_TOKENS` | `tokens.txt` under model dir | override Kokoro tokens |
| `VAICE_KOKORO_DATA_DIR` | `espeak-ng-data` under model dir | Kokoro's espeak-ng data directory |
| `VAICE_KOKORO_LEXICON` | `lexicon-en-us.txt` under model dir | English pronunciation lexicon |
| `VAICE_KOKORO_SPEAKER` | `0` | numeric Kokoro speaker ID from 0 to 10 |
| `VAICE_KOKORO_SPEED` | `1.0` | Kokoro speech speed |

`VAICE_LLAMACPP_URL`, `VAICE_LLAMACPP_PROMPT` and `VAICE_LLAMACPP_TIMEOUT` replaced `VAICE_LLM_URL`,
`VAICE_LLM_PROMPT` (and before it `VAICE_PROMPT`) and `VAICE_LLM_TIMEOUT`. Both spellings work; if you set
both, the new name wins. `VAICE_TEXT_URL` is a separate setting so an OpenAI-compatible API can use a
path prefix without changing the historical `VAICE_LLM_URL` behavior.

`VAICE_BACKEND` and `--backend` were removed with backend selection. If `VAICE_BACKEND` is still set,
vaice stops at startup with instructions to remove it; it does not silently switch an old native selection
to llama-server. `VAICE_NATIVE_MODEL` and `VAICE_NATIVE_PROMPT` are no longer read. Remove those lines from
your `.env`; start a compatible llama-server and set `VAICE_LLAMACPP_URL` only if it uses a different URL.

`VAICE_LLM_MODEL`, `VAICE_TTS_MODEL`, `VAICE_TTS_VOICE`, `VAICE_TTS_SPEED`, `VAICE_VOICE`, and
`VAICE_SPEED` belonged to the removed Gemma + Kokoro path. They remain ignored and must not be used to
configure this text pipeline. Use `VAICE_TEXT_MODEL` and the `VAICE_KOKORO_*` settings instead.

The `local-voice` extra and the Whisper/Kokoro model files are only needed for `VAICE_PIPELINE=text`.
They are imported and loaded lazily on the first text turn; the realtime default does not initialize them.
Kokoro's English model supports numeric speaker IDs 0–10.

## Tests

```bash
.venv/bin/pytest
.venv/bin/ruff check src tests
```

CI (`.github/workflows/ci.yml`) runs Ruff and pytest on Python 3.10 and 3.12 on macOS for every push
to `main` and every pull request.

Unit tests use fakes for the models: the llamacpp backend against a fake `/v1/realtime` server
(`tests/fake_realtime.py`: the session setup and what goes into the instructions, the 48 kHz → 16 kHz
conversion, the frames and text that come back, an `error` event, a server that hangs up or goes
silent, the transcription session, and a process that must still exit when the server never answers),
the startup check of llama-server, transcripts, history
trimming and persistence (including a disk that can't be written), failed replies (before, during
and after the reply, a failure that can't be recorded, and one no one could show in the UI),
barge-in recording (driven with
fastrtc's thread-per-resume pattern), the stale-chunk drop and the reset-button stop (the real
`ReplyOnPause`, interrupted mid-`next()` from another thread), the per-connection `copy()` (its
arguments go by keyword, so a reordered fastrtc signature fails the test), server startup checks,
the removed CLI/environment setting errors, the idle polling throttle (the handler must not spin
while nobody is talking), the page heading and the chatbot's layout, and what **New conversation**
does to the history. The llamacpp tests resume the reply generator on a fresh thread each time, like
fastrtc.
The test suite uses fakes for chat and text generation. In addition, the sherpa-onnx 2.0.1 Whisper Tiny
and Kokoro assets were smoke-tested locally: Whisper transcribed the official sample WAV and Kokoro
generated finite 24 kHz audio. The complete browser microphone path still needs a manual run with the
chosen chat server.

## Layout

```
src/vaice/
  app.py            CLI + Gradio/WebRTC UI; wraps a Responder, logs time to first audio, transcript + reset
  interruptible.py  ReplyOnPause that drops the chunk generated while the user interrupts (fastrtc 0.0.34 race);
                    silences the playing reply for "New conversation"; throttles fastrtc's idle polling;
                    copies itself per connection
  lifecycle.py      the stop signal: Ctrl+C asks the app and its responders to give up
  config.py         env-driven settings (.env.example lists them with their defaults)
  conversation.py   Turn / Conversation: history, trimming, JSON persistence, failure notes
  audio.py          mic frame → mono PCM16 at the model's rate (resampled) and back to float32
  responders/
    base.py         Responder protocol (audio + history in → audio chunks out; reports transcript and spoken text)
    __init__.py     responder package
    llamacpp.py     llama-server /v1/realtime, its parallel transcription session, startup capability checks
    routing.py      known/unknown model capability representation and pipeline selection
    text.py         local Whisper + streamed OpenAI-compatible chat + local Kokoro sentence audio
tests/
  fake_realtime.py  a fake /v1/realtime server for the llamacpp tests
```

`responders/base.py` describes the audio, history, and transcript contract used by the app.

## Notes

- fastrtc polls the reply handler in a tight loop while nobody is talking (`utils.py`:
  `player_worker_decode`), which burned most of a core (~32k calls/s, measured). An idle
  `emit()` therefore waits for the pause event instead of returning at once: ~37 calls/s and
  ~1% of a core. The wait ends as soon as the pause is detected, so the reply still starts
  right away.
- llama-server sockets live on daemon threads of their own (one per session, one
  session for the reply and one for the transcript), never on the thread fastrtc resumes the
  generator on. The `websockets` sync client itself would cope with being read from a different
  thread each time (only *concurrent* `recv()` calls are refused), but `connect()` and `close()`
  can block for their timeouts, and a stuck thread of fastrtc's executor keeps the process alive
  after Ctrl+C. The generator only reads a queue, polling every 0.25 s for the stop signal and
  for the end of the turn; closing the socket is what cancels the reply on the server.
- llama-server's realtime sessions reject `conversation.item.create` with the assistant role
  ("only user messages and function call outputs are supported"), so earlier turns can't be replayed
  as items. Each reply is a fresh session with the history as text in `instructions`
  (the model uses it: "And roughly how many people live there?" after a question about Paris
  is answered about Paris). A conversation session never sends a transcript of the user's audio,
  so a second session of type `transcription` hears the same audio.
- The sample rates come from `session.created` (`audio.input.format.rate`, `audio.output.format.rate`:
  16 kHz in, 24 kHz out for LFM3-Audio), never assumed. The mic's 48 kHz audio is resampled to the
  input rate (`scipy.signal.resample_poly`) and sent in one `input_audio_buffer.append`; the pause
  detection is fastrtc's (the session's `turn_detection` is `null`).
- A reply cut off by `max_output_tokens` still ends with `response.done` status `completed`; only
  `usage.output_tokens` (at the limit) gives it away, which is what the log warning checks.
- Gradio 6.0 flips `gr.Chatbot(allow_tags)` to `True`, which would render whatever the model
  writes as HTML. `app.py` sets it to `False` explicitly, so the transcript is always plain text.
- fastrtc copies the handler for every connection, and its own `ReplyOnPause.copy()` builds a
  plain `ReplyOnPause`, which would lose the stale-chunk fix. `interruptible.py` overrides
  `copy()` and passes every argument **by keyword**, so a fastrtc release that adds, drops or
  reorders a parameter fails loudly there instead of silently handing the handler the wrong value.
