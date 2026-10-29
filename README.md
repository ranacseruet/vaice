# vaice

Talk to a local model and hear it answer; the spoken reply streams back as it is generated.
The conversation carries over between questions and is kept as text history.

vaice uses one speech-to-speech model ([LFM3-Audio-2B](https://huggingface.co/LiquidAI/LFM3-Audio-2B-GGUF) or
another compatible model) on a llama.cpp `llama-server`. The server hears your audio and streams its spoken
reply over the `/v1/realtime` WebSocket. Choose and run the model separately; vaice does not bundle a model.

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

## Requirements

- Python 3.10–3.12 (what CI tests; the dependencies also resolve on 3.13 and 3.14, but nothing was run there)
- [uv](https://docs.astral.sh/uv/)
- A llama.cpp `llama-server` **b11960 or newer** (v0.8.0) running a model with audio output. It needs the
  `/v1/realtime` WebSocket. At the time of writing, Homebrew's llama.cpp (0.7.0) is too old, so build it
  from source:

  ```bash
  git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
  cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j --target llama-server
  build/bin/llama-server -hf LiquidAI/LFM3-Audio-2B-GGUF:Q4_K_M --port 8001
  ```

  **Only llama.cpp `llama-server` is supported and tested.** vaice speaks its `/v1/realtime` events and reads
  `/health` and `/v1/models` to check that the selected model advertises audio output. Other servers (Ollama,
  LM Studio, oMLX, …) are untested and will not work. Point `VAICE_LLAMACPP_URL` at a different host or port
  if llama-server isn't on `:8001`. The CI runner is macOS; Linux and Windows audio behavior has not been
  validated.

  Current upstream llama.cpp documentation describes `/v1/realtime` conversation support for LFM3-Omni and
  the `--reasoning off` server option. vaice's integration has been exercised with LFM3-Audio; Omni behavior
  is not verified end to end here. On llama.cpp builds that provide the option, add `--reasoning off` to the
  server command if your selected model's chat template supports disabling reasoning. This is a server startup
  option, not a vaice setting.
  See the [llama.cpp server documentation](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md).

## Setup

```bash
uv sync --extra dev      # creates .venv from uv.lock: the exact versions CI tests
```

`uv sync` installs from the committed `uv.lock`, so every setup gets the same dependency
versions (`.python-version` picks Python 3.12; add `--python 3.10` for another supported
version). A plain `uv pip install -e ".[dev]"` or `pip install -e ".[dev]"` also works, but
it resolves fresh from the ranges in `pyproject.toml` — the lockfile pins only `uv sync`
installs of this repository, not downstream installs of the published package.

## Dependency updates

Nothing bumps itself: refresh the lock deliberately, test the new set, and commit the lock.

```bash
uv lock --upgrade-package fastrtc   # or --upgrade to refresh everything
uv sync --extra dev
.venv/bin/pytest && .venv/bin/ruff check src tests
```

Then push the regenerated `uv.lock` on a branch: CI re-verifies the whole locked set on
Python 3.10 and 3.12. Non-locked installs (`pip`, `uv pip`) ignore `uv.lock` and follow the
ranges in `pyproject.toml`, which is why parts of the set are bounded there. Known
compatibility limits:

- **fastrtc** — `interruptible.py` and `tests/test_interruptible.py` are built against
  fastrtc 0.0.34's `ReplyOnPause` internals, so `pyproject.toml` admits only patch releases
  of that line (`<0.1`); a newer minor or major needs a tested update of both the code and
  the range.
- **gradio** — declared directly because `app.py` imports it; held below 6.0 because
  fastrtc 0.0.34 requires `gradio<6.0` and Gradio 6.0 flips `gr.Chatbot(allow_tags)` to
  `True` (which `app.py` also guards against explicitly).
- **onnxruntime** (transitive, via `fastrtc[vad]`) — held below 1.24 for Python 3.10 with a
  marker dependency: from 1.24 it declares `>=3.10` support but only ships cp311+ wheels, so
  3.10 installations must use the 1.23 line; 3.11/3.12 take any release.

## Run

```bash
.venv/bin/vaice                      # connect to llama-server at VAICE_LLAMACPP_URL or localhost:8001
```

llama-server downloads its model with `-hf`. When you see
`Running on local URL: http://127.0.0.1:7860`, open it in Chrome, allow the microphone,
click record, ask a question, then stop talking. Use headphones, otherwise the speakers
feed back into the mic.

The page identifies llama-server as the active voice service. At startup, vaice checks the server:
an unreachable or still-loading server is logged and the UI starts, so it can become available
before your first question. A reachable server whose model does not advertise audio output is rejected.
For each reply the log prints `[llamacpp] time to first audio: N.NNs` and `[llamacpp] reply finished in N.NNs`.
Replies are capped at 512 tokens (text and audio frames together); the log reports truncation.

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
  - The current question is sent as audio; earlier turns are text in the session instructions
    (`User: …` / `Assistant: …` lines after the prompt). llama-server's realtime sessions refuse earlier
    assistant turns as conversation items, so every reply opens a fresh session and vaice stays the
    single owner of the history (windowing, barge-in cuts and **New conversation** all apply). A second,
    transcription session processes the same audio in parallel with the reply. No language is set, so
    the model detects it.
  - The small models use the last few turns well but often forget things said much earlier
    (e.g. your name from the first turn), even though they are in the prompt.
- **Barge-in:** if you talk over a reply, it stops when you stop talking, and history keeps the
  part of the reply that had played, marked `(interrupted)`. That part is estimated, not measured:
  the words of the audio handed out, cut to the share that the time since the first audio could
  have played (fastrtc queues audio ahead of playback). At least one word is kept, so an
  interrupted reply is never recorded as empty. The LFM models write their text well
  ahead of their audio, so the text is also split by the share of audio received so far. Expect it to be
  off by a few words. The chunk that was being generated when you interrupted is dropped instead of played;
  fastrtc would queue it after clearing its
  playback queue ([#2](https://github.com/ranacseruet/vaice/issues/2)). And if the reply happens to
  end in the very moment you interrupt, fastrtc's end-of-reply reset would wipe your question with
  it; vaice restores it, so the question still gets its reply
  ([#13](https://github.com/ranacseruet/vaice/issues/13)) — a **New conversation** click, or an even
  newer utterance you have started saying, always wins over that restore.
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
| `VAICE_LLAMACPP_URL` | `http://localhost:8001` | llama-server URL; only scheme, host and port are used (`/health`, `/v1/models` and `ws://…/v1/realtime` hang off it). The old `VAICE_LLM_URL` alias, including URLs with paths, still works |
| `VAICE_LLAMACPP_PROMPT` | `Reply naturally and concisely in 1 to 2 spoken sentences.` | reply instructions; vaice adds `Respond with interleaved text and audio.` in front. The old `VAICE_LLM_PROMPT` and `VAICE_PROMPT` aliases still work |
| `VAICE_LLAMACPP_TIMEOUT` | `120` | seconds to open a connection or the longest silence from llama-server before a reply fails; the old `VAICE_LLM_TIMEOUT` alias still works |

`VAICE_LLAMACPP_URL`, `VAICE_LLAMACPP_PROMPT` and `VAICE_LLAMACPP_TIMEOUT` replaced `VAICE_LLM_URL`,
`VAICE_LLM_PROMPT` (and before it `VAICE_PROMPT`) and `VAICE_LLM_TIMEOUT`. Both spellings work; if you set
both, the new name wins, so an old `.env` keeps working while you migrate.

`VAICE_BACKEND` and `--backend` were removed with backend selection. If `VAICE_BACKEND` is still set,
vaice stops at startup with instructions to remove it; it does not silently switch an old native selection
to llama-server. `VAICE_NATIVE_MODEL` and `VAICE_NATIVE_PROMPT` are no longer read. Remove those lines from
your `.env`; start a compatible llama-server and set `VAICE_LLAMACPP_URL` only if it uses a different URL.

`VAICE_LLM_MODEL`, `VAICE_TTS_MODEL`, `VAICE_TTS_VOICE` (`VAICE_VOICE`) and `VAICE_TTS_SPEED` (`VAICE_SPEED`)
belonged to the old Gemma + Kokoro pipeline; nothing reads them any more, so setting them is harmless and
has no effect. The voice is the model's own.

## Tests

```bash
.venv/bin/pytest
.venv/bin/ruff check src tests
```

CI (`.github/workflows/ci.yml`) installs the locked dependencies (`uv sync --locked`) and runs
Ruff and pytest on Python 3.10 and 3.12 on macOS for every push to `main` and every pull request.

Unit tests use fakes for the models: the llamacpp backend against a fake `/v1/realtime` server
(`tests/fake_realtime.py`: the session setup and what goes into the instructions, the 48 kHz → 16 kHz
conversion, the frames and text that come back, an `error` event, a server that hangs up or goes
silent, the transcription session, and a process that must still exit when the server never answers),
the startup check of llama-server, transcripts, history
trimming and persistence (including a disk that can't be written), failed replies (before, during
and after the reply, a failure that can't be recorded, and one no one could show in the UI),
barge-in recording (driven with
fastrtc's thread-per-resume pattern), the stale-chunk drop, the lost-utterance restore (the real
`ReplyOnPause`, interrupted mid-`next()` from another thread) and the reset-button stop (the real
`ReplyOnPause`, interrupted mid-`next()` from another thread), the per-connection `copy()` (its
arguments go by keyword, so a reordered fastrtc signature fails the test), server startup checks,
the removed CLI/environment setting errors, the idle polling throttle (the handler must not spin
while nobody is talking), the page heading and the chatbot's layout, and what **New conversation**
does to the history. The llamacpp tests resume the reply generator on a fresh thread each time, like
fastrtc.
There is no automated test against the real models; verify the real audio path by hand with the
steps above.

## Layout

```
src/vaice/
  app.py            CLI + Gradio/WebRTC UI; wraps a Responder, logs time to first audio, transcript + reset
  interruptible.py  ReplyOnPause that drops the chunk generated while the user interrupts and keeps the
                    interrupting utterance when the reply ends mid-interrupt (fastrtc 0.0.34 races);
                    silences the playing reply for "New conversation"; throttles fastrtc's idle polling;
                    copies itself per connection
  lifecycle.py      the stop signal: Ctrl+C asks the app and its responders to give up
  config.py         env-driven settings (.env.example lists them with their defaults)
  conversation.py   Turn / Conversation: history, trimming, JSON persistence, failure notes
  audio.py          mic frame → mono PCM16 at the model's rate (resampled) and back to float32
  responders/
    base.py         Responder protocol (audio + history in → audio chunks out; reports transcript and spoken text)
    __init__.py     responder package
    llamacpp.py     llama-server /v1/realtime: a reply session and a transcription session per turn, startup check
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
  plain `ReplyOnPause`, which would lose the barge-in fixes. `interruptible.py` overrides
  `copy()` and passes every argument **by keyword**, so a fastrtc release that adds, drops or
  reorders a parameter fails loudly there instead of silently handing the handler the wrong value.
