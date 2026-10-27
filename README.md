# vaice

Talk to a local model and hear it answer; the spoken reply streams back as it is generated.
The conversation carries over between questions and is kept as text history.

Two interchangeable backends produce the reply:

| Backend | How it works | Strengths |
|---|---|---|
| **native** | One speech-to-speech model ([LFM2.5-Audio-1.5B](https://huggingface.co/mlx-community/LFM2.5-Audio-1.5B-8bit) via mlx-audio) hears your audio directly and answers in audio | Fastest: first audio ≈ 0.3–0.45 s after you stop talking |
| **cascade** | Gemma (llama.cpp) transcribes your audio, then replies in text; Kokoro speaks each sentence as it completes | Much bigger model (12B) → smarter answers, better at using the conversation; ≈ 1.5–2.5 s to first audio (≈ 5 s on the first question) |

`auto` (the default) uses native if it loads and falls back to cascade, with a warning, if not.

```
browser mic ─WebRTC─▶ fastrtc (VAD: detects you stopped talking)
                         │  audio
                         ▼
                  Responder.respond(audio, history, turn) ──▶ audio chunks ─WebRTC─▶ browser speakers
                   ├─ NativeResponder   mlx-audio speech-to-speech model
                   └─ CascadeResponder  llama-server :8001 (transcribe, then reply) → sentences → Kokoro TTS
                         │ both also record your transcript and what they said on `turn`
                         ▼
                  Conversation (~/.vaice/conversation.json) ──▶ transcript in the UI
```

## Requirements

- Apple Silicon Mac (mlx-audio runs on MLX)
- Python 3.10–3.12 (spaCy, pulled in by `misaki[en]`, has no wheels for 3.13+ yet)
- [uv](https://docs.astral.sh/uv/)
- **Native backend:** nothing else; the model (~2.3 GB) downloads on first run.
- **Cascade backend:** a llama.cpp `llama-server` running an **audio-capable** model with its `--mmproj`, e.g.:

  ```bash
  llama-server --model ./gemma-4-12b/gemma-4-12B-it-UD-Q4_K_XL.gguf \
    --mmproj ./gemma-4-12b/mmproj-BF16.gguf --port 8001 \
    --chat-template-kwargs '{"enable_thinking":false}'
  ```

  **Only llama.cpp `llama-server` is supported and tested.** The cascade sends audio as an OpenAI-style
  `input_audio` content part and relies on llama-server's `/health` endpoint, so other
  OpenAI-compatible servers (Ollama, LM Studio, oMLX, …) are untested and may not accept audio at all.
  Point `VAICE_LLM_URL` at a different host or port if llama-server isn't on `:8001`.

## Setup

```bash
uv venv .venv            # picks Python 3.12 from .python-version
uv pip install -e ".[dev]"
```

## Run

```bash
.venv/bin/vaice                      # auto: native if it loads, else cascade
.venv/bin/vaice --backend cascade    # force Gemma + Kokoro (needs llama-server on :8001)
.venv/bin/vaice --backend native     # force the speech-to-speech model; fail if it can't load
```

The first run downloads models (about a minute, longer for native). When you see
`Running on local URL: http://127.0.0.1:7860`, open it in Chrome, allow the microphone,
click record, ask a question, then stop talking. Use headphones, otherwise the speakers
feed back into the mic.

The heading says which backend is answering (`native` or `cascade`), so the page tells you what
you are talking to without reading the log.

The log prints `using native|cascade backend` at startup, and for every reply
`[native] time to first audio: N.NNs` / `[native] reply finished in N.NNs`, so you can always
tell which path answered and compare the two backends. A `WARNING ... falling back to cascade`
means `auto` couldn't load native; the traceback after it says why. A
`WARNING ... llama-server not reachable` at startup means cascade replies will fail until
llama-server is up. Run it once with each `--backend` and ask the same question.

Notes on the native backend: LFM2.5-Audio is a 1.5B model, English only, so expect simpler
answers than Gemma 12B. Replies are capped at 512 tokens (text and audio frames together);
if one is cut off, the log says so.

On the native backend, Ctrl+C stops the app: the reply that is in flight is abandoned and the
process exits right away, even if the model call never returns (the log says `the app is
stopping; the reply is abandoned`). The turns that were already finished are on disk. The
cascade does not have this guarantee: its reply runs inside a blocking request to
llama-server, so Ctrl+C may wait for that request to time out (`VAICE_LLM_TIMEOUT`).

## Conversation history

Both backends remember the conversation. It is saved to `~/.vaice/conversation.json` (see
`VAICE_DATA_DIR`) after every change, so it survives page reloads and restarts, and the page shows
it as a transcript. The **New conversation** button (and the chatbot's trash icon) clears it and
silences a reply that is still playing or queued; a question you were half-way through saying when
you click it is discarded, so say it again. What you say *after* the click is the first turn of the
new conversation: it is answered as usual, but the model gets no earlier turns, because they are
gone.

- **What the model sees:** earlier turns as plain text. A turn whose transcript has not arrived
  yet (it is still being transcribed) is left out of the prompt entirely, so the model never sees
  a reply to a question it cannot read and never reads it as `(inaudible)`. A transcript that
  arrived but is empty shows `(inaudible)`.
  - **native:** the current question as audio. Your side of history is transcribed by LFM's ASR
    mode on the model's worker thread after the reply is generated, so it doesn't delay the first audio.
  - **cascade:** a transcript of the current question, so the whole conversation is text. Gemma first
    transcribes your audio (greedy, ≈ 0.4–0.8 s), then replies to the transcript. Sending the audio
    straight to Gemma works for a single question, but with earlier turns in the prompt Gemma often
    misheard the new clip or answered an earlier question; in a measured 10-turn conversation the
    text route got the follow-ups right ("And what about Germany?", "Multiply that by 10.") where the
    audio route mostly didn't. If the transcript is empty (a cough, noise), there is no reply.
  - The 1.5B native model uses the last few turns well but often forgets things said much earlier
    (e.g. your name from the first turn), even though they are in its prompt.
- **Barge-in:** if you talk over a reply, it stops when you stop talking, and history keeps the
  part of the reply that had played, marked `(interrupted)`. That part is estimated, not measured:
  the words of the audio handed out, cut to the share that the time since the first audio could
  have played (fastrtc queues audio ahead of playback). At least one word is kept, so an
  interrupted reply is never recorded as empty. The native model writes its text well
  ahead of its audio, so its text is also split by the share of audio generated. Expect it to be
  off by a few words. The chunk that was being generated when you interrupted (≈ 0.8 s native, a
  sentence in cascade) is dropped instead of played; fastrtc would queue it after clearing its
  playback queue ([#2](https://github.com/ranacseruet/vaice/issues/2)).
- **Trimming:** each prompt includes at most the last `VAICE_HISTORY_TURNS` exchanges. The oldest
  are dropped in blocks of half that, not one at a time, so the start of the prompt stays the same
  for several turns and llama-server can keep reusing its cached prefix.
- A corrupt `conversation.json` is renamed to `conversation.json.corrupt` (with a warning in the log;
  if it cannot even be renamed, vaice starts empty and leaves the file where it is)
  and a fresh conversation starts.
- **A conversation that can't be written to disk** (full disk, read-only file) is only a warning in
  the log: the reply goes on and the conversation stays in memory, but it is lost when the app stops.
  The file that was there before is left untouched.
- **A failed reply** (llama-server down or a model error) does not end the session: the log gets the
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
| `VAICE_BACKEND` | `auto` | `auto`, `cascade` or `native` (overridden by `--backend`) |
| `VAICE_PORT` | `7860` | UI port |
| `VAICE_DATA_DIR` | `~/.vaice` | where `conversation.json` is kept |
| `VAICE_HISTORY_TURNS` | `10` | past exchanges included in each prompt (`0` = none); older ones are dropped in blocks |
| `VAICE_NATIVE_MODEL` | `mlx-community/LFM2.5-Audio-1.5B-8bit` | mlx-audio speech-to-speech model; empty disables native |
| `VAICE_NATIVE_PROMPT` | `Respond with interleaved text and audio. Reply naturally and concisely in 1 to 2 spoken sentences.` | system prompt for native |
| `VAICE_LLM_URL` | `http://localhost:8001/v1/chat/completions` | llama-server chat endpoint (cascade); the `/health` check uses the same host and port. Only llama.cpp is tested |
| `VAICE_LLM_MODEL` | `gemma-4-12b` | model name sent in the request (informational for llama-server) |
| `VAICE_PROMPT` | `Reply naturally and concisely in 1 to 2 spoken sentences.` | system prompt for the reply (cascade) |
| `VAICE_LLM_TIMEOUT` | `120` | seconds (cascade) |
| `VAICE_TTS_MODEL` | `mlx-community/Kokoro-82M-bf16` | MLX Kokoro repo (cascade) |
| `VAICE_VOICE` | `af_heart` | Kokoro voice (cascade) |
| `VAICE_SPEED` | `1.1` | speech speed (cascade) |

## Tests

```bash
.venv/bin/pytest
.venv/bin/ruff check src tests
```

CI (`.github/workflows/ci.yml`) runs both on Python 3.10 and 3.12, on an Apple Silicon macOS runner
— the native tests need MLX, which has no Linux wheels — for every push to `main` and every pull
request.

Unit tests use fakes for the models: sentence splitting, SSE parsing (including llama.cpp's
`"content": null` first chunk), llama-server errors (HTTP and mid-stream), transcripts, history
trimming and persistence (including a disk that can't be written), failed replies (before, during
and after the reply, a failure that can't be recorded, and one no one could show in the UI),
barge-in recording (driven with
fastrtc's thread-per-resume pattern), the stale-chunk drop and the reset-button stop (the real
`ReplyOnPause`, interrupted mid-`next()` from another thread), the per-connection `copy()` (its
arguments go by keyword, so a reordered fastrtc signature fails the test), cascade streaming
order, native chunking/threading/cancellation and the exit path (a model call that never returns
must not keep the process alive), the idle polling throttle (the handler must not spin while
nobody is talking), the page heading and the chatbot's layout, what **New conversation** does to
the history, backend selection/fallback, and env-var parsing. The native tests are skipped where
MLX isn't installed.
There is no automated test against the real models; verify the real audio path by hand with the
steps above.

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
  audio.py          mic frame → base64 WAV / float mono
  responders/
    base.py         Responder protocol (audio + history in → audio chunks out; reports transcript and spoken text)
    __init__.py     create_responder(): auto / cascade / native selection + fallback
    cascade.py      llama.cpp text stream → sentences → Kokoro
    native.py       mlx-audio speech-to-speech model
  llm.py            llama.cpp client: streaming reply + transcription (cascade)
  sentences.py      text fragments → sentences; markdown cleanup for TTS (cascade)
  tts.py            Kokoro synthesis (cascade)
tests/
```

To add a backend, implement `respond(audio, history, turn) -> Iterator[(sample_rate, float32 samples)]`
(see the contract in `responders/base.py`: report what was spoken and the user's transcript on `turn`)
and register it in `responders/__init__.py`.

## Notes

- The native detokenizer is stateless, so each 10-frame chunk is decoded together with the
  previous 10 frames as left context and then trimmed. Decoding chunks independently
  measurably diverged from decoding the whole reply.
- MLX streams are bound to the thread that created them, but fastrtc resumes the reply
  generator from a different worker thread for each chunk. `NativeResponder` therefore loads
  and runs the model on one dedicated thread and hands chunks back through a queue. The
  cascade's Kokoro TTS is also MLX and does run on fastrtc's threads; that works only because
  Kokoro uses MLX's default stream. Put any model that creates its own MLX streams on a single
  worker thread the same way. The native transcription (ASR) runs on that same worker, right after
  the reply is generated.
- fastrtc polls the reply handler in a tight loop while nobody is talking (`utils.py`:
  `player_worker_decode`), which burned most of a core (~32k calls/s, measured). An idle
  `emit()` therefore waits for the pause event instead of returning at once: ~37 calls/s and
  ~1% of a core. The wait ends as soon as the pause is detected, so the reply still starts
  right away.
- The native MLX worker is a **daemon** thread, not a `ThreadPoolExecutor`: at exit the
  interpreter joins a thread pool's workers (`concurrent.futures.thread._python_exit`) and those
  threads are not daemons, so a model call that never returns would keep the app alive. A daemon
  thread is abandoned with the process instead, and Ctrl+C asks the reply in flight to give up
  (`lifecycle.py`).
- Gemma needs the text prompt **before** the audio in the message content (the cascade's
  transcription request).
- Raw 48 kHz browser audio works as-is on both backends: llama.cpp and the mlx-audio
  processor each resample internally.
- llama.cpp's `llama-server` only outputs text; it ignores `modalities`/`audio` request
  fields, which is why the cascade needs a separate TTS step.
- The cascade speaks one sentence at a time, so the first audio only comes when the first
  sentence is complete. A sentence ends at `.`, `!` or `?` followed by whitespace, except
  after a title (`Dr.`), an initialism (`U.S.`), a list number, `etc.` and `a.m.`/`p.m.`,
  which end a sentence only when the next word is capitalized ("at 3 p.m. Tomorrow" is two
  sentences, "at 7 a.m. and then" is one).
- Gradio 6.0 flips `gr.Chatbot(allow_tags)` to `True`, which would render whatever the model
  writes as HTML. `app.py` sets it to `False` explicitly, so the transcript is always plain text.
- fastrtc copies the handler for every connection, and its own `ReplyOnPause.copy()` builds a
  plain `ReplyOnPause`, which would lose the stale-chunk fix. `interruptible.py` overrides
  `copy()` and passes every argument **by keyword**, so a fastrtc release that adds, drops or
  reorders a parameter fails loudly there instead of silently handing the handler the wrong value.
- Kokoro's sample rate is fixed by the model, so it lives in `tts.py` (`SAMPLE_RATE`), not in the
  env-driven settings in `config.py`.
