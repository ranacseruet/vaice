# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`vaice` is a single-user voice demo: browser mic → fastrtc (WebRTC + VAD) → a `Responder` → spoken reply streamed back, with a persisted multi-turn conversation. The native backend hears the mic audio directly; the cascade transcribes it first (see Gotchas). Apple Silicon only (MLX), Python 3.10–3.12 (spaCy has no 3.13+ wheels). See `README.md` for the user-facing description and the env-var table.

## Commands

```bash
uv venv .venv && uv pip install -e ".[dev]"   # setup
.venv/bin/pytest                               # all tests
.venv/bin/pytest tests/test_native.py -k left_context   # single test
.venv/bin/vaice [--backend auto|cascade|native]    # run; UI at http://127.0.0.1:7860
```

There is no linter or formatter configured. Unit tests use fakes for the models; native tests skip where MLX isn't installed. There is **no automated test against the real models or the browser mic**: verify the audio path by hand (run each `--backend`, ask the same question, check the `[native] time to first audio` log lines).

## Architecture

`app.py` wraps a `Responder` (`responders/base.py`: `respond(audio, history, turn) -> Iterator[(sample_rate, float32 samples)]`) in a fastrtc `ReplyOnPause` handler, logs time to first audio, and shows the conversation in a Gradio chatbot with a "New conversation" button. `conversation.py` owns history: `Conversation.start_turn()` closes the previous turn (unfinished = interrupted), `history()` gives the windowed text turns, and everything is saved to `~/.vaice/conversation.json`. Responders report back on the `Turn`: the user transcript (exactly once, even on failure) and the text whose audio was handed out. `responders/__init__.py:create_responder()` picks the backend: `auto` tries native and falls back to cascade with a warning; `native` fails hard; `cascade` skips native. A new backend = implement `Responder` (honour the `Turn` contract in `base.py`) and register it there.

- **`NativeResponder`** (`responders/native.py`): one speech-to-speech model (LFM2.5-Audio via mlx-audio). History goes in as text turns; LFM's ASR mode transcribes the question on the worker after the reply.
- **`CascadeResponder`** (`responders/cascade.py`): `llm.transcribe` (audio-capable Gemma on llama.cpp `llama-server` + `--mmproj`, greedy) → `llm.stream_reply` streams a text reply to the transcript with history → `sentences.py` splits into sentences / strips markdown → `tts.py` speaks each with Kokoro. `llama-server` ignores audio-output request fields, hence the separate TTS.

Config is entirely env-driven in `config.py` (`VAICE_*`); `--backend` overrides `VAICE_BACKEND`.

## Gotchas (each has bitten this repo)

- **MLX threading:** MLX streams are bound to the creating thread, but fastrtc resumes the reply generator on a different worker thread per chunk. `NativeResponder` therefore loads and runs the model on one dedicated worker thread and hands chunks back through a queue. The cascade's Kokoro TTS (also MLX) does run on fastrtc's threads; that works only because Kokoro uses the default stream. Any model that creates its own MLX streams must go through a single worker like `NativeResponder`. When testing code that runs inside a framework executor, reproduce its real threading (fresh thread per resume) and confirm the test fails against known-bad code first (see `tasks/lessons.md`).
- **Native chunk decoding:** the detokenizer is stateless, so each 10-frame chunk is decoded with the previous 10 frames as left context, then trimmed. Independent chunk decoding audibly diverged.
- **Single connection:** fastrtc's `Stream` defaults to `concurrency_limit=1`, so a second WebRTC connection is refused (`concurrency_limit_reached`), not queued. Barge-in: fastrtc's `close()` on the reply generator fails ("generator already executing") because `emit()` is mid-`next()` on another thread; the reply actually stops when CPython finalizes the dropped generator. Native replies are capped at 512 tokens (text + audio frames); truncation is logged. On barge-in, the next `start_turn()` freezes the old turn, so a late finalization can't change history; recorded text is what was handed out, not necessarily played (#2).
- **Gemma content order:** the text prompt must come **before** the audio in the message content.
- **Gemma + history:** with earlier turns in the prompt, Gemma often mishears a new audio clip or answers an earlier question, so the cascade transcribes first and keeps the conversation in text. Transcribe with `temperature: 0`; the server's default sampling mishears words.
- **mlx-audio LFM multi-turn:** `generate_from_chat_state` can't replay earlier assistant audio and prefills from scratch each call, so history is text and only the current clip is audio (one clip per `ChatState`).
- **Input audio:** raw 48 kHz browser audio works as-is on both backends (each resamples internally); `audio.py` handles int16 → float scaling and base64 WAV.

## Repo conventions

- `tasks/todo.md` holds the plan/review notes for the current change; `tasks/lessons.md` records lessons after user corrections. Read `lessons.md` at session start.
- Behavior/config changes must update `README.md` (env-var table, layout) in the same change.
