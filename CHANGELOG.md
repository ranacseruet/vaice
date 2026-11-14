# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `VAICE_LLAMACPP_TOOLS`: voice access to llama-server's built-in tools over `/v1/realtime`. Tool definitions are fetched from the server's `GET /tools` at startup, declared in each `session.update` (flat Responses API format, `tool_choice: "required"`), a spoken question's `function_call` events are executed through the server's `POST /tools`, and the model speaks the result. Off by default; `builtin` takes every tool the server advertises. Validated live against llama-server b12359 with LFM3-Audio: the call rate depends on the prompt (default prompt ≈ 1 call in 6 runs; a "use the available tools" prompt 4 in 4), `auto` never calls, 3+ tools hallucinate, a tool round is capped at 2, and executor failures are spoken, not raised.
- Tool results are compacted before they are fed back (`_speech_friendly`: JSON → short `key: value` pairs, values truncated at 40 chars, the whole output capped at 120 chars) — the voice model reads the fed-back result aloud (a 2B-model limitation; prompt lines forbidding it have no measurable effect), so the compaction keeps the spoken dictation short while the first sentence answers.
- The UI shows tool calls and their results in a collapsed *Tool activity* panel under the chatbot (the last 10, with arguments cut); the conversation itself stays speech-only.

### Changed
- `VAICE_LLAMACPP_TOOLS` is now on/off (`builtin` = every tool the server advertises, unset = off): the tool set is the server operator's decision (`llama-server --tools`), which is also what `POST /tools` executes (an unadvertised tool is refused by the server, live-verified 404) — the client-side subset and refusal guard were a second source of truth and are removed. A comma-separated value now exits with a message pointing at the server's `--tools`.

### Fixed
- Docs honesty: the echo troubleshooting entry and the "Language & voice" note blamed llama-server's `--tts-lang`/`--tts-voice` for making LFM3-Audio parrot user speech, citing a repro. A controlled re-test on the same build (b12359, `Q8_0 --temp 0`) did not reproduce the parroting — with the flags, with per-session `audio.output.language`/`audio.output.voice`, or with the exact injected prompt line on a flagless server, across clean TTS and noisy mic inputs (`tasks/repro_tts_lang.py`). The flags are no longer presented as a proven echo cause; the advice to omit them stands (English-only voice model, the pin buys nothing). History clearing and `--temp 0` remain the echo fixes.

## [0.1.0] - 2026-10-29

### Added
- Real-time speech-to-speech voice UI powered by `fastrtc` (WebRTC + VAD), Gradio, and `llama.cpp` `llama-server` (`/v1/realtime` WebSocket with `LFM3-Audio` and compatible audio-output models).
- Parallel `/v1/realtime` transcription session per turn with automatic language detection and live UI refresh when a transcript lands after audio finishes.
- Persistent multi-turn conversation history (`~/.vaice/conversation.json`) with block-based window trimming (`VAICE_HISTORY_TURNS`), atomic writes, corrupt-file recovery, and a **New conversation** reset control that silences in-flight playback.
- `InterruptSafeReplyOnPause` wrapper fixing `fastrtc` 0.0.34 barge-in races (stale-chunk drop and end-of-reply utterance restoration) and throttling idle `emit()` polling from ~32k calls/s down to ~37 calls/s.
- Barge-in history trimming based on audio duration handed out vs elapsed playback time.
- Startup health and audio-modality validation against `llama-server` (`/health` and `/v1/models`), plus clean `Ctrl+C` shutdown via daemon WebSocket threads.
- Reproducible dependency lockfile (`uv.lock`) and CI workflow testing Python 3.10 and 3.12 on macOS.
