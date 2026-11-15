# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- A `/v1/realtime` WebSocket probe in `check_server`: after the health and model checks pass, the startup check opens the realtime endpoint once (2 s timeout) and closes it. A server whose HTTP endpoints answer but which does not route `/v1/realtime` (an older llama.cpp, a proxy) now fails at startup with an actionable message instead of at the first question after a 120 s stall. A rejected handshake (e.g. 404) raises — it is permanent; a connection-level failure only warns, like the health check.
- `VAICE_LLAMACPP_TOOLS`: voice access to llama-server's built-in tools over `/v1/realtime`. Tool definitions are fetched from the server's `GET /tools` at startup, declared in each `session.update` (flat Responses API format, `tool_choice: "required"`), a spoken question's `function_call` events are executed through the server's `POST /tools`, and the model speaks the result. Off by default; `builtin` takes every tool the server advertises. Validated live against llama-server b12359 with LFM3-Audio: the call rate depends on the prompt (default prompt ≈ 1 call in 6 runs; a "use the available tools" prompt 4 in 4), `auto` never calls, 3+ tools hallucinate, a tool round is capped at 2, and executor failures are spoken, not raised.
- Tool results are compacted before they are fed back (`_speech_friendly`: JSON → short `key: value` pairs, values truncated at 200 chars, the whole output capped at 800 chars) — the voice model reads the fed-back result aloud (a 2B-model limitation; prompt lines forbidding it have no measurable effect), so the compaction keeps the spoken dictation short while the first sentence answers. The caps are configurable with `VAICE_LLAMACPP_TOOL_VALUE_CHARS` and `VAICE_LLAMACPP_TOOL_OUTPUT_CHARS`; the old 40/120 defaults cut a `get_info` timestamp off before the model could speak it.
- A startup warning names any advertised tool the server marks `permissions.write` (`exec_shell_command`, `write_file`, `edit_file`): they run with the server's own permissions, so the server's `--tools` should be a read-only subset for a voice demo.
- The UI shows tool calls and their results in a collapsed *Tool activity* panel under the chatbot (the last 10, with arguments cut); the conversation itself stays speech-only.
- Every reply's instructions now carry the current local date and time (built per reply; the model has no clock of its own). A bare `Current date and time: …` line was treated as inert text — the model echoed it and still answered "I don't have access" (live A/B, 6 runs) — so the line is phrased as something the model knows and is told to use; that wording answered with the real time 6/6.
- A reply loop breaker: under `--temp 0` the model sometimes cannot stop after answering and repeats the same short block to the token cap (seconds of junk audio). If the same 1–4-line block repeats three times back-to-back (compared with case and punctuation ignored), the reply is cut there: the repeated tail is dropped from the transcript and the realtime session is closed, freeing the server slot.

### Changed
- Around a tool round, only the final round's text is recorded: whatever the model says before a tool result (the working-out-loud preamble) is spoken all the same, but it is not written to the conversation — the text and the spoken-part share restart at each tool-round boundary, so a wrong first pass cannot reinforce itself in history.
- Tool-call follow-ups no longer inherit the session's `tool_choice: "required"`: after a tool result the next response is requested with `auto`, and after the second round a final answer is forced with `none` (a call in that forced round is ignored). The first response still uses `required`.
- The tool-result compaction defaults are raised from 40/120 to 200/800 characters, so answer-bearing data (an OS build timestamp, the end of a file) survives the round trip.
- `VAICE_LLAMACPP_TOOLS` is now on/off (`builtin` = every tool the server advertises, unset = off): the tool set is the server operator's decision (`llama-server --tools`), which is also what `POST /tools` executes (an unadvertised tool is refused by the server, live-verified 404) — the client-side subset and refusal guard were a second source of truth and are removed. A comma-separated value now exits with a message pointing at the server's `--tools`.

### Fixed
- A tool round could not end in an answer: the follow-up response inherited `tool_choice: "required"`, so the model called again, repeated the raw result in its text, and could loop until the token cap (live: the same sentence repeated to 1024 tokens, 62.7 s of audio).
- A reply that hit the 2-round tool cap now gets a forced final answer instead of stopping after the last call.
- The "reply had no audio" warning no longer fires for a tool-call response, which has no audio by design.
- `read_file` (and any tool with parameters) never received its arguments: the executor sent the model's JSON string as `arguments`, while `POST /tools` reads the parameters object from `params`. `get_info` worked only because it takes none; `read_file` got `key 'path' not found`. The arguments are now parsed and sent as `params`.
- A plain-text `read_file` result lost its last word when the per-value compaction cap (200) cut a 210-character file; a `plain_text_response` payload is now bounded by the whole-output cap (800) instead.
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
