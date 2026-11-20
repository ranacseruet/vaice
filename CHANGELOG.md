# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

The net change since 0.1.0, which was a spoken chat with no tools. Features that were built and removed again before
this release (a separate text model with `/tts` speech, a helper model, spoken progress lines, spoken yes/no
approval, the MCP `auto` and `voice` policies, llama-server's built-in tools) are not listed; their history is in
the pull requests (#53 to #126).

### Added
- **A voice assistant for the Mac.** Every reply is one speech-to-speech session whose tools are vaice's own skills,
  run by vaice with its own permissions (#94, #125): disk space, battery, system info, top processes, finding files,
  opening and quitting apps, volume and mute (#96). The model reads a skill's result and speaks the answer; results are
  compacted first (`VAICE_LLAMACPP_TOOL_VALUE_CHARS`, `VAICE_LLAMACPP_TOOL_OUTPUT_CHARS`, #54). A reply makes at most
  two tool rounds and always ends in an answer (#76, #78).
- **Button-only confirmation.** A skill that cannot be undone (`quit_app`, `run_shell`, `forget_everything`, MCP tools
  with the `click` policy, browser steps) becomes a request on the screen with its exact arguments; only **Run** runs
  the stored call, and the outcome is then spoken (#96, #115, #125). Nothing is approved by voice.
- **Opt-in skills**, each off by default:
  - shell commands, `VAICE_SHELL` (#97);
  - long-term memory of explicit notes, `VAICE_MEMORY` (#101, #8);
  - screen vision, `look_at_screen` through `VAICE_VISION_URL` (#103, #107, #108);
  - MCP tools from stdio servers with a `run` or `click` policy per tool, `VAICE_MCP_CONFIG` behind the `mcp` extra
    (#104);
  - browser tasks through the Pi agent, `VAICE_BROWSE=pi` with its own `VAICE_BROWSE_URL` model (#117, #118, #121,
    #122, #123): they run in the background, quietly, a step that changes a page needs **Run**, and
    `browser_status` / `stop_browsing` and a Stop button report on and end them.
- **`vaice up`**: one command that starts `llama-server` when none is running and opens the UI (#89).
- **A live eval** (`evals/run.py`) that measures how reliably spoken requests become the right skill call, with repeats
  (#88).
- The *Tool activity* panel (#55); the current date and time in every reply's instructions (#78); a loop breaker that
  cuts a reply repeating itself (#78); a `/v1/realtime` probe and a temperature warning at startup (#47, #74, #81).
- Settings: `VAICE_VAD_THRESHOLD`, `VAICE_VAD_MIN_SILENCE_MS`, `VAICE_VAD_SPEECH_PAD_MS`, `VAICE_CAN_INTERRUPT` (#49),
  `VAICE_LLAMACPP_MAX_OUTPUT_TOKENS` (#48), a `.env` file in the working directory (#51), and `VAICE_DATA_DIR=""` to keep
  everything in memory (#84).

### Changed
- **Default model: LFM3-Omni-2B `Q8_0`** (`VAICE_LLAMACPP_MODEL`; LFM3-Audio-2B still works). It also reads images, so
  one server can be the vision model (#106).
- History goes to the model as ChatML turns, and the VAD is tuned with a 0.3 s pre-roll so opening words are not clipped
  (#46).
- `VAICE_LLAMACPP_PROMPT` is now `VAICE_PROMPT`, with a default written for the assistant. An old or removed setting
  that is still set is ignored and named in one startup warning (#124, #125; see `docs/configuration.md`).
- **Python 3.12 only** (was 3.10 to 3.12, #102). CI can be started by hand (#105).
- The documentation lives in `docs/` (#106); the code in `src/vaice` is flat (`realtime.py`, `assistant.py`, #126).

### Fixed
- The transcript and the end of every reply were reported 1 s late, waiting on a close handshake `llama-server` never
  completes (#90).
- A transcription session whose server never answered the upgrade held its thread for the whole timeout (#71).
- Model markup (`<tool_call>`, `</think`, control tokens) no longer reaches the recorded text (#77).
- The echo troubleshooting entry no longer blames `--tts-lang` / `--tts-voice`: a controlled re-test did not reproduce
  it (#52).

## [0.1.0] - 2026-10-29

### Added
- Real-time speech-to-speech voice UI powered by `fastrtc` (WebRTC + VAD), Gradio, and `llama.cpp` `llama-server` (`/v1/realtime` WebSocket with `LFM3-Audio` and compatible audio-output models).
- Parallel `/v1/realtime` transcription session per turn with automatic language detection and live UI refresh when a transcript lands after audio finishes.
- Persistent multi-turn conversation history (`~/.vaice/conversation.json`) with block-based window trimming (`VAICE_HISTORY_TURNS`), atomic writes, corrupt-file recovery, and a **New conversation** reset control that silences in-flight playback.
- `InterruptSafeReplyOnPause` wrapper fixing `fastrtc` 0.0.34 barge-in races (stale-chunk drop and end-of-reply utterance restoration) and throttling idle `emit()` polling from ~32k calls/s down to ~37 calls/s.
- Barge-in history trimming based on audio duration handed out vs elapsed playback time.
- Startup health and audio-modality validation against `llama-server` (`/health` and `/v1/models`), plus clean `Ctrl+C` shutdown via daemon WebSocket threads.
- Reproducible dependency lockfile (`uv.lock`) and CI workflow testing Python 3.10 and 3.12 on macOS.
