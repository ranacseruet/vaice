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
- **Architecture diagrams and an interactive explorer.** `docs/architecture.md` draws the module imports, one reply,
  a tool round, a Run click, the outbox and `emit()`, the browser task over ACP, every named thread and barge-in in
  Mermaid, and shows a `conversation.json` turn and the ChatML instructions. `docs/architecture-explorer.html` is the
  same model to click through: each module's job, key symbols and rules, a step-by-step view of six flows by thread,
  the threads and three state machines. `tests/test_architecture_docs.py` fails when either names a module, symbol
  or thread that no longer exists or misses one.
- **A status line while a browser task runs.** Beside *Stop the browser task*: the goal, the time run against the
  limit (`2:14 of 10:00`), the tool calls so far, the latest host it opened and "waiting for Run" while a step waits
  for approval. The 1 s timer now re-sends only the components whose own view changed, so the ticking time does not
  redraw the chat.
- **Follow-ups about a finished browser task.** `browser_status` also reports the last ended task (its goal, how long
  ago it ended and how; a finished one adds its spoken summary and more of the answer, capped at 700 characters), so a
  follow-up such as "What did you find?" can be answered from it (on `evals/browse_flow.py` the follow-up did what it
  should in 26/36 and 27/36 runs, 21/36 and 22/36 before; it still starts a new task in about a third of the runs
  that should be answered from the result). It is kept in memory only: a new task
  replaces it and *New conversation* or a restart forgets it. A task cancelled by Stop or exit counts as stopped even
  if its agent still returned an answer. `evals/browse_flow.py` gets four follow-up cases and an `f-right` score.
- **Browser task results.** The reply that starts a browser task ends at the call and vaice has a fixed sentence
  said through a fresh speech session (the model no longer talks on and invents results, calls `browse` twice or follows it with `look_at_screen`, #131, #132); the outcome spoken is the agent's own `Summary:` line (or
  the answer without its narration); the whole answer appears in the chat as its own message and is saved with the
  turn (`"reports"` in `conversation.json`, not in the model's history); the *Tool activity* outcome row starts with
  `finished: `. `evals/browse_flow.py --outcome report|say` compares the two outcome wordings.
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
    #122, #123), or any [ACP](https://agentclientprotocol.com) agent, `VAICE_BROWSE=acp` with `VAICE_BROWSE_AGENT`
    (both behind the `browse` extra): they run in the background, quietly; with Pi a step that changes a page needs
    **Run** (another agent asks only what its own config makes it ask: OpenCode, checked, does not ask before every
    step); `browser_status` / `stop_browsing` and a Stop button report on and end them.
- **`vaice up`**: one command that starts `llama-server` when none is running and opens the UI (#89).
- **A live eval** (`evals/run.py`) that measures how reliably spoken requests become the right skill call, with repeats
  (#88); research requests in `evals/browse_cases.json`, and `evals/browse_live.py`, an opt-in check that the browser
  agent finishes research tasks with the configured model (headless, muted, never in CI); `evals/browse_flow.py`,
  which scores a whole browser task as the user hears it (the start reply, the spoken outcome, the chat, a follow-up)
  with a fake agent that replays recorded answers.
- The *Tool activity* panel (#55); the current date and time in every reply's instructions (#78); a loop breaker that
  cuts a reply repeating itself (#78); a `/v1/realtime` probe and a temperature warning at startup (#47, #74, #81).
- Settings: `VAICE_VAD_THRESHOLD`, `VAICE_VAD_MIN_SILENCE_MS`, `VAICE_VAD_SPEECH_PAD_MS`, `VAICE_CAN_INTERRUPT` (#49),
  `VAICE_LLAMACPP_MAX_OUTPUT_TOKENS` (#48), a `.env` file in the working directory (#51), and `VAICE_DATA_DIR=""` to keep
  everything in memory (#84).

### Changed
- **The last browser result is in the prompt.** While the last browser task finished with an answer, every reply's
  system block carries one line with its goal, how long ago it ended and its spoken summary (the agent's `Summary:` line, or the start of its report; cleaned of chat
  markers, capped at 400 characters, nothing else of the answer), so "What did you find?" is answered from it instead
  of starting a new task (#136). It goes with the task: a new task, *New conversation* or a restart forgets it, and a
  stopped, declined or failed task adds nothing.
- **`VAICE_BROWSE_SECONDS` defaults to 600** (was 180): research tasks took 79 to 302 s live and two of six ran past
  180.
- **Default model: LFM3-Omni-2B `Q8_0`** (`VAICE_LLAMACPP_MODEL`; LFM3-Audio-2B still works). It also reads images, so
  one server can be the vision model (#106).
- History goes to the model as ChatML turns, and the VAD is tuned with a 0.3 s pre-roll so opening words are not clipped
  (#46).
- `VAICE_LLAMACPP_PROMPT` is now `VAICE_PROMPT`, with a default written for the assistant. An old or removed setting
  that is still set is ignored and named in one startup warning (#124, #125; see `docs/configuration.md`).
- **Python 3.12 only** (was 3.10 to 3.12, #102). CI can be started by hand (#105).
- CI on a push to `main` skips the macOS job when a passed run already tested the same tree, usually the PR's last
  run: the squash merge is a new commit, and the second run often found no macOS runner (#150, #91).
- The documentation lives in `docs/` (#106); the code in `src/vaice` is flat (`realtime.py`, `assistant.py`, #126).

- **Browser tasks speak ACP.** vaice talks to the browser agent with the Agent Client Protocol (the official Python SDK,
  the optional `browse` extra: `uv sync --extra browse`) instead of Pi's own JSONL protocol. Pi is now reached through
  `pi-acp` (pinned, started with `npx`, so Node.js is needed); a Pi extension still gates its tools, checked at startup
  and by a marker before each task's first tool call. With Pi the Chrome window now closes between tasks (`pi-acp` ends the
  previous session); the profile, and so logins, is kept.
- **Research goes to the browser.** The `browse` description now names looking things up, research, news and prices,
  not only opening a site: before, the assistant said it could not browse the web to most research requests (0 of 30
  with the screen skill on). Each task's prompt also carries today's date, so "this week" means now.

### Fixed
- *Tool activity* now uses dynamic backtick fencing for arguments and output so untrusted tool results cannot break out of code blocks, and UI chat/report rendering escapes raw HTML tags so `<img>` and similar elements cannot trigger remote fetches (#161).
- `realtime._clean_turn_text` now strips all `<|...|>` special tokens, unclosed fragments, and legacy markers iteratively so nested tokens like `<|<|im_start|>im_start|>` cannot bypass history sanitization (#162).
- The browser task gate (`browse.GATE_POLICY`, `browse_gate.ts`) now uses an explicit allowlist of read/navigation tools instead of a wildcard rule, escalates `navigate_page` / `new_page` with `initScript` or non-`http(s)` URLs to **Run**, and passes `--no-file-navigations` and `--no-javascript-evaluation` to `chrome-devtools-mcp` so `file://` navigations, `evaluate_script`, `set_logpoints`, `initScript`, and `javascript:`/`data:` URLs are blocked even if unasked (#160).
- A start no longer sends any request to `huggingface.co` once the Silero voice-activity model is downloaded: vaice
  loads it from the Hugging Face cache instead of letting fastrtc check for a newer revision every time. The first
  start still downloads it; with `HF_HUB_OFFLINE` set (either way) fastrtc's own loader runs as before (#143).
- `open_app` and `quit_app` know a few names that are not words of the app's name: "VS Code" and "VSCode" (Visual
  Studio Code, which runs as "Code"), "System Preferences" (System Settings) and "iTunes" (Music). It is a short fixed
  list, `APP_ALIASES` in `skills.py`; initials are not matched (#140).
- While a browser step waits for **Run**, a request from an ordinary reply that also needs Run (`quit_app`,
  `run_shell`, `forget_everything`, an MCP `click` tool) no longer replaces the step, which answered it no and stopped
  the browser task although you never declined. The new request is refused, and the model tells you a step is
  waiting (#120).
- Clearing fastrtc's playback queue (a barge-in, talking over a Run outcome or a browser task's outcome, *New
  conversation*) now runs on the connection's event loop, which fills and plays that queue, instead of on the
  thread that asked for it; the clear raced the loop's own reads and writes of the queue and of its playback
  clock (#67).
- A browser step's spoken heads-up ("... Press Run to allow it, or Cancel.") is no longer added to the turn that
  asked for the task, so it is not in that chat bubble, in `conversation.json` or in the history of later replies;
  the confirmation panel still shows the step (#127).
- Gradio's telemetry and its "newer version" check (three requests to `huggingface.co` and `api.gradio.app` at every
  start) are off: `vaice` sets `GRADIO_ANALYTICS_ENABLED=False` unless you set it yourself (#98).
- `open_app` said an app was not installed when it was asked for by its short name ("Chrome" for Google
  Chrome): it now opens the one installed app whose name has those words, or names the apps it could be; `quit_app`
  matches running apps the same way, and Run quits only the app its request named (#130).
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
