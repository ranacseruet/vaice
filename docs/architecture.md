# Architecture

How `vaice` is put together today. For where it is going, see the [vision](vision.md); for the reasons behind
individual decisions and the bugs that shaped them, see [AGENTS.md](../AGENTS.md).

## The shape of it

`vaice` is one Python process that sits between a browser and a few model servers. It owns the conversation, the
skills and every safety decision; the model only listens and speaks, and asks for a skill when it needs one.

```
 Browser                         vaice process (Python 3.12)                        Model servers
┌──────────┐  WebRTC audio  ┌────────────────────────────────────────┐        ┌──────────────────────────┐
│ mic      │───────────────▶│ fastrtc: VAD finds the end of your     │        │ llama-server  (required) │
│ speaker  │◀───────────────│ utterance, plays the reply, barge-in   │        │  /v1/realtime  speech    │
│ Gradio UI│◀──── HTTP ────▶│ Assistant: one session per             │──────▶ │  and transcription       │
└──────────┘                │  reply, skills as its tools            │        └──────────────────────────┘
                            │ Conversation: history, saved to disk   │        ┌──────────────────────────┐
                            │ Skills + confirmation + memory         │        │ vision model  (optional) │
                            └───────────┬────────────────────────────┘        │ browser agent (optional) │
                                        │  optional, each its own endpoint    └──────────────────────────┘
                                        ├────────────────────────────────────▶
                                        ▼  stdio
                                 MCP servers (optional, started by vaice)
```

Principles that decide most design questions (the long form is in the [vision](vision.md)):

- **The model proposes; vaice decides.** Skills, confirmation, timeouts and cancellation are vaice's code, not
  something a 2B model is trusted with.
- **One path.** Every reply is one speech-to-speech session. There is no second text model and no separate voice step to
  keep in sync, and everything that is not a reply (the outcome of a Run click, a browser task) is spoken by a session of
  the same kind.
- **Opt-in for anything that widens reach.** Shell, memory, screen vision, a remote model and MCP tools are off
  until you turn them on, and each says what it can see or do.
- **A reply never takes the app down.** Errors are caught inside the reply, spoken or shown, and the connection
  carries on.

## The front end

| Piece | Job |
|---|---|
| `app.py` | The CLI (`vaice`, `vaice up`) and the Gradio page: the microphone, the transcript, the *Tool activity*, confirmation and *Memory* panels, the **New conversation** button. Wraps the assistant and logs time to first audio |
| `interruptible.py` | `InterruptSafeReplyOnPause`, a fastrtc `ReplyOnPause` subclass. Tunes the VAD, keeps 0.3 s of pre-roll so opening words are not clipped, fixes the races that made barge-in drop the interrupting question, and serves the outbox when no reply is playing |
| `outbox.py` | `Outbox`: audio frames and callback markers that a worker thread queues for speech no reply carries (the outcome of a Run click, a browser task); the handler's idle `emit()` serves it |
| `audio.py` | Browser frames to mono PCM16 at the model's rate, and replies back to float32; the `AudioFrame` type, `(sample_rate, samples)` |
| `conversation.py` | `Conversation` and `Turn`: the windowed history that goes into each prompt, barge-in trimming, atomic JSON persistence |
| `server.py` | `vaice up`: reuse a running `llama-server` or start one, wait for `/health`, stop it at exit |
| `lifecycle.py` | The one stop signal: Ctrl+C asks the app and every in-flight call to give up |

The user speaks, fastrtc's VAD decides the utterance ended, and the assistant (`Assistant.respond(audio, history, turn)`) produces a reply as a stream of
`(sample_rate, float32 samples)` chunks, reporting the transcript and the spoken text on the `turn`. If the user speaks again while a reply plays, fastrtc interrupts it ("barge-in"); the `Turn` then keeps only the
part of the reply that could have been heard.

## A reply: one session, skills as its tools

`RealtimeResponder` (`realtime.py`) opens a fresh `/v1/realtime` WebSocket session per reply. It sends the
whole utterance as audio and receives the spoken reply as streamed frames with the model's own text alongside. History
travels as ChatML turns inside the session `instructions`. A second `transcription` session reports what the user said.
`Assistant` (`assistant.py`) is that session with vaice's skills attached:

```
audio ─▶ realtime session ─▶ speech frames ─▶ fastrtc
          │  ▲   instructions = prompt + history + clock + memory notes
          │  └── result of each skill goes back as a function_call_output
          └─ function_call ─▶ executor ─▶ read/action: run now
                                          confirm: a pending request, nothing runs (button only)
                                          browse: start the browser task in the background
```

1. **The skills are the session's tools,** declared in the flat Responses format. The first response is requested with
   `tool_choice: "required"`, later ones with `auto`, and after `MAX_TOOL_ROUNDS` (2) with `none`, which forces an answer.
2. **A per-reply executor** (`_Executor`) handles each call on the session thread. A `read` or `action` skill runs at
   once through `Skills.run`, which never raises: a failure is the result the model reads. A `confirm` skill goes through
   `Skills.prepare`, which checks the arguments and builds the question; the executor creates the pending request and the
   result tells the model that only the Run button runs it. A `background` skill (`browse`) starts the task. `remember` is
   refused once any other skill was called in the same reply, so a file name or a command's output cannot become a note.
3. **Results are compacted** (`_speech_friendly`) before they go back, because the model reads a fed-back result aloud.
   Only the answer after the last tool round is recorded as the reply's text; the words before a result are spoken but
   dropped from the record.
4. **Slow work on daemon threads.** The generator that fastrtc drives only polls a queue; the reply session and the
   transcription session each run on a daemon thread of their own. See [Threading and cancellation](#threading-and-cancellation).

### Confirmation

A `confirm` skill (`quit_app`, `run_shell`, `forget_everything`, an MCP tool with the `click` policy) does not run when
the model asks for it. `confirm.py` holds one pending request, shown in the UI at once with the skill and its exact
arguments; **only the Run button settles it**, and nothing a user says does. It expires 60 s after it was made; a
newer request replaces it, and a stale panel's buttons carry the old id and settle nothing. A pending request is taken
atomically, so two clicks cannot both run it. A request, a browser task or a step from a reply that began before *New conversation*
is refused (`Confirmations.epoch`), and outcomes and heads-ups are spoken one at a time, never interleaved.

**Run** executes the stored call (never what the model says afterwards), records the result on the asking turn and then
has a fresh realtime session speak the outcome: a session with no tools whose input is a text message
(`The approved action is finished. Result: ... In one short sentence, tell the user what happened.`), so the model phrases it in its own voice. Its frames go to the
`Outbox` (`app.build_stream` creates it, passes it to the handler and calls the assistant's `attach_outbox`), followed by
one marker that appends the spoken text to the asking turn (`Turn.append_assistant_text`, the one write a closed turn
accepts), so a barge-in that clears the outbox leaves the text unrecorded too. The handler's idle `emit()` (no pause, no
reply generator, the user not talking) serves the outbox **paced**: fastrtc's playback queue is unbounded, so the next
item is served only while the audio served so far (a reply's included) is estimated to end less than `LEAD_SECONDS`
(0.4 s) ahead, which keeps the markers close to real playback. Starting to talk with no reply running clears the outbox
and, when its audio is estimated still playing, fastrtc's playback queue; with `VAICE_CAN_INTERRUPT=0` it does neither.
*New conversation* drops the request and clears the outbox; speech queued while no connection polled for 30 s is dropped.
**Cancel** records `cancelled` and speaks nothing.

### The skills and their extensions

| Skill group | Module | Notes |
|---|---|---|
| Read-only checks (disk, battery, system, processes, find files) | `skills.py` | Run at once; output spoken as clean names |
| Actions (open app, volume, mute) and `quit_app` (confirm) | `skills.py` | Results name their object (`opened Safari`), so the model's answer and the *Tool activity* record agree |
| `run_shell` (opt-in) | `shell.py` | Always a confirm skill: through `zsh -c` after Run, with a 30 s timeout, an output cap and a process group that stopping kills; `sudo` is refused |
| `remember`, `forget`, `forget_everything` (opt-in) | `memory.py` | Explicit notes only, saved to `memory.json` and appended to the instructions. A note is text that outlives its conversation, so it is cleaned, capped, and refused if a tool ran earlier in the same reply |
| `look_at_screen` (opt-in) | `vision.py` | One `screencapture` + `sips` shrink, asked of a vision model, never kept or logged. See [Screen vision](screen-vision.md) |
| `browse` (opt-in) | `browse_task.py`, `browse.py` | A `background` skill: the reply says one line and ends, then a task thread runs the Pi agent quietly. A step that changes a page is a button-only pending request with a spoken heads-up; the outcome is spoken by a text-input session through the outbox and recorded on the asking turn. A new utterance does not stop the task; the Stop button, `stop_browsing`, *New conversation* and exit do, and `browser_status` reports on it. One task at a time. See [Browser tasks](browser-tasks.md) |
| MCP tools (opt-in) | `mcp_skills.py` | Each listed tool of each stdio server becomes `<server>__<tool>` with a `run` or `click` policy. See [MCP tools](mcp-tools.md) |

## Safety model

Layers, from the outside in:

1. **Off by default.** Anything that reaches further than the audio loop is an opt-in setting.
2. **No shell unless approved.** Skills are argv lists. The one shell path is behind confirmation: it shows the exact
   command and runs only after you press Run.
3. **Confirmation is vaice's, not the model's.** The approval gate and its words are code, and the only approval is a
   button press; a model cannot approve its own request, and nothing said aloud can, so speaker echo cannot either. A
   replayed or stale request cannot run.
4. **Untrusted text stays in the tool result.** What a file name, a command, a web page on screen or an MCP server returns
   goes to the model only as a skill's result. `remember` refuses once any skill has run in the reply, so tool output cannot be
   turned into a persistent note.
5. **Everything is cancellable.** Barge-in, **New conversation** and Ctrl+C stop skills mid-run and end child processes.
6. **Secrets are opaque.** API keys and MCP `env` values live in configuration and in one header; no log line, error or
   result holds them.

Honest limits: a 2B model still misjudges when to call a skill (and can say it did something it did not), it sometimes
reads a skill's raw result aloud, and speaker echo can still cut a reply short (use headphones). See [Privacy](privacy.md)
and the "Not verified" notes on each feature's page.

## Threading and cancellation

fastrtc resumes a reply generator on a different worker thread for each chunk, and blocking network calls must never hold
one. So each reply's slow work (`connect()`, the realtime sessions, skills) runs on **daemon
threads of its own**, and the generator only polls a queue every `POLL_SECONDS` while checking `lifecycle.stopping` and
whether the turn was closed. When the generator is dropped (barge-in), its `finally` sets a cancel event and the worker
closes its sockets and stops its skills. Because the threads are daemons and the generator never blocks, a server that
never answers cannot keep the process alive or make Ctrl+C hang. MCP servers run on one asyncio loop on their own thread
(`vaice-mcp`), and each call polls its `stop()` like the shell runner does.

## State on disk

All under `VAICE_DATA_DIR` (default `~/.vaice`, files `0600`; empty means nothing is written):

| File | Holds |
|---|---|
| `conversation.json` | The conversation: turns, tool events, written atomically after every update |
| `memory.json` | The notes (memory on) |
| `llama-server.log` | Output of a `llama-server` that `vaice up` started |
| `mcp-<name>.log` | Stderr of an MCP server vaice started |

## Testing and measuring

Unit tests use fakes for the server and the model; a fake `/v1/realtime` server (`tests/fake_realtime.py`) exercises
the assistant end to end, and `tests/threads.py` reproduces fastrtc's real threading. Nothing automated hears a real
model or the browser microphone. Claims about model behavior come from the **live eval** in `evals/`, run by hand with
repeats per build and quantization; only its scoring is unit-tested. See
[Development](development.md).

## Code map

```
src/vaice/
  app.py            CLI + Gradio/WebRTC UI; wraps the assistant, logs time to first audio, transcript + reset, the panels
  interruptible.py  ReplyOnPause subclass fixing fastrtc 0.0.34 barge-in races, VAD tuning, pre-roll buffer, reset silencing, idle CPU polling, and serving the outbox
  outbox.py         Outbox: audio + markers for speech with no reply in flight (a Run click's or a browser task's outcome), drained by the idle emit()
  server.py         `vaice up`: reuse a running llama-server or start one (log in the data dir), wait for /health, stop it at exit
  lifecycle.py      Stop signal handler: Ctrl+C asks the app and the assistant to abandon in-flight calls
  config.py         Environment-driven settings (.env.example lists them with defaults) and the removed/renamed names it warns about
  conversation.py   Turn / Conversation: history windowing, barge-in trimming, atomic JSON persistence, error notes
  skills.py         The skills (read-only answers, plus open/quit app, volume, mute), the Skills registry and the flat tool definitions
  confirm.py        The one pending request of a confirm skill (or a browser step), button-only, with its 60 s expiry
  vision.py         Opt-in: the look_at_screen skill (one screenshot via screencapture + sips, asked of a vision model)
  memory.py         Opt-in: long-term notes (Memory, kept in memory.json) and the remember / forget / forget_everything skills
  mcp_skills.py     Opt-in (the `mcp` extra): MCP servers over stdio as skills, with a per-tool run/click policy
  shell.py          Opt-in: the run_shell skill and the runner (timeout, output cap, process group)
  browse.py         Opt-in: PiAgent drives the Pi coding agent (`pi --mode rpc`) with a goal, through chrome-devtools-mcp; its config directory, child process, abort/kill handling and the cleanup of Chrome processes on its profile
  browse_task.py    Opt-in: the `browse`, `browser_status` and `stop_browsing` skill definitions and the pure pieces (fixed lines, step questions, result text, the status record)
  browse_gate.ts    The Pi extension browse.py loads: the approval policy for the browser's tools (read-only run, page-changing ask, the rest ask); shipped in the package
  audio.py          Mic frame → mono PCM16 at the model's rate (resampled via scipy) and back to float32; the `AudioFrame` type
  realtime.py       llama-server /v1/realtime: `RealtimeResponder` (the reply session: tool rounds, repetition breaker), the parallel transcription session, text-input speech (`speak`), `spoken_part`, startup check
  assistant.py      `Assistant`: a `RealtimeResponder` with the skills as tools, the per-reply executor, Run/Cancel, and the browser task thread
tests/
  fake_realtime.py  Fake /v1/realtime WebSocket + HTTP server for testing the assistant end to end
  threads.py        fastrtc's real threading (a fresh thread per resume) for generator tests
  fixtures/fake_pi.py  A fake `pi --mode rpc` (scripted scenarios, records the commands it gets) for tests/test_browse.py
  fixtures/mcp_server.py  A small stdio MCP server the MCP tests start for real (tests/test_mcp_skills.py)
evals/
  run.py            Live tool-calling eval against a real llama-server (run by hand; see evals/README.md)
  scoring.py        Pure scoring and aggregation for the eval (unit-tested)
  cases.json        Spoken questions and expected skills (memory ones carry `notes`)
  shell_cases.json, vision_cases.json, browse_cases.json  The same for the opt-in skills (`--shell`, `--vision`, `--browse`)
  recordings/       Optional real-mic clips, <case id>.wav
```

## Where this is heading

The architecture was built for a roadmap, not the other way round: skills and every safety decision in vaice's hands, one
replaceable model. The [vision](vision.md) says what that is for and what is still missing.
