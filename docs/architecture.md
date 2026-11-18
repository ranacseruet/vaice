# Architecture

How `vaice` is put together today. For where it is going, see the [vision](vision.md); for the reasons behind
individual decisions and the bugs that shaped them, see [AGENTS.md](../AGENTS.md).

## The shape of it

`vaice` is one Python process that sits between a browser and a few model servers. It owns the conversation, the
skills and every safety decision; the models only produce speech and text.

```
 Browser                         vaice process (Python 3.12)                        Model servers
┌──────────┐  WebRTC audio  ┌────────────────────────────────────────┐        ┌──────────────────────────┐
│ mic      │───────────────▶│ fastrtc: VAD finds the end of your     │        │ llama-server  (required) │
│ speaker  │◀───────────────│ utterance, plays the reply, barge-in   │        │  /v1/realtime  speech    │
│ Gradio UI│◀──── HTTP ────▶│ Responder: turns audio into a reply    │──────▶ │  /v1/chat/completions    │
└──────────┘                │ Conversation: history, saved to disk   │        │  /tts                    │
                            │ Skills + confirmation + memory         │        └──────────────────────────┘
                            └───────────┬────────────────────────────┘        ┌──────────────────────────┐
                                        │  optional, each its own endpoint    │ helper brain  (optional) │
                                        ├────────────────────────────────────▶│ vision model  (optional) │
                                        │                                     └──────────────────────────┘
                                        ▼  stdio
                                 MCP servers (optional, started by vaice)
```

Principles that decide most design questions (the long form is in the [vision](vision.md)):

- **The models propose; vaice decides.** Skills, confirmation, timeouts and cancellation are vaice's code, not
  something a 2B model is trusted with.
- **Text boundaries between roles.** Ears, brain, hands and mouth exchange text, so each can be replaced (a bigger
  brain, another voice) without touching the others.
- **Opt-in for anything that widens reach.** Shell, memory, screen vision, a remote model and MCP tools are off
  until you turn them on, and each says what it can see or do.
- **A reply never takes the app down.** Errors are caught inside the reply, spoken or shown, and the connection
  carries on.

## The front end (both modes)

| Piece | Job |
|---|---|
| `app.py` | The CLI (`vaice`, `vaice up`) and the Gradio page: the microphone, the transcript, the *Tool activity*, confirmation and *Memory* panels, the **New conversation** button. Wraps a `Responder` and logs time to first audio |
| `interruptible.py` | `InterruptSafeReplyOnPause`, a fastrtc `ReplyOnPause` subclass. Tunes the VAD, keeps 0.3 s of pre-roll so opening words are not clipped, and fixes the races that made barge-in drop the interrupting question |
| `audio.py` | Browser frames to mono PCM16 at the model's rate, and replies back to float32 |
| `responders/base.py` | The `Responder` contract: `respond(audio, history, turn)` yields `(sample_rate, float32 samples)` chunks and reports the transcript and the spoken text on the `turn` |
| `conversation.py` | `Conversation` and `Turn`: the windowed history that goes into each prompt, barge-in trimming, atomic JSON persistence |
| `server.py` | `vaice up`: reuse a running `llama-server` or start one, wait for `/health`, stop it at exit |
| `lifecycle.py` | The one stop signal: Ctrl+C asks the app and every in-flight call to give up |

The user speaks, fastrtc's VAD decides the utterance ended, and the `Responder` produces a reply as a stream of
audio chunks. If the user speaks again while a reply plays, fastrtc interrupts it ("barge-in"); the `Turn` then
keeps only the part of the reply that could have been heard.

## Chat mode: one model does it all

`LlamaCppResponder` (`responders/llamacpp.py`) opens a fresh `/v1/realtime` WebSocket session per reply. It sends the
whole utterance as audio and receives the spoken reply as streamed frames with the model's own text alongside. History
travels as ChatML turns inside the session `instructions`. A second `transcription` session reports what the user
said. Everything about the model's speech and wording is the model's; vaice only streams it and records it. See
[Chat mode](chat-mode.md).

## Assistant mode: ears, brain, hands, mouth

`AssistantResponder` (`responders/assistant.py`) splits the work into steps so that vaice controls what is said and
done:

```
audio ─▶ transcribe ─▶ decide(brain ⇄ skills) ─▶ speakable()+sentences() ─▶ /tts per sentence ─▶ audio frames
         (realtime     │  ▲                                                                    ▲
          transcription│  └─ result of each skill goes back to the brain                       │
          session)     └─ a confirm skill stops here and asks ──▶ question spoken, request armed
```

1. **Transcribe.** The realtime `transcription` session turns the audio into text (about 0.15 s).
2. **Decide.** `brain.ChatBrain` makes a `/v1/chat/completions` call with the transcript, the history and the skills as
   tools. If the brain asks for skills, vaice runs them and calls the brain again, at most `MAX_SKILL_ROUNDS` (3) times;
   the last round has no tools, so it must answer. The brain is the local `llama-server` at temperature 0, or any
   OpenAI-compatible endpoint you configure ([helper model](helper-model.md)).
3. **Speak.** The answer is cleaned for speech (`voice.speakable`: no markdown, no paths), split into sentences, and
   each sentence is streamed from `/tts`. Text is recorded sentence by sentence as its audio is queued, so a barge-in
   keeps what was actually spoken.

Details that matter:

- **Skills** (`skills.py`) have a `kind`: `read` and `action` run at once, `confirm` asks first. A skill runs an argv list
  with a timeout and no shell, validates its arguments, and never raises; a failure becomes a spoken sentence. Results
  that must not be paraphrased (a refusal, a missing permission) carry a fixed `answer` that ends the reply.
- **History carries tool results.** Each answered turn's skill results are replayed as a real tool exchange, so a
  follow-up ("and how much memory?") is answered from the result, not from the model's head.
- **Spoken progress.** If the decision takes longer than 2.5 s, a second thread says "One moment." and then "Still
  working." through the same `/tts` path, never recording the filler in history.
- **One worker thread per reply.** The generator that fastrtc drives only polls a queue; the slow work happens on a daemon
  thread. See [Threading and cancellation](#threading-and-cancellation).

### Confirmation

A `confirm` skill does not run when the brain asks for it. `decide()` returns a `Question` (a fixed sentence that
names the skill but never reads the arguments), and `confirm.py` holds one pending request: it is shown in the UI at
once and **armed only after the question's audio was actually handed out**, so an interrupted question cannot be
answered by accident. It expires 60 s after arming. The next utterance approves only when the *whole* transcript is a yes
phrase, or the **Run** button is clicked; anything else drops the request and is handled as a new question. A pending
request is taken atomically, so a click and a spoken yes cannot both run it. A request that only a click may approve (shell
commands outside a tiny allowlist, `click`-policy MCP tools) never accepts a spoken yes at all. Question texts contain
none of the words a user answers with, because speaker echo can reach the microphone.

An `auto`-policy MCP tool adds a timer: the announcement (`Question.auto_seconds`) is armed like any question, and a
daemon thread (`vaice-assistant-auto`) then runs the request `VAICE_ASSISTANT_AUTO_SECONDS` after the announcement's estimated end of playback (audio handed out minus time since the first frame; fastrtc reports no playback position) through
`AssistantResponder.approve`, the same atomic `take` a click uses, so a Cancel click, a spoken stop or yes, "New
conversation" or an expiry that comes first means it runs once or not at all. If another skill ran earlier in the
same reply, `decide()` swaps the question for the tool's click-only `fallback` (that skill's output may be untrusted
text); content read in an earlier turn is not tracked.

### The skills and their extensions

| Skill group | Module | Notes |
|---|---|---|
| Read-only checks (disk, battery, system, processes, find files) | `skills.py` | Run at once; output spoken as clean names |
| Actions (open app, volume, mute) and `quit_app` (confirm) | `skills.py` | Results name their object, because history replays calls without their arguments |
| `run_shell` (opt-in) | `shell.py` | A small allowlist of harmless commands may be approved by voice or click; everything else is click-only, through `zsh -c`, with a 30 s timeout, an output cap and a process group that barge-in kills |
| `remember`, `forget`, `forget_everything` (opt-in) | `memory.py` | Explicit notes only, saved to `memory.json` and appended to the system prompt. A note is text that outlives its conversation, so it is cleaned, capped, and refused if a tool ran earlier in the same reply |
| `look_at_screen` (opt-in) | `vision.py` | One `screencapture` + `sips` shrink, asked of a vision model, never kept or logged. See [Screen vision](screen-vision.md) |
| MCP tools (opt-in) | `mcp_skills.py` | Each listed tool of each stdio server becomes `<server>__<tool>` with a `run`, `voice`, `auto` or `click` policy. See [MCP tools](mcp-tools.md) |

## Safety model

Layers, from the outside in:

1. **Off by default.** Anything that reaches further than the audio loop is an opt-in setting.
2. **No shell unless approved.** Skills are argv lists. The one shell path is behind confirmation, shows the exact
   command, and is click-only unless the command is provably harmless.
3. **Confirmation is vaice's, not the model's.** The approval gate, its timing and its words are code; a model cannot
   approve its own request, and a replayed or stale request cannot run.
4. **Untrusted text stays in the `tool` role.** What a file name, a command, a web page on screen or an MCP server returns
   goes to the brain only as tool output. `remember` refuses once any skill has run in the reply, so tool output cannot be
   turned into a persistent note.
5. **Everything is cancellable.** Barge-in, **New conversation** and Ctrl+C stop skills mid-run and end child processes.
6. **Secrets are opaque.** API keys and MCP `env` values live in configuration and in one header; no log line, error or
   result holds them.

Honest limits: the confirmation words are matched on a transcript, a 2B model still misjudges when to call a skill, and
speaker echo is a known risk (use headphones). See [Privacy](privacy.md) and the "Not verified" notes in AGENTS.md.

## Threading and cancellation

fastrtc resumes a reply generator on a different worker thread for each chunk, and blocking network calls must never hold
one. So each reply's slow work (`connect()`, the realtime sessions, brain calls, `/tts`, skills) runs on **daemon
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
| `memory.json` | The notes (assistant mode, memory on) |
| `llama-server.log` | Output of a `llama-server` that `vaice up` started |
| `mcp-<name>.log` | Stderr of an MCP server vaice started |

## Testing and measuring

Unit tests use fakes for the server and the model; a fake `/v1/realtime` server (`tests/fake_realtime.py`) exercises
the responder end to end, and `tests/threads.py` reproduces fastrtc's real threading. Nothing automated hears a real
model or the browser microphone. Claims about model behavior come from the **live eval** in `evals/`, run by hand with
repeats per build and quantization, on the chat and assistant paths; only its scoring is unit-tested. See
[Development](development.md).

## Code map

```
src/vaice/
  app.py            CLI + Gradio/WebRTC UI; wraps a Responder, logs time to first audio, transcript + reset
  interruptible.py  ReplyOnPause subclass fixing fastrtc 0.0.34 barge-in races, VAD tuning, pre-roll buffer, reset silencing, and idle CPU polling
  server.py         `vaice up`: reuse a running llama-server or start one (log in the data dir), wait for /health, stop it at exit
  lifecycle.py      Stop signal handler: Ctrl+C asks the app and its responders to abandon in-flight calls
  config.py         Environment-driven settings (.env.example lists them with defaults)
  conversation.py   Turn / Conversation: history windowing, barge-in trimming, atomic JSON persistence, error notes
  brain.py          Assistant mode: ChatBrain, one chat-completions call with tools (local llama-server or a helper endpoint), BrainError
  skills.py         Assistant mode: the skills (read-only answers, plus open/quit app, volume, mute) and the Skills registry
  vision.py         Assistant mode, opt-in: the look_at_screen skill (one screenshot via screencapture + sips, asked of a vision model)
  memory.py         Assistant mode: long-term notes (Memory, kept in memory.json) and the remember / forget / forget_everything skills
  mcp_skills.py     Assistant mode, opt-in (the `mcp` extra): MCP servers over stdio as skills, with a per-tool approval policy
  confirm.py        Assistant mode: the one pending request of a confirm skill, the spoken yes/no test and its 60 s expiry
  shell.py          Assistant mode, opt-in: the run_shell skill: how a command may be approved, and the runner (timeout, output cap, process group)
  voice.py          Assistant mode: speakable() cleans text for speech, sentences() splits it for /tts
  audio.py          Mic frame → mono PCM16 at the model's rate (resampled via scipy) and back to float32
  responders/
    base.py         Responder protocol (audio + history in → audio chunks out; reports transcript and spoken text)
    __init__.py     Responder package
    llamacpp.py     llama-server /v1/realtime: parallel reply and transcription sessions on daemon threads, startup check
    assistant.py    Assistant mode: transcribe, brain + skills loop, streamed /tts per sentence, on one daemon thread
tests/
  fake_realtime.py  Fake /v1/realtime WebSocket + HTTP server for testing LlamaCppResponder end to end
evals/
  run.py            Live tool-calling eval against a real llama-server (run by hand; see evals/README.md)
  scoring.py        Pure scoring and aggregation for the eval (unit-tested)
  cases.json        Spoken commands and expected tool calls; tools.json holds the stub tool set
  assistant_cases.json  Spoken questions and expected skills (memory ones carry `notes`) for the assistant path (`--paths assistant`)
  assistant_vision_cases.json  The same for `look_at_screen` (run with `--vision`; the screen is a stub)
  recordings/       Optional real-mic clips, <case id>.wav
```

## Where this is heading

The architecture was built for a roadmap, not the other way round: separate roles behind text boundaries, skills in
vaice's hands, a swappable brain. The [vision](vision.md) says what that is for and what is still missing.
