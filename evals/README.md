# Skill-calling evals

A live harness that measures how reliably spoken commands turn into the right skill call, with repeats, against a real `llama-server`. It is a developer tool: the app does not import it and CI does not run it (only `scoring.py` and the harness's pure parts are unit-tested, in `tests/test_eval_scoring.py` and `tests/test_eval_run.py`). Use it before making a claim about how the model behaves, and after changing the prompt, a skill's description or the server settings.

## What it measures

Each case is a spoken sentence and an expectation (see [Cases](#cases)). Every case runs `--repeats` times on the app's own path: `Assistant` sends the audio over `/v1/realtime` with the real skill definitions declared as the session's tools, the model answers in text and audio, and its calls go through the real executor. The skills are stubs with canned results (`disk_space` says `160 gigabytes free of 460 gigabytes`, and so on; nothing is executed on this machine). A skill that needs the Run button (`quit_app`, `run_shell`, `forget_everything`) creates its request as in the app and answers the model "not done yet, press Run": nothing runs, and the call counts as the call. `browse` starts no browser task (the harness has no agent); it answers `Working in the browser.` and the call counts. The calls are read back from the turn's tool events, and the spoken answer is the text the model's audio carried.

Recorded per run: the calls, the transcript (from the parallel transcription session), the answer text, the time to first audio and the seconds of audio. The summary prints the median and max time to first audio, the median audio length and how many replies had under 0.5 s of audio (a spoken-nothing reply).

A run is `correct`, or one of:

| verdict | meaning |
|---|---|
| `missed` | a call was expected, but none was made, or the set of tools was wrong (extra or missing tools are named in the `reason`) |
| `overcall` | no call was expected, but the model called a tool |
| `wrong_args` | the right tools, but an argument did not match (or the arguments were not a JSON object) |
| `error` | the run itself failed (server error, timeout); the error text is recorded and the eval goes on |

`duplicate` is reported on top of the verdict: the same tool with the same arguments called more than once in one run.

## Prerequisites

- A running `llama-server` with the audio model and the realtime endpoint, as in the main README (default `http://localhost:8001`, or `--url`). The eval reads `/health` and `/props` first and stores the build, model alias and temperature in the results. Start the server with `--temp 0` (the model card's recommendation), or the results mix in sampling noise.
- The audio for each case: a recording in `evals/recordings/<case id>.wav`, or macOS `say`, which synthesizes the sentence once per case (16 kHz). On a platform without `say`, every case needs a recording.
- The dev environment: `uv sync --extra dev`.

## Run it

```bash
.venv/bin/python evals/run.py                                   # all cases, 5 repeats
.venv/bin/python evals/run.py --repeats 1 --cases open-safari,know-sky   # a quick check
.venv/bin/python evals/run.py --label Q8_0                      # label stored in the results
.venv/bin/python evals/run.py --memory --repeats 1              # with the memory skills (and the memory-* cases)
.venv/bin/python evals/run.py --shell --repeats 1 --cases-file shell_cases.json     # with run_shell
.venv/bin/python evals/run.py --vision --repeats 1 --cases-file vision_cases.json   # with look_at_screen
.venv/bin/python evals/run.py --browse --repeats 1 --cases-file browse_cases.json   # with browse
```

Options:

- `--url`: the llama-server (default `VAICE_LLAMACPP_URL`).
- `--repeats N` (default 5), `--cases id,id`, `--cases-file FILE` (a file in `evals/` or a path; default `cases.json`).
- `--memory`: also offer `remember`, `forget` and `forget_everything`, which work for real on a **fresh in-memory store for every run** (`Memory(None)`, refused if it has a path), so a run never reads or writes your `~/.vaice/memory.json` and one run's notes never reach the next. A case's `"notes": [...]` are loaded into that store and put into the prompt as the app does (`NOTES_HEADER`). Off by default, as in the app (`VAICE_MEMORY`): without it the `memory-*` cases are expected to fail, so pass `--cases` or another file.
- `--shell`: also offer the stubbed `run_shell` (it asks and nothing runs: the stub has no runner).
- `--vision`: also offer the stubbed `look_at_screen` (a fixed fake screen; nothing is captured or sent).
- `--browse`: also offer `browse`, `browser_status` and `stop_browsing`. `browse` starts a task on a fake agent that runs until it is stopped, as in the app, so the reply ends at the call; the run reads the calls, then stops the task and waits for its thread (the task's own outcome is not among the calls). Runs that ended in a started task have no reply audio by design: they are left out of `silent`, the audio median and the latency and first-audio medians, and the summary counts them apart (`browser starts: N`; `browse_starts` in the results). Older results (where `browse` answered `Working in the browser.` and the model talked on) are not comparable on those numbers.
- `--prompt TEXT`: the reply instructions (default `VAICE_PROMPT`).
- `--history/--no-history` (default on: two fixed earlier exchanges, `Hey, are you there?` / `Yes, I'm here.` and `What day is it?` / `It's Monday.`).
- `--out FILE` (default `evals/results/<UTC timestamp>.json`, which is gitignored) and `--label TEXT`.

Ctrl+C stops the run and reports the runs so far. The full `cases.json` is 69 cases, around 350 runs at 5 repeats; use `--cases` and `--repeats 1` while iterating.

## Read the report

The summary prints the count and percentage of each verdict, `duplicate`, and the timing lines. Then a per-case table of `k/N correct`: a case at `3/5` is flaky, `0/5` is a plain failure. Percentages are of all runs, errors included.

Runs whose transcript differs from the spoken sentence (case and punctuation ignored) are listed, so a wrong call caused by a mishearing is told apart from a model failure. The comparison is literal: the transcript `50` against the sentence `fifty` counts as a difference.

The results file holds the config, server info, label, and every run's raw record (calls, texts, timings, error) with its score, so a surprising number can be traced to the run that produced it.

## Live browser-agent check

`run.py --browse` scores only whether the voice model decides to call `browse`; it never starts a browser task. `browse_live.py` is the agent's side: it opens the browser agent exactly as the app does (`VAICE_BROWSE`, `VAICE_BROWSE_URL`, `_MODEL`, `_API_KEY`, `VAICE_BROWSE_AGENT`, `VAICE_BROWSE_CONFIRM`, `VAICE_DATA_DIR`, from the environment and a `.env` in the working directory), gives it research goals with no named site, and records what it did. It does not involve the voice model, the audio path or `llama-server`.

```bash
.venv/bin/python evals/browse_live.py                                   # the three built-in goals, once each
.venv/bin/python evals/browse_live.py --repeats 3 --label qwen3-32b     # k/N over repeats
.venv/bin/python evals/browse_live.py --goal "Find the latest stable Python release." --visible
```

Options: `--goal TEXT` (repeatable; replaces the built-in goals), `--repeats N` (default 1), `--seconds S` (per task; default `VAICE_BROWSE_SECONDS`), `--visible` (show Chrome's window), `--out FILE` (default `evals/results/browse-live-<UTC timestamp>.json`, gitignored) and `--label TEXT`.

- It contacts real sites and the configured browse model, so it needs `VAICE_BROWSE` set (and `VAICE_BROWSE_URL` for Pi, or `VAICE_BROWSE_AGENT` for `acp`), the `browse` extra (`uv sync --extra dev --extra browse`) and Node.js. When the agent cannot be started it exits with the app's start-up warnings above the error. CI never runs it.
- Chrome is headless and always muted (`--mute-audio`, `--audio-muffling`), unless `--visible` (still muted).
- It uses the browser profile under `VAICE_DATA_DIR` (your logins included), and ends whatever holds that profile before the agent starts: while vaice itself is running, point it elsewhere, e.g. `VAICE_DATA_DIR=/tmp/vaice-live .venv/bin/python evals/browse_live.py` (the environment wins over `.env`).
- It approves every step the agent asks about, whatever `VAICE_BROWSE_CONFIRM` says, so do not point it at a profile or goals you would not want clicked through.
- Goals run one after another on the same agent, as in the app. Per run it records the outcome (`answer`, or the exception class: `BrowseError`, `BrowseDeclined`, ...), the answer or the error and its spoken sentence, the seconds, the tool calls in order, the steps it was asked about and the pages visited. After the last run it closes the agent and counts the processes still holding the Chrome profile a second later.
- Exit status is 0 when every run produced an answer and nothing was left holding the profile, 1 otherwise (also when the process table could not be read: the cleanup is then unverified). Ctrl+C stops the running task, closes the agent and reports what finished.

It measures whether the agent finishes a research task and how (tools, pages, time); it does not grade whether the answer is right, read that yourself. Results are per agent, model and build and vary run to run: quote `k/N` answered over repeats with the label, never a single run.

## Browser task, end to end

`run.py --browse` scores whether the voice model calls `browse`, and `browse_live.py` the browser agent; neither shows what the user gets around them. `browse_flow.py` runs a whole browser task as the user experiences it, with the real voice model (llama-server `/v1/realtime`) and a **fake browser agent that replays recorded real answers** (`browse_answers.json`: the six full answers of two live `browse_live.py` runs, Pi 1.5.2 with a local model, each starting with the agent's own narration before its report). So it measures vaice and the voice model, not the browser agent (that is `browse_live.py`). It needs only llama-server: no browser, Pi, network or browse model.

```bash
.venv/bin/python evals/browse_flow.py --cases coffee-real --repeats 1   # a quick check
.venv/bin/python evals/browse_flow.py --label Q8_0                      # all 12 cases, 3 repeats
```

Options: `--url`, `--cases id,id`, `--repeats N` (default 3), `--task-seconds S` (how long the fake task takes, default 8), `--vision` (also offer the stubbed `look_at_screen`, as the app does with vision on), `--outcome report|say` (how the outcome is spoken: `REPORT`, the model's words, the app's default, or `SAY`, as it is; it patches `assistant.OUTCOME` for the run and is recorded in the results), `--prompt TEXT`, `--label TEXT` and `--out FILE` (default `evals/results/browse-flow-<UTC timestamp>.json`, gitignored).

Each run is the app's own wiring: an `Assistant` with a browser agent (so `_start_browse`, the task thread, the real busy check, `browser_status`, `stop_browsing` and the outcome speech are the app's code) over the stub skills of `run.py` (nothing runs on this machine), a `Conversation` whose turns and history are made as `app.make_handler` makes them, and an `Outbox` drained by a thread that does what the idle `emit()` does (paced by the same lead, serving nothing while a reply is in flight; frames are counted, nothing is played). The reply that starts the task ends at the call and vaice says its fixed sentence (the run waits for it, as a listener would, before the follow-up); the outcome is the speech recorded after that one. The sequence: the request (spoken by `say`, or a recording named `<case id>.wav`; the follow-up's recording is `<case id>-followup.wav`), the task, then the follow-up, asked while the fake task still runs (`"when": "running"`: the task is held open until the follow-up's reply is over) or after the outcome was spoken and recorded (`"finished"`). Afterwards the run stops any task and joins its threads.

A case file entry is `{"id", "say", "fixture", "followup": {"say", "when", "expect"}}` (`browse_flow_cases.json`); `expect` is optional: `"answer"` (the default) means the follow-up asks about the task, so it should be answered from what the task found with no new task, and `"browse"` means it is a new request ("Now look up ..."), so a new task is right (the new-request cases use `"finished"`). The loader refuses any other value. a fixture is `{"id", "goal", "answer", "keywords"}`, where `keywords` are two to four names from the report part of the answer (products, missions) that any summary should keep.

Recorded per run, raw data first (so a number can be traced): the start reply's calls, text, time to first audio and audio seconds; the outcome speech's text and the seconds from the task's end to its first audio (frames served after both the task's end and the fixed sentence's marker; a running case's outcome also waits behind the follow-up's answer); the chat as the UI would show it (`ui`) and every turn with its tool events; the follow-up's calls, text and timings; the fake agent's run count and goals; the errors. On top, per run:

| flag | meaning |
|---|---|
| `start_browse` | the request's reply called `browse` |
| `start_invented` | the start text is over 120 characters, or it names a fixture keyword before the task returned (the reply made up findings) |
| `outcome_spoken` | an outcome was recorded on the asking turn |
| `outcome_named` | the outcome names a fixture keyword (case ignored): it says what was found |
| `outcome_narration` | the outcome matches self-narration (`compil`, `let me`, `I now have`, `I have gathered`, `final answer`) |
| `chat_report` | the chat holds a whole keyword-bearing line of the report body: the report itself, not a spoken summary |
| `followup_rebrowse` | the follow-up called `browse` again, or the fake agent ran more than once |
| `followup_status` | the follow-up called `browser_status` |
| `followup_named` | the follow-up's answer names a fixture keyword |
| `followup_right` (`f-right`) | the follow-up did what its case expects: `answer` and `finished`, no new task and a keyword named; `answer` and `running`, no new task and `browser_status` called; `browse`, `browse` called |

plus `start_extra_chars`. The summary prints `k/N` per flag per case and overall (N counts the runs where the flag applies: no follow-up after a failed start, no `outcome_narration` without an outcome), and the medians of the start length and of the first-audio times.

The heuristics are crude: the length threshold and the keyword test only say that a start reply said more than "I'll look into it", not that what it said is wrong; `outcome_named` passes a reply that names a product and still garbles it; the narration pattern can miss a rewording, and fires on a legitimate "let me know"; `chat_report` wants an exact line, so a report shown reformatted fails it. A narrow case (`k/N` of 3) is a lead, not a measurement: read the failing runs' `start`, `outcome` and `followup` texts in the results before quoting a number, with the label and server info. The recorded answers come from one agent and model; they are the test input, not a claim about other agents.

## Cases

A case file is a list of `{"id", "say", "expect"}`, plus an optional `"history"`: a list of `["user" | "assistant", text]` earlier turns that replaces the default history for that case (also under `--no-history`), for a follow-up that only makes sense after an earlier answer (`vision-look-again`: "what do you see now?" after a screen answer). History carries the spoken words only: the app does not replay tool results. `expect` is one of:

```json
{"tools": ["open_app"], "args": {"open_app": {"name": "Safari"}}}
{"tools": ["open_app", "set_volume"]}
{"tools": []}
{"any_of": ["run_shell", "disk_space"]}
```

- `tools`: exactly these tools are called (order and repeated calls do not matter; an extra different tool is a `missed`). `args` is optional and matches per tool as a subset: strings compare case-insensitively after stripping, numbers by value (`20`, `20.0` and `"20"` agree), a list is a set of accepted alternatives (`"name": ["Notes", "Notes app"]`), and `{"contains": "text"}` accepts a string that holds `text`, case ignored (free text such as a note the model words itself). If a tool is called more than once, one matching call is enough.
- `{"tools": []}`: no tool may be called (chit-chat, general knowledge).
- `any_of`: at least one call, and every call is to a tool in the list. Arguments are not checked.

Two optional expectations say something about the answer or the command, apart from the call verdict (a call can be right while the answer is not). They sit in `expect` next to `tools`:

```json
{"tools": ["disk_space"], "answer_has": ["160"]}
{"tools": [], "answer_has": ["Nadia"], "notes": ["The user's sister is called Nadia."]}
{"tools": ["run_shell"], "command_matches": "Downloads", "command_not": "/Users/me"}
```

- `answer_has`: every listed string must appear in the spoken answer's text, case ignored (with `tools: []` it checks a recall from the case's `notes`). Not applicable (`-`) when the run ended in a request for the Run button or a browser task: the model's words about that are not the answer.
- `command_matches` / `command_not`: regular expressions searched, case-sensitive, in the `command` argument of the `run_shell` call (every `run_shell` call must satisfy both; no such call fails the check). The `run_shell` stub only asks, so the command is the model's proposal, never run.

The summary prints `answer faithful k/N` and `command ok k/N` when a case asked for them (over completed runs), and the per-case table gets an `answer` and a `command` column. They are exact-string checks of a stub fact: the answer `one hundred sixty` fails `160`, and `git -C ~/projects status` fails `git status`, so read a failing run before blaming the model.

`cases.json` names the skills in `src/vaice/skills.py` (`disk_space`, `battery_status`, `system_info`, `top_processes`, `find_files`, the actions, and with `--memory` `remember`, `forget`, `forget_everything`); its must-not-call cases (`{"tools": []}`) include things a skill might wrongly match (the time, opening an app, the weather, a timer, volume). `tests/test_eval_scoring.py` checks that every expected tool is a real skill and every skill is exercised.

`vision_cases.json` is for `--vision` only: 14 cases. 7 questions about the screen should call `look_at_screen` (two of them follow an earlier screen answer and must look again); the rest must not (one asks about that earlier answer, the others are battery, disk space, "What is a screen saver?", general knowledge and opening an app). The stub answers from a fixed fake screen (a Numbers window titled Quarterly report, and a dialog "Your disk is almost full" when the question is about a dialog, error or message), so `answer_has` is checked against that text. The vision model itself is not measured here.

`browse_cases.json` is for `--browse` only: requests to do something on a website that should call `browse` (the goal must name the site or topic, matched with `contains`), questions about a running task, or what a finished one found, that should call `browser_status`, a request to stop it that should call `stop_browsing`, and requests that must call none of them. The run ends at the call (`browse` answers `Working in the browser.`), so what the browser agent does is not measured here.

`shell_cases.json` is for `--shell` only: spoken requests to run a command, tempting must-not-call ones such as "What does the ls command do?" and "Goodbye.", and two other skills.

To add a case: append an entry with a unique `id` (a slug, also the recording's file name) and an expectation using the skills in `src/vaice/skills.py`, then run `.venv/bin/pytest tests/test_eval_scoring.py` to check the files are consistent. To add a skill, it needs a stub answer in `evals/run.py` (`SKILL_STUBS`, `ACTION_STUBS`, ...).

## Real-mic recordings

Synthesized speech is clean; a real microphone is not, and the two can differ a lot. To test with your own voice, record the sentence in the case file and save it as `evals/recordings/<case id>.wav`: 16-bit PCM WAV, mono, any sample rate (the assistant resamples). When a recording exists it is used instead of `say`. A partial set is fine: the other cases still use `say`.

A real recording set beats `say`: put 16-bit PCM mono WAVs named `<case id>.wav` in `evals/recordings/`.

## Offline VAD silence tuning

`evals/vad.py` benchmarks `fastrtc`'s stateless chunked `ReplyOnPause` against `InterruptSafeReplyOnPause`'s 32 ms streaming Silero VAD offline (no `llama-server`, browser, or mic required, `#64`):

```bash
.venv/bin/python evals/vad.py
```

Options: `--recordings DIR` (optional 48 kHz mono PCM WAVs), `--voices Samantha,Daniel` (macOS `say` voices), `--out FILE` (default `evals/results/vad-<UTC timestamp>.json`). It measures reply-start latency on clean utterances (including 1-word commands), false mid-sentence turn splits across `300`–`800 ms` pauses, false triggers on brief noise/clap bursts, and steady-state CPU cost per second of audio.

## Caveats

Results are per llama.cpp build, model and quantization, and they move with the prompt and the history. Even at `--temp 0` a 2B model is not perfectly repeatable across audio, so one run proves nothing: compare `k/N` over repeats, and record the `--label` and the server info with any number you quote. Each run also runs a transcription session in parallel, as the app does.

Numbers measured before the single realtime path (the old text-routing brain, spoken `chat` mode) are not comparable with this harness's: the model now decides inside a speech session, and tool results are not replayed in history.

## Measured

| When | Setup | Result |
|---|---|---|
| 2026-11-19, before the switch | A spike harness (not this file) running both paths on the same cases: LFM3-Omni-2B Q8_0, llama-server b12359, `--temp 0`, the 53 cases without memory, 5 repeats | One-session path 219/265 correct (should call 139/170, must not call 80/95), first audio median 0.82 s; the earlier steps path 248/265, answer text after 1.07 s |
| 2026-11-19, after the switch | This harness, the same server, `--repeats 1`, 6 cases | 5/6 correct (the miss: `disk-free`, also missed in the spike), first audio median 0.74 s |
| 2026-11-24, offline VAD sweep (`#64`) | `evals/vad.py` on 48 kHz `say` corpus (`Samantha`, `Daniel`, 6 phase offsets; `SileroVADModel`) | Chunked `150`/`100`/`75 ms`: `581/782 ms` med/p90, `5/48` missed, `22/36` FS@300 ms, `34/36` FS@500 ms. Streaming `750 ms` (`VAICE_VAD_MIN_SILENCE_MS=750`): `814/862 ms`, `0/48` missed, `0/36` FS at 300/500/600 ms, `0/6` noise/clap FPs, `4.5 ms/s` CPU |

A full run of this harness on the current path has not been recorded yet; quote one here with its label and server info when it is.
