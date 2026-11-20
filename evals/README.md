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
- `--browse`: also offer `browse`, `browser_status` and `stop_browsing` (stubbed as above).
- `--prompt TEXT`: the reply instructions (default `VAICE_PROMPT`).
- `--history/--no-history` (default on: two fixed earlier exchanges, `Hey, are you there?` / `Yes, I'm here.` and `What day is it?` / `It's Monday.`).
- `--out FILE` (default `evals/results/<UTC timestamp>.json`, which is gitignored) and `--label TEXT`.

Ctrl+C stops the run and reports the runs so far. The full `cases.json` is 69 cases, around 350 runs at 5 repeats; use `--cases` and `--repeats 1` while iterating.

## Read the report

The summary prints the count and percentage of each verdict, `duplicate`, and the timing lines. Then a per-case table of `k/N correct`: a case at `3/5` is flaky, `0/5` is a plain failure. Percentages are of all runs, errors included.

Runs whose transcript differs from the spoken sentence (case and punctuation ignored) are listed, so a wrong call caused by a mishearing is told apart from a model failure. The comparison is literal: the transcript `50` against the sentence `fifty` counts as a difference.

The results file holds the config, server info, label, and every run's raw record (calls, texts, timings, error) with its score, so a surprising number can be traced to the run that produced it.

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

`vision_cases.json` is for `--vision` only: 5 questions about the screen that should call `look_at_screen`, and the rest must not (battery, disk space, "What is a screen saver?", general knowledge, opening an app). The stub answers from a fixed fake screen (a Numbers window titled Quarterly report, and a dialog "Your disk is almost full" when the question is about a dialog, error or message), so `answer_has` is checked against that text. The vision model itself is not measured here.

`browse_cases.json` is for `--browse` only: requests to do something on a website that should call `browse` (the goal must name the site or topic, matched with `contains`), questions about a running task that should call `browser_status`, a request to stop it that should call `stop_browsing`, and requests that must call none of them. The run ends at the call (`browse` answers `Working in the browser.`), so what the browser agent does is not measured here.

`shell_cases.json` is for `--shell` only: spoken requests to run a command, tempting must-not-call ones such as "What does the ls command do?" and "Goodbye.", and two other skills.

To add a case: append an entry with a unique `id` (a slug, also the recording's file name) and an expectation using the skills in `src/vaice/skills.py`, then run `.venv/bin/pytest tests/test_eval_scoring.py` to check the files are consistent. To add a skill, it needs a stub answer in `evals/run.py` (`SKILL_STUBS`, `ACTION_STUBS`, ...).

## Real-mic recordings

Synthesized speech is clean; a real microphone is not, and the two can differ a lot. To test with your own voice, record the sentence in the case file and save it as `evals/recordings/<case id>.wav`: 16-bit PCM WAV, mono, any sample rate (the assistant resamples). When a recording exists it is used instead of `say`. A partial set is fine: the other cases still use `say`.

## Caveats

Results are per llama.cpp build, model and quantization, and they move with the prompt and the history. Even at `--temp 0` a 2B model is not perfectly repeatable across audio, so one run proves nothing: compare `k/N` over repeats, and record the `--label` and the server info with any number you quote. Each run also runs a transcription session in parallel, as the app does.

Numbers measured before the single realtime path (the old text-routing brain, spoken `chat` mode) are not comparable with this harness's: the model now decides inside a speech session, and tool results are not replayed in history.
