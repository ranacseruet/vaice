# Tool-calling evals

A live harness that measures how reliably spoken commands turn into the right tool call, with repeats, against a real `llama-server`. It is a developer tool: the app does not import it and CI does not run it (only `scoring.py` is unit-tested, in `tests/test_eval_scoring.py`). Use it before making a claim about how the model behaves, and after changing a prompt, the tool set or the server settings.

## What it measures

Each case is a spoken sentence and an expectation (see [Cases](#cases)). Every case runs `--repeats` times on two paths (default), plus an opt-in third, `assistant`:

- **interleaved**: the app's own path. `LlamaCppResponder` sends the audio over `/v1/realtime` with the stub tools declared, the model answers in text and audio, and the tool calls it makes are recorded. Time to first audio is measured. Tool results are fed back, so a call is followed by a spoken answer, as in the app.
- **text**: the audio is transcribed by a realtime `transcription` session, then the transcript is routed through `POST /v1/chat/completions` with the same tools (`temperature` 0). Routing only: no tool result is fed back. Time to decision is transcription plus routing.

- **assistant** (opt-in: `--paths assistant`): assistant mode's own code up to the answer text, without `/tts`. The audio is transcribed, then `build_messages` and `decide` (the brain/skills loop of `AssistantResponder`, `POST /v1/chat/completions`) run with the real skill definitions and the prompt `VAICE_ASSISTANT_PROMPT`. The skills are stubbed with canned results (`disk_space` says `160 gigabytes free of 460 gigabytes`, and so on; nothing is executed; a `confirm` skill such as `quit_app` always asks, and the pending call counts as the call, since the confirmation itself is unit-tested), so the calls are scored like the other paths, and the final answer text is recorded. Time to answer text is transcription plus the brain rounds. It has its own cases (`assistant_cases.json`, 65 cases, 25 of them must-not-call), used automatically when `assistant` is the only path; `--cases-file` picks another file. It is not in the default `--paths` because its tools are the app's skills, not `tools.json`.

  Memory is on in this path (the app's default): the brain also gets `remember`, `forget` and `forget_everything`, which work for real on a **fresh in-memory store for every run** (`Memory(None)`, refused if it has a path), so a run never reads or writes your `~/.vaice/memory.json` and one run's notes never reach the next. A case may carry `"notes": [...]`, loaded into that store before the run and put into the system prompt exactly as the app does (`NOTES_HEADER`); `forget_everything` asks and counts as the call, like the other `confirm` skills. The 14 `memory-*` cases are: 5 spoken `remember` requests, 3 recall questions (two answered from the notes, one distractor answered from the model's knowledge), `forget` and `forget_everything`, and 4 must-not-call ones ("Do you remember the movie Titanic?", "I can never remember my passwords.", "What do you know about me?", "Remind me about my dentist appointment on Friday."; vaice cannot remind, and the first full eval had both models turn "Remind me to call mom tomorrow at nine." into `remember`). `--no-memory` runs the assistant path without memory: the previous skill set, no notes, and the memory cases are then expected to fail, so pass `--cases` or another `--cases-file`. A single run says little, use `--repeats`.

  `--shell` also offers the `run_shell` skill (the app's `VAICE_ASSISTANT_SHELL`): its stub always asks and nothing is run (the stub has no runner, so no command is started on your machine), and the pending call counts as the call. Its cases are in `assistant_shell_cases.json` (`--cases-file assistant_shell_cases.json`, a name in `evals/` or an absolute path; 12 cases: spoken requests to run a command, tempting must-not-call ones such as "What does the ls command do?" and "Goodbye.", and two other skills). The call verdict only expects the tool; what the command says is checked apart, by the optional `command_matches` / `command_not` (see [Cases](#cases)). Without `--shell` the skill set is unchanged and that file does not apply (`run_shell` would be an unknown tool).

  `--brain-url` and `--brain-model` point the assistant path's brain at an OpenAI-compatible endpoint (the base URL, `/v1` included, e.g. `http://localhost:1234/v1`; the defaults are `VAICE_BRAIN_URL` and `VAICE_BRAIN_MODEL`, empty = the local llama-server). The API key is read only from the environment (`VAICE_BRAIN_API_KEY`, or a `.env`), never from a flag, and the results file records the brain's host and model, never the key or the URL's path. The report header and the `assistant` heading say which brain ran. Transcription still uses the local llama-server. **Privacy:** with `--brain-url` on a remote host, the case sentences (as transcripts), the history and the stub skills' results are sent to that host (the script prints a note); the stubs hold no real data (the computer's name is `Example-MacBook`). Without temperature 0 a helper's runs vary, so use `--repeats` and read the per-case `k/N`; a single run says little.

The tools in `tools.json` are stubs with canned results (`disk_space` says `160 gigabytes free of 460`, and so on); nothing is ever executed on your machine.

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
.venv/bin/python evals/run.py                                   # all cases, both paths, 5 repeats
.venv/bin/python evals/run.py --repeats 1 --cases open-safari,know-sky   # a quick check
.venv/bin/python evals/run.py --paths interleaved --label Q8_0  # one path; label stored in the results
.venv/bin/python evals/run.py --prompt "Use the available tools to answer questions about this system."
```

```bash
.venv/bin/python evals/run.py --paths assistant --repeats 1      # assistant mode: its own cases and skills
.venv/bin/python evals/run.py --paths assistant --shell --repeats 1 --cases-file assistant_shell_cases.json   # with run_shell
```

Options: `--url`, `--no-memory` (assistant path: leave out the memory skills and notes), `--brain-url` / `--brain-model` (assistant path: the brain's endpoint and model), `--shell` (assistant path: also offer the stubbed `run_shell` skill), `--paths interleaved,text,assistant` (default `interleaved,text`), `--cases-file FILE` (a file in `evals/` or a path; default `cases.json`, or `assistant_cases.json` when `--paths` is `assistant` alone), `--repeats N` (default 5), `--cases id,id`, `--prompt TEXT` (the reply instructions on the interleaved path and the system prompt on the text path; the default is `VAICE_LLAMACPP_PROMPT` for interleaved and a short spoken-assistant prompt for text), `--history/--no-history` (default on: two fixed earlier exchanges, `Hey, are you there?` / `Yes, I'm here.` and `What day is it?` / `It's Monday.`), `--out FILE` (default `evals/results/<UTC timestamp>.json`, which is gitignored) and `--label TEXT`. Ctrl+C stops the run and reports the runs so far.

The full set is about 33 cases x 2 paths x 5 repeats, around 330 runs; use `--cases` and `--repeats 1` while iterating.

## Read the report

The summary prints per path the count and percentage of each verdict, `duplicate`, and the median and max time to first audio (interleaved) or to decision (text). Then a per-case table of `k/N correct` for each path: a case at `3/5` is flaky, `0/5` is a plain failure. Percentages are of all runs, errors included.

Text-path runs whose transcript differs from the spoken sentence (case and punctuation ignored) are listed, so a wrong tool call caused by a mishearing is told apart from a routing failure. The comparison is literal: the transcript `50` against the sentence `fifty` counts as a difference.

The results file holds the config, server info, label, and every run's raw record (calls, texts, timings, error) with its score, so a surprising number can be traced to the run that produced it.

## Cases

`cases.json` is a list of `{"id", "say", "expect"}`. `expect` is one of:

```json
{"tools": ["open_app"], "args": {"open_app": {"name": "Safari"}}}
{"tools": ["open_app", "set_volume"]}
{"tools": []}
{"any_of": ["run_shell", "disk_space"]}
```

- `tools`: exactly these tools are called (order and repeated calls do not matter; an extra different tool is a `missed`). `args` is optional and matches per tool as a subset: strings compare case-insensitively after stripping, numbers by value (`20`, `20.0` and `"20"` agree), a list is a set of accepted alternatives (`"name": ["Notes", "Notes app"]`), and `{"contains": "text"}` accepts a string that holds `text`, case ignored (free text such as a note the model words itself). If a tool is called more than once, one matching call is enough.
- `{"tools": []}`: no tool may be called (chit-chat, general knowledge).
- `any_of`: at least one call, and every call is to a tool in the list. Arguments are not checked.

Two optional expectations on the assistant path say something about the final answer or the command, apart from the call verdict (a call can be right while the answer is not). They sit in `expect` next to `tools`:

```json
{"tools": ["disk_space"], "answer_has": ["160"]}
{"tools": [], "answer_has": ["Nadia"], "notes": ["The user's sister is called Nadia."]}
{"tools": ["run_shell"], "command_matches": "Downloads", "command_not": "/Users/me"}
```

- `answer_has`: every listed string must appear in the final answer text, case ignored (with `tools: []` it checks a recall from the case's `notes`). Not applicable (`-`) when the run ended in a confirmation question (the pending call, such as `quit_app` or `run_shell`): that question is not the answer.
- `command_matches` / `command_not`: regular expressions searched, case-sensitive, in the `command` argument of the `run_shell` call (every `run_shell` call must satisfy both; no such call fails the check). The `run_shell` stub only asks, so the command is the model's proposal, never run.

The summary prints `answer faithful k/N` and `command ok k/N` when a case asked for them (over completed runs), and the per-case table gets an `answer` and a `command` column. The results JSON gains `answer_ok` / `command_ok` (`true`, `false` or `null`) in each run's `score`; nothing else changed. They are exact-string checks of a stub fact: the answer `one hundred sixty` fails `160`, and `git -C ~/projects status` fails `git status`, so read a failing run before blaming the model.

`assistant_cases.json` has the same shape; its `tools` name the skills in `src/vaice/skills.py` (`disk_space`, `battery_status`, `system_info`, `top_processes`, `find_files`, `remember`, `forget`, `forget_everything`), `any_of` is not used, and its must-not-call cases (`{"tools": []}`) include things a skill might wrongly match (the time, opening an app, the weather, a timer, volume). `tests/test_eval_scoring.py` checks that every expected tool is a real skill and every skill is exercised.

To add a case: append an entry with a unique `id` (a slug, also the recording's file name) and an expectation using the tools in `tools.json`, then run `.venv/bin/pytest tests/test_eval_scoring.py` to check the files are consistent. To add a tool, add it to `tools.json` as `{"definition": <flat Responses API definition>, "result": "<canned result>"}`.

## Real-mic recordings

Synthesized speech is clean; a real microphone is not, and the two can differ a lot. To test with your own voice, record the sentence in `cases.json` and save it as `evals/recordings/<case id>.wav`: 16-bit PCM WAV, mono, any sample rate (the responder resamples). When a recording exists it is used instead of `say`, for both paths. A partial set is fine: the other cases still use `say`.

## Caveats

Results are per llama.cpp build, model and quantization, and they move with the prompt and the history. Even at `--temp 0` a 2B model is not perfectly repeatable across audio, so one run proves nothing: compare `k/N` over repeats, and record the `--label` and the server info with any number you quote. The interleaved path also runs a transcription session in parallel, as the app does, which loads the server a little more than the text path's sequential requests do.

Results recorded before the fix for #90 include about 1 s of socket close in the text path's
`transcribe_s` and time to decision; the transcript itself took about 0.2 s.
