# Tool-calling evals

A live harness that measures how reliably spoken commands turn into the right tool call, with repeats, against a real `llama-server`. It is a developer tool: the app does not import it and CI does not run it (only `scoring.py` is unit-tested, in `tests/test_eval_scoring.py`). Use it before making a claim about how the model behaves, and after changing a prompt, the tool set or the server settings.

## What it measures

Each case is a spoken sentence and an expectation (see [Cases](#cases)). Every case runs `--repeats` times on two paths:

- **interleaved**: the app's own path. `LlamaCppResponder` sends the audio over `/v1/realtime` with the stub tools declared, the model answers in text and audio, and the tool calls it makes are recorded. Time to first audio is measured. Tool results are fed back, so a call is followed by a spoken answer, as in the app.
- **text**: the audio is transcribed by a realtime `transcription` session, then the transcript is routed through `POST /v1/chat/completions` with the same tools (`temperature` 0). Routing only: no tool result is fed back. Time to decision is transcription plus routing.

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

Options: `--url`, `--paths interleaved,text`, `--repeats N` (default 5), `--cases id,id`, `--prompt TEXT` (the reply instructions on the interleaved path and the system prompt on the text path; the default is `VAICE_LLAMACPP_PROMPT` for interleaved and a short spoken-assistant prompt for text), `--history/--no-history` (default on: two fixed earlier exchanges, `Hey, are you there?` / `Yes, I'm here.` and `What day is it?` / `It's Monday.`), `--out FILE` (default `evals/results/<UTC timestamp>.json`, which is gitignored) and `--label TEXT`. Ctrl+C stops the run and reports the runs so far.

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

- `tools`: exactly these tools are called (order and repeated calls do not matter; an extra different tool is a `missed`). `args` is optional and matches per tool as a subset: strings compare case-insensitively after stripping, numbers by value (`20`, `20.0` and `"20"` agree), and a list is a set of accepted alternatives (`"name": ["Notes", "Notes app"]`). If a tool is called more than once, one matching call is enough.
- `{"tools": []}`: no tool may be called (chit-chat, general knowledge).
- `any_of`: at least one call, and every call is to a tool in the list. Arguments are not checked.

To add a case: append an entry with a unique `id` (a slug, also the recording's file name) and an expectation using the tools in `tools.json`, then run `.venv/bin/pytest tests/test_eval_scoring.py` to check the files are consistent. To add a tool, add it to `tools.json` as `{"definition": <flat Responses API definition>, "result": "<canned result>"}`.

## Real-mic recordings

Synthesized speech is clean; a real microphone is not, and the two can differ a lot. To test with your own voice, record the sentence in `cases.json` and save it as `evals/recordings/<case id>.wav`: 16-bit PCM WAV, mono, any sample rate (the responder resamples). When a recording exists it is used instead of `say`, for both paths. A partial set is fine: the other cases still use `say`.

## Caveats

Results are per llama.cpp build, model and quantization, and they move with the prompt and the history. Even at `--temp 0` a 2B model is not perfectly repeatable across audio, so one run proves nothing: compare `k/N` over repeats, and record the `--label` and the server info with any number you quote. The interleaved path also runs a transcription session in parallel, as the app does, which loads the server a little more than the text path's sequential requests do.

Results recorded before the fix for #90 include about 1 s of socket close in the text path's
`transcribe_s` and time to decision; the transcript itself took about 0.2 s.
