# Installation and requirements


## Quickstart (From Scratch in 3 Steps)

### 1. Install `llama-server`

`vaice` requires [llama.cpp](https://github.com/ggml-org/llama.cpp) `llama-server` **b11960 or newer** (v0.8.0+) for the `/v1/realtime` WebSocket. Install it via Homebrew (macOS/Linux):

```bash
brew install llama.cpp
```

*(If you prefer to build `llama-server` from source or download a pre-built release binary, see the [llama.cpp installation guide](https://github.com/ggml-org/llama.cpp#usage).)*

### 2. Install and run `vaice up`

Clone this repository, install the locked environment with [uv](https://docs.astral.sh/uv/), and run `vaice up`:

```bash
git clone https://github.com/ranacseruet/vaice.git && cd vaice
uv sync --extra dev
.venv/bin/vaice up
```

`vaice up` makes sure a `llama-server` is running, then starts the UI and opens it in your browser:

- If something already answers `GET /health` at `VAICE_LLAMACPP_URL` (the default is `http://localhost:8001`), it is used as it is and never stopped.
- Otherwise it starts `llama-server -hf LiquidAI/LFM3-Omni-2B-GGUF:Q8_0 --temp 0 --host localhost --port 8001` (host and port come from `VAICE_LLAMACPP_URL`, which must be a local address for this; the model from `VAICE_LLAMACPP_MODEL`; add flags such as `--ctx-size 16384` with `VAICE_LLAMACPP_SERVER_ARGS`) and waits for it to load. The first run downloads about 2.4 GB of model weights (about 4 GB of memory once loaded), so the wait can be long (`VAICE_LLAMACPP_START_TIMEOUT`, default 900 s); a progress line appears every 10 s.
- It checks that `llama-server` is installed and is build 11960 or newer, and tells you how to install or upgrade it. It never installs anything itself.
- The server's output goes to `~/.vaice/llama-server.log` (`$VAICE_DATA_DIR/llama-server.log`; a temporary directory when `VAICE_DATA_DIR` is empty). Open it if the start fails or takes long.
- When `vaice` stops (Ctrl+C or any error), the `llama-server` it started is stopped too: terminated, then killed if it does not exit within 5 s.

Plain `vaice` (no `up`) never starts or stops a server and never opens the browser: it expects `llama-server` to be running already.

### 3. Talk in the browser

When you see `Running on local URL: http://127.0.0.1:7860`:

1. Open `http://127.0.0.1:7860` in Chrome (`vaice up` opens it for you).
2. Put on **headphones** (otherwise speaker audio feeds back into the microphone).
3. Allow microphone access, click **Record**, ask a question, and pause when done.

For each reply, the terminal logs `[assistant] time to first audio: N.NNs` and `[assistant] reply finished in N.NNs`.

### Manual alternative: run `llama-server` yourself

To control the server directly, run it in one terminal and plain `vaice` in a second one:

```bash
llama-server -hf LiquidAI/LFM3-Omni-2B-GGUF:Q8_0 --temp 0 --port 8001
```

```bash
.venv/bin/vaice
```

> **Sampling:** the LFM3-Audio model card recommends **greedy decoding (temperature 0) for text output** — hence `--temp 0` above (and in the command `vaice up` builds). The `/v1/realtime` API has no temperature field, so reply text is sampled with the server's own setting; at the default 0.8 a 2B audio model drifts off-topic and loses the conversation thread (`vaice` warns about this at startup — and logs a distinct hint when it cannot verify the temperature at all, e.g. a router-style `/props`). Speech sampling is separate and comes from the GGUF (audio temperature 1.0, top_k 128, repetition penalty 1.1), so no audio flags are needed.
> **Language & voice:** voice chat is English-only for now; ASR and TTS cover 18 languages. Leave `--tts-lang`/`--tts-voice` out of the `llama-server` command: for an English-only voice model the pin buys nothing, and for LFM3 templates llama.cpp turns those flags (and per-session `audio.output.language`/`audio.output.voice`) into a `Speak in English using a female voice.` line prepended to the system prompt. An early single-session repro blamed that line for making the model parrot your words back, but a controlled re-test on the same build (b12359, `Q8_0 --temp 0`, clean TTS and noisy mic inputs) did **not** reproduce the parroting — with or without the flags; on noisy input the injected line degraded one reply into a rambling run-on. Treat the flags as unnecessary for voice chat, not as a proven echo cause. Leave the *transcription* language alone too — the default auto-detect is measurably better than pinning it.
> **Context:** the model supports 32K tokens. If you pass `-c/--ctx-size` explicitly, use at least 8192 — a realtime conversation that overflows its slot fails with "the conversation does not fit in the context".
> **How `-hf` works:** `-hf` (`--hf-repo`) is a built-in `llama-server` flag — you do **not** need the `huggingface-cli` / `hf` command-line tool installed. On first run, `llama-server` downloads the GGUF model weights and audio projector directly from Hugging Face into its local cache (`~/Library/Caches/llama.cpp` on macOS or `~/.cache/llama.cpp` on Linux) and starts listening on `http://localhost:8001`.

---

## Requirements & Compatibility

- **Python 3.12** (tested in CI on macOS; `uv sync` downloads it when it is missing; dependencies also resolve on 3.13 and 3.14, but runtime behavior is not validated there).
- **[uv](https://docs.astral.sh/uv/)** for reproducible dependency management (`uv sync` installs the exact versions from `uv.lock`).
- **llama.cpp `llama-server` b11960+** running a model with audio output.
  - **Only llama.cpp `llama-server` is supported and tested.** `vaice` speaks its `/v1/realtime` WebSocket protocol and checks `/health` and `/v1/models` at startup to verify that the loaded model advertises audio output (`output_modalities` includes `"audio"`). It also reads `/props` and warns when the server's text sampling temperature is above 0 (the LFM3-Audio model card recommends `--temp 0`); when `/props` reports no params or the read fails, `vaice` logs a distinct hint instead of staying silent, so a missing warning is never mistaken for `--temp 0` being set. Other servers (Ollama, LM Studio, oMLX, …) will not work.
  - Point `VAICE_LLAMACPP_URL` at a different host or port if `llama-server` isn't on `:8001`.
  - Current upstream `llama.cpp` documentation also describes `/v1/realtime` conversation support for LFM3-Omni and the `--reasoning off` server option. The default model is LFM3-Omni-2B (`Q8_0`): it speaks and listens like LFM3-Audio and also reads images, which is what `look_at_screen` needs, so one server can do both (see [Screen vision](screen-vision.md)). It is the model the live eval and the browser checks ran on (b12359; see `evals/README.md`); LFM3-Audio-2B (`LiquidAI/LFM3-Audio-2B-GGUF:Q4_K_M`) still works as a smaller, audio-only alternative. On `llama.cpp` builds that provide the option, add `--reasoning off` to the server command if your selected model's chat template supports disabling reasoning. See the [llama.cpp server documentation](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md).
  - CI runs on macOS; Linux and Windows audio behavior has not been validated.
