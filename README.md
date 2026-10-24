# vaice

Talk to a local LLM and hear it answer, with no separate speech-to-text step.
Mic audio goes **directly** to Gemma (served by llama.cpp), and the reply text is
streamed into Kokoro TTS sentence by sentence, so you hear the first sentence while
the rest is still being generated.

```
browser mic ─WebRTC─▶ fastrtc (VAD: detects you stopped talking)
                         │  WAV (base64)
                         ▼
                 llama-server :8001  (Gemma, audio input, streamed text)
                         │  text deltas → sentence splitter
                         ▼
                 Kokoro TTS (MLX) ─WebRTC─▶ browser speakers
```

## Requirements

- Apple Silicon Mac (Kokoro runs on MLX)
- Python 3.10–3.12 (spaCy, pulled in by `misaki[en]`, has no wheels for 3.13+ yet)
- [uv](https://docs.astral.sh/uv/)
- A llama.cpp `llama-server` running an **audio-capable** model with its `--mmproj`, e.g.:

  ```bash
  llama-server --model ./gemma-4-12b/gemma-4-12B-it-UD-Q4_K_XL.gguf \
    --mmproj ./gemma-4-12b/mmproj-BF16.gguf --port 8001 \
    --chat-template-kwargs '{"enable_thinking":false}'
  ```

## Setup

```bash
uv venv .venv            # picks Python 3.12 from .python-version
uv pip install -e ".[dev]"
```

## Run

```bash
curl localhost:8001/health      # expect {"status":"ok"}
.venv/bin/vaice                 # or: .venv/bin/python -m vaice
```

The first run downloads Kokoro and a small spaCy model (about a minute). When you
see `Running on local URL: http://127.0.0.1:7860`, open it in Chrome, allow the
microphone, click record, ask a question, then stop talking. You should hear the
answer after a short pause. Use headphones, otherwise the speakers feed back into the mic.

## Configuration

All optional, via environment variables:

| Variable | Default | Meaning |
|---|---|---|
| `VAICE_LLM_URL` | `http://localhost:8001/v1/chat/completions` | llama-server chat endpoint |
| `VAICE_LLM_MODEL` | `gemma-4-12b` | model name sent in the request (informational for llama-server) |
| `VAICE_PROMPT` | `Reply naturally and concisely in 1 to 2 spoken sentences.` | instruction sent before the audio |
| `VAICE_LLM_TIMEOUT` | `120` | seconds |
| `VAICE_TTS_MODEL` | `mlx-community/Kokoro-82M-bf16` | MLX Kokoro repo |
| `VAICE_VOICE` | `af_heart` | Kokoro voice |
| `VAICE_SPEED` | `1.1` | speech speed |
| `VAICE_PORT` | `7860` | UI port |

## Tests

```bash
.venv/bin/pytest
```

Unit tests cover the sentence splitter and SSE parsing (including llama.cpp's
`"content": null` first chunk). The audio path itself needs the running model and a
real mic, so verify it by hand with the steps above.

## Layout

```
src/vaice/
  config.py     env-driven settings
  audio.py      mic frame → base64 WAV
  llm.py        streaming llama.cpp client
  sentences.py  text fragments → sentences; markdown cleanup for TTS
  tts.py        Kokoro synthesis
  app.py        wiring + Gradio/WebRTC UI (entry point: `vaice`)
tests/
```

## Notes

- Gemma needs the text prompt **before** the audio in the message content.
- Raw 48 kHz browser audio works as-is: llama.cpp resamples internally.
