# vaice

Talk to a local model and hear it answer. Your mic audio goes **directly** to the model
(no separate speech-to-text step), and the spoken reply streams back as it is generated.

Two interchangeable backends produce the reply:

| Backend | How it works | Strengths |
|---|---|---|
| **native** | One speech-to-speech model ([LFM2.5-Audio-1.5B](https://huggingface.co/mlx-community/LFM2.5-Audio-1.5B-8bit) via mlx-audio) turns audio into audio | Fastest: first audio ≈ 0.3 s after you stop talking |
| **cascade** | Gemma (llama.cpp) hears the audio and streams text; Kokoro speaks each sentence as it completes | Much bigger model (12B) → smarter answers; ≈ 4 s to first audio |

`auto` (the default) uses native if it loads and falls back to cascade, with a warning, if not.

```
browser mic ─WebRTC─▶ fastrtc (VAD: detects you stopped talking)
                         │  audio
                         ▼
                  Responder.respond(audio) ──▶ audio chunks ─WebRTC─▶ browser speakers
                   ├─ NativeResponder   mlx-audio speech-to-speech model
                   └─ CascadeResponder  llama-server :8001 → sentences → Kokoro TTS
```

## Requirements

- Apple Silicon Mac (mlx-audio runs on MLX)
- Python 3.10–3.12 (spaCy, pulled in by `misaki[en]`, has no wheels for 3.13+ yet)
- [uv](https://docs.astral.sh/uv/)
- **Native backend:** nothing else; the model (~2.3 GB) downloads on first run.
- **Cascade backend:** a llama.cpp `llama-server` running an **audio-capable** model with its `--mmproj`, e.g.:

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
.venv/bin/vaice                      # auto: native if it loads, else cascade
.venv/bin/vaice --backend cascade    # force Gemma + Kokoro (needs llama-server on :8001)
.venv/bin/vaice --backend native     # force the speech-to-speech model; fail if it can't load
```

The first run downloads models (about a minute, longer for native). When you see
`Running on local URL: http://127.0.0.1:7860`, open it in Chrome, allow the microphone,
click record, ask a question, then stop talking. Use headphones, otherwise the speakers
feed back into the mic.

The log prints `using native|cascade backend` at startup, and for every reply
`[native] time to first audio: N.NNs` / `[native] reply finished in N.NNs`, so you can always
tell which path answered and compare the two backends. A `WARNING ... falling back to cascade`
means `auto` couldn't load native. Run it once with each
`--backend` and ask the same question.

Notes on the native backend: LFM2.5-Audio is a 1.5B model, English only, so expect simpler
answers than Gemma 12B. Neither backend keeps conversation history; every question is
answered on its own.

## Configuration

All optional, via environment variables:

| Variable | Default | Meaning |
|---|---|---|
| `VAICE_BACKEND` | `auto` | `auto`, `cascade` or `native` (overridden by `--backend`) |
| `VAICE_PORT` | `7860` | UI port |
| `VAICE_NATIVE_MODEL` | `mlx-community/LFM2.5-Audio-1.5B-8bit` | mlx-audio speech-to-speech model; empty disables native |
| `VAICE_NATIVE_PROMPT` | `Respond with interleaved text and audio. Reply naturally and concisely in 1 to 2 spoken sentences.` | system prompt for native |
| `VAICE_LLM_URL` | `http://localhost:8001/v1/chat/completions` | llama-server chat endpoint (cascade) |
| `VAICE_LLM_MODEL` | `gemma-4-12b` | model name sent in the request (informational for llama-server) |
| `VAICE_PROMPT` | `Reply naturally and concisely in 1 to 2 spoken sentences.` | instruction sent before the audio (cascade) |
| `VAICE_LLM_TIMEOUT` | `120` | seconds (cascade) |
| `VAICE_TTS_MODEL` | `mlx-community/Kokoro-82M-bf16` | MLX Kokoro repo (cascade) |
| `VAICE_VOICE` | `af_heart` | Kokoro voice (cascade) |
| `VAICE_SPEED` | `1.1` | speech speed (cascade) |

## Tests

```bash
.venv/bin/pytest
```

Unit tests use fakes for the models: sentence splitting, SSE parsing (including llama.cpp's
`"content": null` first chunk), cascade streaming order, native frame chunking, and backend
selection/fallback. The real audio path needs the models and a mic, so verify it by hand
with the steps above.

## Layout

```
src/vaice/
  app.py            CLI + Gradio/WebRTC UI; wraps a Responder, logs time to first audio
  config.py         env-driven settings
  audio.py          mic frame → base64 WAV / float mono
  responders/
    base.py         Responder protocol (audio in → audio chunks out)
    __init__.py     create_responder(): auto / cascade / native selection + fallback
    cascade.py      llama.cpp text stream → sentences → Kokoro
    native.py       mlx-audio speech-to-speech model
  llm.py            streaming llama.cpp client (cascade)
  sentences.py      text fragments → sentences; markdown cleanup for TTS (cascade)
  tts.py            Kokoro synthesis (cascade)
tests/
tasks/todo.md       plan and review notes for the current change
```

To add a backend, implement `respond(audio) -> Iterator[(sample_rate, float32 samples)]`
and register it in `responders/__init__.py`.

## Notes

- MLX streams are bound to the thread that created them, but fastrtc resumes the reply
  generator from a different worker thread for each chunk. `NativeResponder` therefore loads
  and runs the model on one dedicated thread and hands chunks back through a queue. Keep any
  new MLX-based backend off fastrtc's threads the same way.
- Gemma needs the text prompt **before** the audio in the message content.
- Raw 48 kHz browser audio works as-is on both backends: llama.cpp and the mlx-audio
  processor each resample internally.
- llama.cpp's `llama-server` only outputs text; it ignores `modalities`/`audio` request
  fields, which is why the cascade needs a separate TTS step.
