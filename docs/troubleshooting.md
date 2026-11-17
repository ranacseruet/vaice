# Troubleshooting

- **Replies are off-topic, ignore the question, or lose the conversation thread:**
  Start `llama-server` with `--temp 0`. The `/v1/realtime` API has no temperature field, so reply text is sampled with the server's own setting (default 0.8), and the LFM3-Audio model card recommends greedy decoding (temperature 0) for text output — `vaice` warns about this at startup (and logs a distinct hint when it cannot verify the temperature, so an absent warning does not mean greedy is fine). If the replies are still weak, check the quantization: the default `Q8_0` matches the unquantized model step-for-step in side-by-side runs, while 4-bit backbones (`Q4_K_M`, the smaller download) measurably drift. You can also lower `VAICE_HISTORY_TURNS` (e.g. `4`) — small models follow a short history better than a long one.
- **`llama-server` has no `/v1/realtime` endpoint or returns 404 on WebSocket connect:**
  Your `llama.cpp` installation is older than `b11960` (v0.8.0). Upgrade it (`brew upgrade llama.cpp` or install a newer release from [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)).
- **`RuntimeError: the model on llama-server ... does not output audio`:**
  The model loaded in `llama-server` is text-only or audio-input-only. Start `llama-server` with an audio-output model such as `LiquidAI/LFM3-Omni-2B-GGUF:Q8_0`.
- **The assistant repeats back exactly what you said (echo) instead of answering:**
  1. **Clear the poisoned history first:** click **New conversation** (or delete `~/.vaice/conversation.json` while the app is stopped). Each echo is stored and replayed to the model as ChatML `user: X → assistant: X` pairs, which teach it to echo — the loop self-reinforces across restarts.
  2. **Start `llama-server` with `--temp 0`** (see the off-topic replies item above — weak text sampling is the most common cause of bad replies, including parroting).
  3. **Leave `--tts-lang`/`--tts-voice` out** — not because they are a proven echo cause (an early single-session repro blamed them, but a controlled re-test on the same build did not reproduce it), but because the model is English-only by training, so the pin is unnecessary, and for LFM3 templates the flags alter the voice-chat system prompt (see the "Language & voice" note in *Quickstart*).
  4. On llama.cpp 1.0.0 the penalty flag is `--repeat-penalty` (the old `--rep-pen` alias is gone) — relevant if you experiment with sampling.
- **The assistant interrupts itself or hears its own voice:**
  Speaker audio is feeding back into your microphone and triggering `fastrtc`'s voice activity detection (VAD). Use headphones.
- **`concurrency_limit_reached` when opening a new tab or reconnecting:**
  `vaice` is a single-user demo (`concurrency_limit=1`). Normal page reloads close the old WebRTC connection immediately and reconnect within milliseconds. However, if a browser tab crashes or is force-closed without sending a WebRTC goodbye, the server holds the slot for ~30 seconds until ICE consent checks expire. Wait ~30 seconds and try again.
- **`AbortError: BodyStreamBuffer was aborted` / `Connection errored out` in the browser console on reload:**
  Harmless message from Gradio's frontend client when page navigation cuts its open `queue/data` SSE stream.
