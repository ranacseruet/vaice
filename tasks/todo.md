# Native speech backend (audio in → audio out)

Design: `Responder` protocol (audio_in → Iterator[audio_out]); `CascadeResponder` = current
llama.cpp → sentences → Kokoro flow; `NativeResponder` = mlx-audio speech-to-speech model.
`VAICE_BACKEND=auto|cascade|native`; auto = native if it loads, else cascade (warn loudly).
Out of scope: multi-turn memory.

- [x] 0. Spike: run LFM2.5-Audio standalone (API shape, 24 kHz input, worker-thread safety, time to first audio)
- [x] 1. Refactor: Responder + CascadeResponder + factory, no behavior change; tests with fakes; rerun real e2e
- [x] 2. NativeResponder, config, auto-selection + fallback, TTFA logging
- [x] 3. Tests (selection, fallback, native with fakes) + README  (no automated real-model test; verified by hand)
  - spike result: TTFA 0.3-0.7s, raw 48k input OK (processor resamples), worker thread OK, chunked output intelligible; detokenizer needs (1,8,T); strip <|text_end|>

## Review
- 21 unit tests; real e2e on both backends (factory → handler → audio → Gemma transcript ✓), fallback + strict-native error paths ✓,
  `vaice` entry point serves UI for auto/cascade ✓, native through real ReplyOnPause (VAD + worker thread) ✓.
- Bug caught by tests: to_mono_float skipped int16 scaling (mean() promotes to float64) – fixed.
- Found: first native generation is ~2.4s slower (cold kernels) → warm-up at load (0.33s afterwards).
- Not verified: live browser mic (cannot automate); native answer quality vs Gemma 12B is subjective.

## Code-review follow-up (PR #1)
Verified each finding before acting. Fixed: chunk seams (left-context decode; mean |diff| vs whole decode 0.0142 → 0.0007),
per-token cancellation (also covers a reply cancelled while queued), executor leak on failed load, token-limit warning
(512 reproduced with a long request), invalid VAICE_BACKEND traceback, mlx-audio lower bound, stale metadata/docstring,
test import guard + removal of wall-clock assertion, todo.md overclaim.
Documented rather than fixed: native model is single-session by design (README).
Not changed: consumer `chunks.get()` has no timeout – a wedged MLX/Metal call would hang that reply; out of scope for a demo.
