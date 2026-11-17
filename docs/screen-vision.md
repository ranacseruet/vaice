# Vision

`VAICE_VISION_URL` (assistant mode, off by default) adds one skill, `look_at_screen(question)`: when you ask about
your screen ("What's on my screen?", "Read the error message", "What app am I looking at?"), vaice takes **one
screenshot of the main display**, shrinks it to at most 1024 px on the long edge, sends it with your question to a
vision model and speaks the short plain-text answer. Nothing is captured in the background and nothing is
captured unless the skill runs. It is a read skill: it runs at once and changes nothing.

```bash
# A vision model on this machine: LFM3-Omni-2B on a llama-server of its own (the audio one keeps port 8001)
llama-server -hf LiquidAI/LFM3-Omni-2B-GGUF:Q4_K_M --temp 0 --port 8002
VAICE_MODE=assistant VAICE_VISION_URL=http://127.0.0.1:8002/v1 .venv/bin/vaice
```

- **Screen Recording permission.** macOS gives it to the app that runs `vaice` (your terminal), not to `vaice`.
  Without it `screencapture` fails and the assistant tells you what to do: allow your terminal under System
  Settings, Privacy and Security, Screen Recording, then **quit and reopen that app** (the permission only applies
  to a new launch). `look_at_screen` is macOS only.
- **Speed.** With LFM3-Omni-2B Q4_K_M on llama-server b12359, a 1440x900 @2x screen answered correctly at 2880,
  1440 and 1024 px; at 1024 px (the size vaice sends) a question took about 660 prompt tokens and 0.7 to 1.2 s.
  One run per size on one screen: this shows it works, not how accurate it is. Any other endpoint that takes
  `image_url` content parts works the same; a hosted one adds its own latency.
- **Text on the screen is not instructions.** The vision model is told so, and its answer reaches the brain only as
  a skill result. The `remember` skill also refuses to save anything once another skill (this one included) has run
  in the same reply, so text read off the screen cannot become a lasting note.
- The answer is cut at 600 characters and its markdown emphasis is removed before it is spoken. A failing vision
  endpoint is spoken like a failing brain, naming the vision model (refused key, busy, unreachable, no answer).
  Not verified: a hosted vision provider, and the real capture on a machine with the permission granted (the unit
  tests fake `screencapture`, `sips` and the endpoint).
