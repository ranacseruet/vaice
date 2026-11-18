# Vision

`VAICE_VISION_URL` (assistant mode, off by default) adds one skill, `look_at_screen(question)`: when you ask about
your screen ("What's on my screen?", "Read the error message", "What app am I looking at?"), vaice takes **one
screenshot of the main display**, shrinks it to at most 1024 px on the long edge, sends it with your question to a
vision model and speaks the short plain-text answer. Nothing is captured in the background and nothing is
captured unless the skill runs. It is a read skill: it runs at once and changes nothing.

The default model, LFM3-Omni-2B, reads images, so the server you already run can be the vision model too:

```bash
VAICE_MODE=assistant VAICE_VISION_URL=http://localhost:8001/v1 .venv/bin/vaice up
```

It is off by default because it is not free: offering the skill to the 2B brain cost 15 of 255 points on the
51 ordinary assistant cases (235 against 250, LFM3-Omni-2B Q8_0, b12359, 5 repeats): the "find my screenshot" and
"turn the volume down" cases went from 5/5 to 0/5 with the extra tool on offer (which tool the model called instead
was not inspected). The 11 vision cases themselves score 50/55 (the dialog-box question misses). A separate vision
model (a bigger local one or a hosted one) is just another `VAICE_VISION_URL`.

- **Every question looks again.** A screenshot is taken each time the skill runs, never reused. The 2B model used to
  answer "what do you see now?" from its own earlier answer, which stays in the conversation, without calling the
  skill (replaying a real conversation: 0 of 5 looked again). The prompt now says live facts change and the skill's
  description says each call takes a new screenshot; replayed, 20 of 20 realistic histories look again, and the
  cases `vision-look-again`, `vision-look-again-changed` and `vision-recall-earlier` (a question about the earlier
  answer must not look again) pass 5/5. A reply that says "the same as before" without a new look is still possible
  with a 2B model: say "look at my screen again" to force it.
- **Screen Recording permission.** macOS gives it to the app that runs `vaice` (your terminal), not to `vaice`.
  Without it `screencapture` fails and the assistant tells you what to do: allow your terminal under System
  Settings, Privacy and Security, Screen Recording, then **quit and reopen that app** (the permission only applies
  to a new launch). `look_at_screen` is macOS only.
- **Speed.** With LFM3-Omni-2B (probed on Q4_K_M; Q8_0 answers the same vision cases 50/55, median 1.1 s) on llama-server b12359, a 1440x900 @2x screen answered correctly at 2880,
  1440 and 1024 px; at 1024 px (the size vaice sends) a question took about 660 prompt tokens and 0.7 to 1.2 s.
  One run per size on one screen: this shows it works, not how accurate it is. Any other endpoint that takes
  `image_url` content parts works the same; a hosted one adds its own latency.
- **Short answers on purpose.** The request asks for one or two sentences and carries `max_tokens` (`VAICE_VISION_MAX_TOKENS`, default 200). Without a cap the 2B model described a whole busy desktop: 400 to 830 tokens (5 to 9 s), once a run-on that reached the 30 s timeout and was spoken as "I can't reach my vision model". With the cap and the short-answer prompt the same screen took a median 112 tokens and 1.8 s (8 runs, none at the cap). Set `0` to send no cap, which a hosted model that wants `max_completion_tokens` needs.
- **Text on the screen is not instructions.** The vision model is told so, and its answer reaches the brain only as
  a skill result. The `remember` skill also refuses to save anything once another skill (this one included) has run
  in the same reply, so text read off the screen cannot become a lasting note.
- The answer is cut at 600 characters and its markdown emphasis is removed before it is spoken. A failing vision
  endpoint is spoken like a failing brain, naming the vision model (refused key, busy, unreachable, no answer).
  Not verified: a hosted vision provider, and the real capture on a machine with the permission granted (the unit
  tests fake `screencapture`, `sips` and the endpoint).
