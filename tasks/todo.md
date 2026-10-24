# Conversation history, context management and memory (research + plan)

Status: phase 1 done on `feat/conversation-history` (see Review); phases 2-4 not started.

## Phase 1 checklist
- [x] `conversation.py`: `Turn` (thread-safe; closed turns ignore late assistant-text updates), `Conversation` (start_turn closes + marks the previous unfinished turn interrupted; windowed history in blocks so the cached prefix stays stable; waits briefly for a pending transcript; JSON persistence with atomic write; corrupt file moved aside; reset)
- [x] Responder seam: `respond(audio, history, turn)`; responders record spoken text and the user transcript on `turn`
- [x] cascade: ~~history in llm messages; Gemma transcript in parallel~~ → transcribe first (greedy), then text-only chat with history (user decision after measurement, see Review); sentence-level spoken text
- [x] native: history as text turns in ChatState; generation stops when the turn closes; text up to the last handed-out chunk; LFM ASR on the worker after generation
- [x] app: Conversation wiring, transcript chatbot (AdditionalOutputs), "New conversation" button, history on page load
- [x] config + README (`VAICE_DATA_DIR`, `VAICE_HISTORY_TURNS`), CLAUDE.md
- [x] Tests (barge-in harness with fresh thread per resume, shown failing on a naive commit-in-finally version first); full suite
- [x] Real checks: 10-turn scripted conversation per backend (TTFA turn 1 vs 10, llama-server cache hits), browser UI (chatbot, reset, reload keeps history)

## Research summary (2026-10-24)

| Option | Verdict | Why |
|---|---|---|
| Pipecat 1.14 | Skip; borrow its pattern | Needs Python ≥3.11 (repo floor is 3.10) and would replace fastrtc. Neither backend fits its services: no LFM speech-to-speech service, and raw audio → llama.cpp would be a custom processor anyway. Borrow: its context aggregator records *what was spoken*, not what was generated. |
| LiveKit Agents 1.8 | Skip | Built around a LiveKit server (hypothesis: not verified this run); against single-process local. |
| mem0 2.2 | Skip for now | Defaults to OpenAI LLM + embeddings + Qdrant; an LLM extraction call per write; local llama.cpp config keys unverified. |
| Letta / Zep / LangMem | Skip | Server/graph stacks, far heavier than a single-user demo needs (not researched in depth). |
| sqlite-vec + mlx-embeddings | Later, only if needed | Lightest vector path. sqlite-vec is pre-v1 (pin it). Not needed until facts outgrow the prompt. |
| fastrtc 0.0.34 | Keep | No history helpers, last release Nov 2025 (quiet). Has `AdditionalOutputs` (push transcript to UI) and `get_current_context().webrtc_id`. |

Conclusion: build it in-house (~300-500 lines), no new framework. mlx-audio already ships STT models (parakeet, whisper, moonshine) if needed.

### Verified facts the design rests on
- **mlx-audio 0.6.1 LFM:** `ChatState` is multi-turn, but `generate_from_chat_state` never passes `audio_out_codes`, so prior *assistant audio* can't be replayed; it also builds a fresh KV cache per call (full prefill every turn). → Native history = text turns + the current user clip only. Stacking several user clips in one `ChatState` is also avoided (its per-clip frame count vs. joint conformer subsampling may disagree).
- **LFM2.5-Audio context:** `max_position_embeddings` 128000 (HF config).
- **llama-server (local, build 11669):** 4 slots, n_ctx 262144, audio modality on. Probe (`say` clip, Gemma 12B):
  - turn 1 with audio: prompt 62 tok, 627 ms
  - turn 2 same slot, prior audio in history: `cache_n` 57, only 20 new tokens, 201 ms → **prefix cache works across audio parts when `id_slot` is pinned**
  - Gemma transcription on another slot: exact transcript, 0.74 s
- fastrtc barge-in: the dropped generator is finalized later on another thread (see CLAUDE.md), so "turn finished" can arrive after the next turn started.

## Design

- `Conversation` (new `conversation.py`): ordered turns `{role, text, interrupted}` + optional rolling summary; a lock and a turn sequence number so a late-finalized interrupted reply can't land out of order. Owned by the app layer, one per conversation.
- `Responder.respond(audio, history)`: responders get the past turns (text) and report back what they produced. Assistant text is committed only for audio actually yielded: cascade = sentences fully spoken; native = interleaved text tokens up to the last yielded audio chunk. Interrupted turns are marked `(interrupted)`. Approximate until #2 (yielded ≠ played) is fixed.
- **User transcript** (needed for text history, UI, summary, memory): produced *after* the reply streams, off the critical path.
  - cascade: ~~in parallel on a separate slot~~ superseded: Gemma transcribes first (greedy), the reply is a text chat. No slot pinning; llama-server's own slot selection is used.
  - native: LFM ASR mode ("Perform ASR.", sequential) on the existing MLX worker after generation.
- **Context budget:** system prompt + summary (stable prefix, cache-friendly) + last turns trimmed oldest-first to `VAICE_HISTORY_TURNS` / a token budget.
- UI: show the running transcript (fastrtc `AdditionalOutputs` → Gradio chatbot) and a "New conversation" control.

## Phases (each its own PR)

1. **Multi-turn history** — `Conversation`, responder seam change, transcripts, barge-in-safe commit, trimming, persistence to `~/.vaice` (JSON or SQLite), UI transcript + "New conversation". Tests: history threading, interrupted-turn commit order under fastrtc-style threading (fresh thread per resume; must fail on a naive version first), trimming.
2. **Rolling summary** — when trimmed turns fall out of the window, summarize them in the background (cascade: Gemma on a spare slot) into the stable prefix.
3. **Long-term memory** — small facts list in SQLite (`~/.vaice/`), extracted after each turn in the background, injected into the system prompt; voice commands like "forget that" later.
4. **Retrieval** (only if facts outgrow the prompt) — sqlite-vec + a small MLX embedding model.

## Verification per phase
- Unit tests with fakes (existing pattern); full suite.
- Real check on each backend: 10-turn spoken conversation, follow-up questions that need earlier context ("and what about Germany?"), a barge-in mid-reply; compare `time to first audio` at turn 1 vs turn 10.
- README: env-var table (`VAICE_HISTORY_TURNS`, memory path/toggle, etc.), layout, "no conversation history" line in CLAUDE.md.

## Related open issues
- #5 (one MLX worker for every MLX responder): becomes a prerequisite if the cascade ever transcribes with an MLX STT model; not needed with the Gemma-transcript route.
- #2 (barge-in stale chunk): limits how exactly "what was spoken" can be recorded.

## Decisions (user, 2026-10-24)
- **Lifetime:** persisted to `~/.vaice` (survives reloads and restarts) until "New conversation" is pressed. Persistence moves into phase 1.
- **First PR:** phase 1 only (history + transcripts + barge-in-safe commit + trimming + persistence + UI transcript/reset).
- **Summary/memory model:** llama-server's Gemma on both backends when reachable; when it isn't, native keeps only the plain history window.

- **Cascade (2026-10-24, after measurement):** transcribe-first. Gemma with text history + the new question as audio
  often misheard the clip or answered an earlier question; transcribe-first fixed it for ~0.3-0.5 s more TTFA.

## Review (phase 1)
- 105 tests pass. The barge-in harness (`tests/test_app.py`, fresh thread per resume, late close from another thread)
  failed against a naive "write history in `finally:`" variant, then passed against the real code.
- **Native, 10 spoken turns (`say` clips, real model):** every transcript was exact. Short-range follow-ups used
  history ("And what about Germany?" → Berlin, "Multiply that by ten" → 40, "How many hearts…" → 3). The model did
  not recall the name or city from turn 1 at turns 9-10, though both were in the prompt (1.5B limit). TTFA went
  from 0.33 s to 0.44 s (full prefill each turn). Turn 10 once hit the 512-token cap with 39 s of audio; it did not
  reproduce in 10 more runs (5 with history, 5 without).
- **Cascade, audio-direct + history (first version):** the context questions mostly failed in 3 runs ("didn't catch
  that", answers to earlier questions). Isolated probes: with greedy decoding, the parallel transcription is identical
  to the sequential one (not the cause); with history in the prompt, Gemma mishears the new clip (q5 at 16 kHz with
  history: fine; "what is my name" with history: 0/4 vs 2/2 for text).
- **Cascade, transcribe-first (shipped):** turns 3, 5, 6, 7 and 10 were right ("Your name is Sam! You mentioned that
  at the start"). Turns 8-9 failed only because Gemma misheard the synthetic voice. TTFA was 1.35-2.24 s for turns
  2-10 and ≈5 s on turn 1 (seen in every arm, including audio-direct).
- **Browser (headless Chromium, fake mic file, real WebRTC):** the transcript updates after each reply, survives a
  reload, and both "New conversation" and the chatbot's trash icon clear the stored file.
  - Reloading mid-stream logs `AbortError: BodyStreamBuffer was aborted` / `Connection errored out` in the console.
    `main` doesn't (hypothesis: the additional-outputs stream is cut by the navigation); history reloads fine.
  - With a chatbot, fastrtc uses its component layout, so after granting mic access you also click Record
    (`main` starts at once).
- **Audio-rate anomaly (not acted on):** Gemma transcribed 3 of 10 `say` clips correctly when they were converted
  straight to 16 kHz by afconvert, but not from the 48 kHz files, and not from those resampled to 16 kHz with
  scipy. The cause is unclear; no change made.
- Not verified: a real human voice through a real microphone.

## Review (#3 reload refused, #9 reload console errors)
Measured in headless Chromium (fake mic file, real WebRTC), native backend, fastrtc with DEBUG logging; waits are 3-15 s
after connecting unless noted.
- **#3 refuted for reload.** Reload while connected: the old peer connection logged `connectionState closed` 10 ms after the
  navigation (Chrome closes it), and the reloaded page's offer was accepted 2.1-2.6 s after the reload started (that is the
  reload itself plus the script's 1.5 s mic click; no `concurrency_limit_reached`), 19 of 19 reloads with the chatbot app and 12 of 12 without (idle, mid-reply, 15 s in).
- **The real residual case is an abrupt disconnect, not a reload.** When the page's peer connection is not closed (the tab
  closed through Playwright, or the browser closed), the server holds the slot until aioice gives up on the consent checks (6
  failures x ~5 s). A fresh page retrying every ~4.3 s was first accepted at: +28.7 s (`page.close()`), +32.9 s (`page.close()`,
  repeat), +28.6 s (`page.close({runBeforeUnload: true})`), +33.1 s (whole browser closed). Server-log timestamps of
  three earlier accidental orphans (browser closed at the end of a script) show the connection going to `closed` 28.7, 30.5 and
  ~30 s later. These are headless Chromium via Playwright; a real headed Chrome tab close was not measured.
- **Raw refusals in the early scratch logs are not reload failures.** The first detector (`/webrtc/offer`) was wrong (Gradio posts offers to
  `/gradio_api/component_server/`), so the early `REFUSED` lines were 15 s timeouts; and a `wait8` run was refused for 23.7 s after its
  reload only because the previous script's `browser.close()` had left an orphan connection (its page never got its own connection).
  All numbers above come from runs that started with a free slot and ended with `about:blank` first so the connection closed cleanly.
- **#9 confirmed as a harmless client-side log.** `AbortError: BodyStreamBuffer was aborted` + `Unexpected error Connection
  errored out.` appear in the old document ~1 ms after the reload starts, from Gradio's JS client: the only request in flight
  was `GET /gradio_api/queue/data` (the SSE event stream) and the client logs `console.error("Unexpected error", ...)` for a
  `broken_connection` message. With the chatbot/`additional_outputs`: 10 of 17 reloads logged it (5/5 and 1/2 reloading 3 s after
  connecting, 3/3 at 15 s, 1/1 at 10 s; 0/6 when reloading just as a reply started, not explained). Same app without the chatbot (scratch
  launcher, handler's `AdditionalOutputs` dropped): 0 of 12 reloads. Server side: no error beyond fastrtc DEBUG
  "Exception sending msg" on the closed data channel. After the reload the stored conversation is shown and a new connection works and the transcript keeps updating
  (fresh data dir: 1 user + 1 bot bubble before the reload, 3 + 3 eighteen seconds after reconnecting and one more exchange).
- **Decision: no code change.** Neither problem is fixable in vaice without a product choice or hacking Gradio's client. Options
  for the residual #3 case, if wanted: (a) `concurrency_limit=2` (or `None`): reopens one `Conversation` and one MLX worker shared
  by simultaneous sessions, and does not free the dead connection, only lets a new one in beside it; (b) evict the previous
  connection when a new offer arrives ("newest tab wins"): small, but reaches into fastrtc internals (`pcs`, `clean_up`) and
  silently kicks a live second tab; (c) shorten the wait by patching aioice's module globals `CONSENT_FAILURES` / `CONSENT_INTERVAL`:
  a library-global hack that also drops live connections sooner on a real network hiccup; (d) leave as is and document (done in
  README / CLAUDE.md). Recommendation: (d), revisit (b) if
  the 30 s wait annoys in practice. #9: close with the documentation.

## Review (issue #2: stale chunk after barge-in)
- **Confirmed**, at two levels (fastrtc 0.0.34):
  - Real `ReplyOnPause` driven with fastrtc's threading (emit on a fresh thread per chunk, receive on another): `close()` raised
    "generator already executing" (debug log) and the in-flight `emit()` returned the next chunk after `clear_queue()`.
  - Real app, headless Chromium over real WebRTC, `--backend native`, a long-answer prompt, a fake-mic file with an interrupting
    question (instrumented `emit`/`_close_generator`, run on :7861): `CLOSE ... mid-next=True`, then exactly one 0.80 s chunk
    emitted from the closed generator, ~0.14-0.28 s later. The reply's remaining chunks never came (the generator was finalized).
- **Fix:** `interruptible.py:InterruptSafeReplyOnPause`, a `ReplyOnPause` subclass whose `emit()` returns None (fastrtc skips it)
  instead of audio when `receive()` replaced `self.generator` while it ran. Same browser run with the fix: `CLOSE ... mid-next=True`,
  "dropped a chunk of the interrupted reply", `EMIT None`, then the new reply's chunks. Also overrides `copy()` (fastrtc copies the handler
  per connection and the base builds a plain `ReplyOnPause`).
- **Tests** (`tests/test_interruptible.py`, plus one each in `test_native.py`, `test_app.py`): 3 of 9 interruptible tests fail with the
  drop disabled (mid-chunk, interrupt during the very first `next()`, audio+transcript chunk); the rest guard against over-dropping
  (no interrupt, interrupt between chunks, new reply untouched, transcript-only output kept, `copy()`).
- **Not verified:** that the stale chunk is audible in a browser (measured on the server, at what `emit()` hands to fastrtc), a real
  microphone, the cascade backend (same code path; llama-server wasn't used). **Residual:** an interrupt landing between the check in
  `emit()` and fastrtc queueing the chunk still plays it (microseconds vs the whole `next()`).
- **Other fastrtc races seen in the source (not reproduced, not touched):** if the in-flight `next()` hits `StopIteration`, `emit()` calls `reset()`, which clears the
  event and state that the interrupting utterance just set. `clear_queue()` pokes an `asyncio.Queue` from a non-loop thread. Tracked in #13.
