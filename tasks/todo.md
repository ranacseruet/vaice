# Audit fixes (small, in-repo)

From the full codebase audit. Fixed here: the findings that are small edits to existing files.
Filed as issues instead: barge-in stale chunk (fastrtc race) #2, page reload refused while the old
connection lingers #3, no lockfile #4, shared MLX-thread wrapper #5, CI + remaining test gaps #6.

- [x] README: a second connection is refused (fastrtc `concurrency_limit=1`), not queued; MLX-thread note matches reality
- [x] llm: surface llama-server's error body on HTTP errors; raise on an `error` payload mid-stream
- [x] factory: `auto` fallback logs the traceback; cascade warns at startup if llama-server is unreachable
- [x] native: a reply cancelled before it starts skips prefill; make the test able to tell
- [x] sentences: don't split after "Dr." / "e.g." / list numbers; split on newlines; strip bullets, links, `~~`
- [x] config: bad numeric env var → clean error, not a traceback
- [x] Tests for each (fail without the fix), full suite, real cascade check against llama-server
- [x] CLAUDE.md (from /init, corrected by the audit)

## Review
- 41 tests pass. Against main's `src/`: 8 behavior tests fail (mid-stream error, HTTP reason, fallback traceback,
  cancelled prefill, 3 splitter/cleanup, plus `_number` rejecting bad input); `check_server` and `_number` defaults
  fail only because those functions are new; the stream happy path and "I am 30." guard pass on both.
- Real checks: llama-server 400 now shows `"Failed to load image or audio file"`; a spoken question (Kokoro) → Gemma
  numbered list → 4 clean sentences → 26 s of Kokoro audio; auto with a bogus native model logs the traceback and falls
  back; unreachable llama-server warns at startup; `VAICE_PORT=78x0 vaice` exits 1 with one line; native still loads,
  TTFA 0.33 s.
- Mid-stream `error` payload: llama-server's server library only emits `data: ` / `event: ` SSE prefixes, so errors
  would arrive as a `data:` JSON object; the exact shape is not reproduced (handled generically: any `error` key raises).
- Not verified: live browser mic.
