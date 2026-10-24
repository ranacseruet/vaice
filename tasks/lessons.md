# Lessons

## 2026-10-23: MLX "no Stream in current thread" in the live app
- **What went wrong:** native backend failed after the first audio chunk in the real UI, though my tests passed.
- **Why:** MLX streams are bound to the creating thread; fastrtc resumes a handler generator via `next()` on whichever worker thread is free for each chunk. My VAD harness called `emit()` from one thread, so it never exercised that. I had even flagged the thread risk in the design and "verified" it with a single-thread test.
- **Rule:** when verifying code that runs inside a framework's executor, reproduce the framework's real threading (fresh thread per resume), and confirm the harness fails against the known-bad code before trusting it passing against the fix. All MLX work in this repo goes through the single worker in `NativeResponder`.
