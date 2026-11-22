## Summary

<!-- Briefly describe what this PR changes and why. Link any related issue (e.g., Closes #N). -->

## Configuration & Migration

<!-- Note any new, changed, or removed VAICE_* environment variables or dependency bounds, or write "None". -->
- None

## Verification

- [ ] `.venv/bin/ruff check src tests` passes
- [ ] `.venv/bin/pytest --cov=vaice` passes
- [ ] `docs/` (and `README.md` if the quickstart or features changed) and `.env.example` updated if behavior or configuration changed
- [ ] Architecture views (`docs/architecture.md` diagrams and `docs/architecture-explorer.html`) updated if a module, a named symbol or thread, a hand-off between threads, a state or a flow step changed
- [ ] *(If touching audio/WebSocket/UI paths)* Verified manually with `llama-server` and browser microphone
