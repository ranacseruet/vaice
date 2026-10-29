# Contributing to vaice

Thank you for your interest in contributing to `vaice`! This document covers setting up your development environment, running the test suite and linter, understanding key architectural invariants, and submitting pull requests.

## Development Setup

### Prerequisites
- **Python 3.10–3.12** (`.python-version` defaults to `3.12`)
- **[uv](https://docs.astral.sh/uv/)**
- *(For manual end-to-end audio testing)* A local `llama.cpp` `llama-server` (**b11960+**, e.g. `brew install llama.cpp`) running an audio-output model such as `LiquidAI/LFM3-Audio-2B-GGUF:Q4_K_M`

### Install Dependencies
```bash
git clone https://github.com/ranacseruet/vaice.git
cd vaice
uv sync --extra dev
```

`uv sync --extra dev` creates `.venv` from the committed `uv.lock` so your local environment matches CI.

## Running Tests & Linting

Before opening a pull request, run both the test suite and Ruff:

```bash
.venv/bin/ruff check src tests
.venv/bin/pytest --cov=vaice --cov-report=term-missing
```

To run a single test file or filter by keyword:

```bash
.venv/bin/pytest tests/test_llamacpp.py -k history
```

### Code Style
- **Ruff** is configured in `pyproject.toml` with `line-length = 120` and a focused rule set (`E`, `F`, `B`, `I`, `UP`, `SIM`, `BLE`, `PERF`, `RET`, `RUF`, `C4`).
- Keep changes surgical and minimal. Comments and docstrings should explain *why* an invariant or workaround exists (for example, `fastrtc` threading behavior or `llama-server` protocol constraints), not merely restate the code.

## Testing Guidelines & Architectural Gotchas

Unit tests run completely offline without a real `llama-server` or microphone by using fakes (`tests/fake_realtime.py` and fake responders). When writing or modifying tests, keep these project-specific invariants in mind:

1. **Reproduce `fastrtc`'s thread-per-resume execution:**
   `fastrtc` resumes the reply generator (`next(gen)`) on whichever worker thread is free for each audio chunk. Tests that exercise generator state or sockets across chunks use a fresh thread per `next()` call (`resume()` in `tests/test_app.py` and `tests/test_llamacpp.py`).
2. **Never let an exception escape the reply generator:**
   `fastrtc` wraps per-connection `emit()` calls in `webrtc_error_handler`; an uncaught exception kills the connection's audio task until the page is reloaded. `app.py` catches reply errors inside the generator, records them on the `Turn`, and surfaces them in the UI.
3. **Daemon threads for `llama-server` WebSockets:**
   `websockets.sync.client.connect()` and `close()` can block up to their timeouts, and `fastrtc`'s executor threads are joined at process exit. Each `/v1/realtime` session therefore runs on its own daemon thread, while the reply generator polls an in-memory queue (`POLL_SECONDS`) and checks `lifecycle.stopping` and `turn.closed`.
4. **`InterruptSafeReplyOnPause` and `fastrtc` internals:**
   `src/vaice/interruptible.py` works around `fastrtc` 0.0.34 races and idle CPU polling. `tests/test_interruptible.py` tests against the real `ReplyOnPause` class and checks `ReplyOnPause.__init__`'s signature so any upstream `fastrtc` change fails loudly in tests.
5. **Manual audio verification:**
   When changing audio resampling (`audio.py`), WebSocket session handling (`responders/llamacpp.py`), or WebRTC stream setup (`app.py`, `interruptible.py`), verify the real audio path by running `vaice` against a live `llama-server` (`LiquidAI/LFM3-Audio-2B-GGUF`), speaking into the browser mic in Chrome (with headphones), and checking the `[llamacpp] time to first audio` log line.

## Keeping Documentation in Sync

- Any behavior or configuration change must update `README.md` in the same pull request.
- Adding, removing, or changing the default of a `VAICE_*` environment variable must also update `.env.example`.
- If you change `pyproject.toml` dependencies, run `uv lock` and commit the updated `uv.lock` (CI runs `uv sync --locked` and will fail if `uv.lock` is out of date).

## Submitting a Pull Request

1. Fork the repository and create a descriptive branch (`feat/...`, `fix/...`, `docs/...`, `chore/...`).
2. Make your changes and add or update tests covering the behavior.
3. Confirm `.venv/bin/ruff check src tests` and `.venv/bin/pytest` pass on Python 3.10 and 3.12.
4. Open a pull request and fill out the pull request template.

## Reporting Issues

Use the [GitHub Issue Tracker](https://github.com/ranacseruet/vaice/issues) to report bugs or propose enhancements. Include your OS, Python version, `llama-server --version` (build number), model name, and terminal logs.
