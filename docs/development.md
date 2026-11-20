# Development


## Setup & Dependency Updates

```bash
uv sync --extra dev      # creates .venv from uv.lock: the exact versions CI tests (add --extra mcp for the MCP tools and their tests, --extra browse for the browser agent and its tests)
```

`uv sync` installs from the committed `uv.lock`, so every developer and CI run gets the same dependency versions (`.python-version` selects Python 3.12). A plain `uv pip install -e ".[dev]"` or `pip install -e ".[dev]"` also works and resolves against the version bounds in `pyproject.toml`.

To upgrade dependencies deliberately:

```bash
uv lock --upgrade-package fastrtc   # or --upgrade to refresh everything
uv sync --extra dev
.venv/bin/pytest && .venv/bin/ruff check src tests
```

Known compatibility bounds in `pyproject.toml`:
- **`fastrtc[vad]>=0.0.34,<0.1`** — `interruptible.py` and `tests/test_interruptible.py` rely on `fastrtc` 0.0.34's `ReplyOnPause` internals and verify `ReplyOnPause.__init__`'s signature on every test run.
- **`gradio>=5.50,<6`** — imported directly by `app.py`; capped below 6.0 to match `fastrtc` 0.0.34 and keep `gr.Chatbot(allow_tags=False)` behavior consistent.

## Tests & Code Quality

```bash
.venv/bin/pytest                                         # run all unit tests
.venv/bin/pytest --cov=vaice --cov-report=term-missing   # run with line coverage
.venv/bin/ruff check src tests evals                     # run linter
```

CI (`.github/workflows/ci.yml`) runs Ruff and `pytest --cov=vaice` with `uv sync --locked --extra dev --extra mcp --extra browse` on Python 3.12 on macOS for every push to `main` and every pull request; a newer push to a pull request cancels the older run. It can also be started by hand (`gh workflow run ci.yml --ref main`), for a push whose run never started.

The unit tests never touch a real model. To measure how reliably spoken commands turn into the right tool call (with repeats), run the live eval in [`evals/`](../evals/README.md) against a real `llama-server`: `.venv/bin/python evals/run.py`. It is run by hand, not in CI.

See [CONTRIBUTING.md](../CONTRIBUTING.md) for contributor guidelines and how the fake `/v1/realtime` server and multi-threaded `fastrtc` test harnesses work.
