import logging

import numpy as np
import pytest

from vaice import app
from vaice.app import make_handler, parse_args


class FakeResponder:
    name = "fake"

    def respond(self, audio):
        yield (24000, np.zeros(1, dtype=np.float32))


def test_each_reply_logs_which_backend_answered(caplog):
    with caplog.at_level(logging.INFO, logger="vaice.app"):
        chunks = list(make_handler(FakeResponder())((48000, np.zeros((1, 480), dtype=np.int16))))

    assert len(chunks) == 1
    assert "[fake] time to first audio" in caplog.text
    assert "[fake] reply finished" in caplog.text


def test_backend_flag_overrides_env_default(monkeypatch):
    monkeypatch.setattr(app.config, "BACKEND", "native")
    assert parse_args([]).backend == "native"
    assert parse_args(["--backend", "cascade"]).backend == "cascade"


def test_invalid_backend_from_env_is_a_usage_error_not_a_traceback(monkeypatch, capsys):
    monkeypatch.setattr(app.config, "BACKEND", "Native")
    with pytest.raises(SystemExit) as exc:
        parse_args([])
    assert exc.value.code == 2
    assert "VAICE_BACKEND" in capsys.readouterr().err
