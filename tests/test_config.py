import pytest

from vaice import config


def test_bad_number_is_a_clean_error_not_a_traceback(monkeypatch):
    monkeypatch.setenv("VAICE_PORT", "78x0")
    with pytest.raises(SystemExit, match="VAICE_PORT must be a number"):
        config._number("VAICE_PORT", "7860", int)


def test_number_default_and_override(monkeypatch):
    monkeypatch.delenv("VAICE_SPEED", raising=False)
    assert config._number("VAICE_SPEED", "1.1", float) == 1.1
    monkeypatch.setenv("VAICE_SPEED", "0.9")
    assert config._number("VAICE_SPEED", "1.1", float) == 0.9
