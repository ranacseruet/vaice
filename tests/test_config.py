import pytest

from vaice import config


def test_bad_number_is_a_clean_error_not_a_traceback(monkeypatch):
    monkeypatch.setenv("VAICE_PORT", "78x0")
    with pytest.raises(SystemExit, match="VAICE_PORT must be a number"):
        config._number("VAICE_PORT", default="7860", kind=int)


def test_number_default_and_override(monkeypatch):
    monkeypatch.delenv("VAICE_SPEED", raising=False)
    assert config._number("VAICE_SPEED", default="1.1", kind=float) == 1.1
    monkeypatch.setenv("VAICE_SPEED", "0.9")
    assert config._number("VAICE_SPEED", default="1.1", kind=float) == 0.9


def test_a_bad_number_names_the_alias_that_was_set(monkeypatch):
    monkeypatch.delenv("VAICE_TTS_SPEED", raising=False)
    monkeypatch.setenv("VAICE_SPEED", "fast")
    with pytest.raises(SystemExit, match="VAICE_SPEED must be a number"):
        config._number("VAICE_TTS_SPEED", "VAICE_SPEED", default="1.1", kind=float)


def test_the_new_name_wins_over_the_old_one(monkeypatch):
    monkeypatch.setenv("VAICE_TTS_SPEED", "1.4")
    monkeypatch.setenv("VAICE_SPEED", "0.9")
    assert config._number("VAICE_TTS_SPEED", "VAICE_SPEED", default="1.1", kind=float) == 1.4
    monkeypatch.setenv("VAICE_TTS_VOICE", "am_michael")
    monkeypatch.setenv("VAICE_VOICE", "af_bella")
    assert config._env("VAICE_TTS_VOICE", "VAICE_VOICE", default="af_heart") == "am_michael"


def test_the_old_name_still_works(monkeypatch):
    monkeypatch.delenv("VAICE_TTS_VOICE", raising=False)
    monkeypatch.setenv("VAICE_VOICE", "af_bella")
    assert config._env("VAICE_TTS_VOICE", "VAICE_VOICE", default="af_heart") == "af_bella"
    monkeypatch.delenv("VAICE_TTS_SPEED", raising=False)
    monkeypatch.setenv("VAICE_SPEED", "0.9")
    assert config._number("VAICE_TTS_SPEED", "VAICE_SPEED", default="1.1", kind=float) == 0.9


def test_neither_name_gives_the_default(monkeypatch):
    monkeypatch.delenv("VAICE_LLM_PROMPT", raising=False)
    monkeypatch.delenv("VAICE_PROMPT", raising=False)
    assert config._env("VAICE_LLM_PROMPT", "VAICE_PROMPT", default="be brief") == "be brief"
    monkeypatch.setenv("VAICE_LLM_PROMPT", "")
    assert config._env("VAICE_LLM_PROMPT", "VAICE_PROMPT", default="be brief") == ""  # empty is a choice
