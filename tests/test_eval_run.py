"""The eval harness's assistant path: argument handling, the stub skills and run_assistant, with no server."""
import json

import pytest
import run as eval_run

from vaice.brain import BrainReply, ToolCall
from vaice.skills import default_skills


def test_the_default_paths_leave_the_assistant_path_out():
    args = eval_run.parse_args([])
    assert args.paths == ["interleaved", "text"]
    assert args.cases_file == "cases.json"


def test_the_assistant_path_alone_uses_its_own_cases():
    args = eval_run.parse_args(["--paths", "assistant"])
    assert args.paths == ["assistant"]
    assert args.cases_file == "assistant_cases.json"


def test_a_cases_file_wins_and_a_mixed_run_keeps_the_default_file():
    assert eval_run.parse_args(["--paths", "assistant", "--cases-file", "mine.json"]).cases_file == "mine.json"
    assert eval_run.parse_args(["--paths", "text,assistant"]).cases_file == "cases.json"


def test_an_unknown_path_names_the_choices(capsys):
    with pytest.raises(SystemExit):
        eval_run.parse_args(["--paths", "voice"])
    assert "interleaved, text, assistant" in capsys.readouterr().err


def test_the_stub_skills_offer_the_real_definitions_and_run_nothing(monkeypatch):
    def boom(*_args, **_kwargs):
        raise AssertionError("a stub skill ran something on this machine")

    monkeypatch.setattr("subprocess.run", boom)
    stubs, real = eval_run.stub_skills(), default_skills()

    assert stubs.definitions() == real.definitions()
    assert set(eval_run.SKILL_STUBS) == set(real.names)
    for name in real.names:
        arguments = json.dumps({"name": "report"} if name == "find_files" else {})
        assert stubs.run(name, arguments).speech == eval_run.SKILL_STUBS[name]


class FakeBrain:
    def __init__(self, *replies):
        self.replies = list(replies)

    def complete(self, messages, tools):
        return self.replies.pop(0)


def ask(name, arguments="{}"):
    return BrainReply("", [ToolCall("call_0", name, arguments)], {"role": "assistant", "content": ""})


def say(text):
    return BrainReply(text, [], {"role": "assistant", "content": text})


def test_run_assistant_records_the_calls_the_transcript_and_the_answer(monkeypatch):
    monkeypatch.setattr(eval_run, "_transcribe", lambda url, timeout, audio: "How much disk space?")
    brain = FakeBrain(ask("disk_space"), say("About 160 gigabytes."))

    record = eval_run.run_assistant("http://x", "prompt", eval_run.stub_skills(), object(), [], brain=brain)

    assert record["error"] is None
    assert record["calls"] == [["disk_space", "{}"]]
    assert record["transcript"] == "How much disk space?"
    assert record["answer"] == "About 160 gigabytes."
    assert record["latency_s"] >= record["transcribe_s"] >= 0
    assert len(record["brain_s"]) == 2  # one entry per brain round
    assert record["decide_s"] >= 0


def test_run_assistant_asks_nothing_of_the_brain_when_nothing_was_heard(monkeypatch):
    monkeypatch.setattr(eval_run, "_transcribe", lambda url, timeout, audio: "")
    brain = FakeBrain()  # any call would raise IndexError

    record = eval_run.run_assistant("http://x", "prompt", eval_run.stub_skills(), object(), [], brain=brain)

    assert record["calls"] == []
    assert record["answer"] == eval_run.NOT_HEARD
    assert record["error"] is None


def test_run_assistant_records_a_failure_instead_of_raising(monkeypatch):
    def broken(url, timeout, audio):
        raise ConnectionError("server gone")

    monkeypatch.setattr(eval_run, "_transcribe", broken)

    record = eval_run.run_assistant("http://x", "prompt", eval_run.stub_skills(), object(), [], brain=FakeBrain())

    assert record["error"] == "ConnectionError: server gone"
    assert record["latency_s"] is None
