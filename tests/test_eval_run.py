"""The eval harness's assistant path: argument handling, the stub skills and run_assistant, with no server."""
import json

import pytest
import run as eval_run

from vaice.brain import BrainReply, ToolCall
from vaice.skills import Question, default_skills


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
    assert set(eval_run.SKILL_STUBS) | set(eval_run.ACTION_STUBS) == set(real.names)
    assert not set(eval_run.SKILL_STUBS) & set(eval_run.ACTION_STUBS)
    for skill in real.skills:
        arguments = json.dumps(dict.fromkeys(skill.parameters.get("required", []), "report"))
        speech = stubs.run(skill.name, arguments).speech
        if skill.name in eval_run.SKILL_STUBS:
            assert speech == eval_run.SKILL_STUBS[skill.name]
        else:
            assert speech == eval_run.ACTION_STUBS[skill.name](json.loads(arguments))


def test_the_shell_skill_is_offered_only_with_the_flag_and_its_stub_never_runs_a_command(monkeypatch):
    def boom(*_args, **_kwargs):
        raise AssertionError("the stub started a process")

    monkeypatch.setattr("subprocess.Popen", boom)
    monkeypatch.setattr("subprocess.run", boom)

    assert "run_shell" not in eval_run.stub_skills().names
    assert eval_run.stub_skills(shell=False).definitions() == default_skills().definitions()
    stubs = eval_run.stub_skills(shell=True)
    assert stubs.definitions() == default_skills(shell=True).definitions()
    assert stubs.names[-1] == "run_shell"
    (skill,) = [s for s in stubs.skills if s.name == "run_shell"]
    assert skill.run_stoppable is None  # the real runner cannot be reached

    asked = stubs.prepare("run_shell", '{"command": "rm -rf /"}')
    assert asked == Question("Shall I run run shell?")  # always asks, never looks at the command
    assert stubs.run("run_shell", '{"command": "rm -rf /"}', lambda: False).speech == "the command finished. Output: ok"


def test_the_shell_flag_is_off_by_default_and_asks_for_the_stub_skill():
    assert eval_run.parse_args(["--paths", "assistant"]).shell is False
    assert eval_run.parse_args(["--paths", "assistant", "--shell"]).shell is True


def test_a_shell_request_is_counted_as_the_call_though_nothing_runs(monkeypatch):
    monkeypatch.setattr(eval_run, "_transcribe", lambda url, timeout, audio: "Run ls in my Downloads folder.")
    brain = FakeBrain(ask("run_shell", '{"command": "ls ~/Downloads"}'))

    record = eval_run.run_assistant("http://x", "prompt", eval_run.stub_skills(shell=True), object(), [], brain=brain)

    assert record["calls"] == [["run_shell", '{"command": "ls ~/Downloads"}']]
    assert eval_run.scoring.score({"tools": ["run_shell"]}, [tuple(c) for c in record["calls"]])["verdict"] == "correct"


def test_the_action_stubs_answer_from_their_own_arguments():
    stubs = eval_run.stub_skills()

    assert stubs.run("open_app", '{"name": "Calculator"}').speech == "opened Calculator"
    assert stubs.run("quit_app", '{"name": "Mail"}').speech == "quit Mail"
    assert stubs.run("set_volume", '{"level": 20}').speech == "volume set to 20 percent"
    assert stubs.run("mute_sound", '{"muted": false}').speech == "sound unmuted"
    assert stubs.run("mute_sound", "{}").speech == "sound muted"
    assert stubs.run("volume_up", "{}").speech == "volume set to 60 percent"
    assert stubs.run("volume_down", "{}").speech == "volume set to 40 percent"


def test_a_stub_confirm_skill_always_asks_without_looking_at_the_machine(monkeypatch):
    def boom(*_args, **_kwargs):
        raise AssertionError("the stub looked at this machine")

    monkeypatch.setattr("subprocess.run", boom)

    asked = eval_run.stub_skills().prepare("quit_app", '{"name": "NoSuchApp"}')

    assert isinstance(asked, Question)  # the real ask would say the app is not running


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


def test_run_assistant_counts_a_confirm_skill_as_the_call_though_it_never_runs(monkeypatch):
    monkeypatch.setattr(eval_run, "_transcribe", lambda url, timeout, audio: "Quit Safari.")
    brain = FakeBrain(ask("quit_app", '{"name": "Safari"}'))  # a second brain call would raise IndexError

    record = eval_run.run_assistant("http://x", "prompt", eval_run.stub_skills(), object(), [], brain=brain)

    assert record["error"] is None
    assert record["calls"] == [["quit_app", '{"name": "Safari"}']]
    assert record["answer"] == "Shall I run quit app?"
    score = eval_run.scoring.score({"tools": ["quit_app"], "args": {"quit_app": {"name": "Safari"}}},
                                   [tuple(c) for c in record["calls"]])
    assert score["verdict"] == "correct"


def test_calls_before_a_confirm_skill_are_kept_and_the_confirm_skill_comes_last(monkeypatch):
    monkeypatch.setattr(eval_run, "_transcribe", lambda url, timeout, audio: "Space, then quit Safari.")
    first = BrainReply("", [ToolCall("c0", "disk_space", "{}"), ToolCall("c1", "quit_app", '{"name": "Safari"}')],
                       {"role": "assistant", "content": "", "tool_calls": []})

    record = eval_run.run_assistant("http://x", "prompt", eval_run.stub_skills(), object(), [],
                                    brain=FakeBrain(first))

    assert record["calls"] == [["disk_space", "{}"], ["quit_app", '{"name": "Safari"}']]


def test_run_assistant_asks_nothing_of_the_brain_when_nothing_was_heard(monkeypatch):
    monkeypatch.setattr(eval_run, "_transcribe", lambda url, timeout, audio: "")
    brain = FakeBrain()  # any call would raise IndexError

    record = eval_run.run_assistant("http://x", "prompt", eval_run.stub_skills(), object(), [], brain=brain)

    assert record["calls"] == []
    assert record["answer"] == eval_run.NOT_HEARD
    assert record["error"] is None


def test_the_system_info_stub_has_a_neutral_hostname():
    assert eval_run.SKILL_STUBS["system_info"] == (
        "macOS 15.6 on Example-MacBook, up 3 days and 4 hours, 48 gigabytes of memory")


def test_the_brain_options_default_to_the_configuration_and_the_url_is_validated(monkeypatch, capsys):
    monkeypatch.setattr(eval_run.config, "BRAIN_URL", "")
    monkeypatch.setattr(eval_run.config, "BRAIN_MODEL", "")
    args = eval_run.parse_args(["--paths", "assistant"])
    assert (args.brain_url, args.brain_model) == ("", "")
    assert eval_run.make_brain(args) is None  # the local llama-server

    args = eval_run.parse_args(["--paths", "assistant", "--brain-url", "http://localhost:1234/v1/",
                                "--brain-model", "lfm2.5-24b-a2b"])
    assert (args.brain_url, args.brain_model) == ("http://localhost:1234/v1", "lfm2.5-24b-a2b")
    assert eval_run.make_brain(args) is not None

    with pytest.raises(SystemExit):
        eval_run.parse_args(["--paths", "assistant", "--brain-url", "http://host/v1/chat/completions"])
    assert "--brain-url" in capsys.readouterr().err

    monkeypatch.setattr(eval_run.config, "BRAIN_URL", "https://example.test/v1")
    monkeypatch.setattr(eval_run.config, "BRAIN_MODEL", "m")
    assert eval_run.parse_args(["--paths", "assistant"]).brain_url == "https://example.test/v1"  # from the config


def test_there_is_no_flag_for_the_key():
    with pytest.raises(SystemExit):
        eval_run.parse_args(["--paths", "assistant", "--brain-api-key", "fake-key-123"])


def run_main(monkeypatch, tmp_path, *argv, answer="About 160 gigabytes.", calls=(("disk_space", "{}"),)):
    """eval_run.main with no server, `say` or model: every assistant run is the canned record below."""
    monkeypatch.setattr(eval_run, "server_info", lambda url: {"url": url})
    monkeypatch.setattr(eval_run, "audio_for", lambda case, scratch: (16000, None))

    def fake_run(url, prompt, skills, audio, history, brain=None):
        fake_run.brains.append(brain)
        return {"error": None, "calls": [list(c) for c in calls], "transcript": "x", "answer": answer,
                "pending": False, "latency_s": 0.5}

    fake_run.brains = []
    monkeypatch.setattr(eval_run, "run_assistant", fake_run)
    out = tmp_path / "out.json"
    eval_run.main(["--paths", "assistant", "--repeats", "2", "--cases", "disk-free,chat-goodbye", "--out", str(out),
                   *argv])
    return fake_run.brains, json.loads(out.read_text()), out.read_text()


def test_a_helper_run_says_which_brain_in_the_report_and_records_only_host_and_model(monkeypatch, tmp_path, capsys):
    key = "fake-key-do-not-write-1a2b"
    monkeypatch.setattr(eval_run.config, "BRAIN_API_KEY", key)

    brains, document, raw = run_main(monkeypatch, tmp_path, "--brain-url", "https://openrouter.ai/api/v1",
                                     "--brain-model", "some/model")

    shown = capsys.readouterr()
    assert all(brain is not None for brain in brains)  # the helper is passed to every run
    assert "assistant brain: openrouter.ai, model some/model" in shown.out
    assert "== assistant: 4 runs, brain: openrouter.ai, model some/model ==" in shown.out
    assert "sent to that endpoint" in shown.out
    assert document["config"]["brain"] == {"host": "openrouter.ai", "model": "some/model"}
    for text in (raw, shown.out, shown.err):
        assert key not in text
    assert "/api/v1" not in raw  # not even the URL's path


def test_a_helper_on_this_machine_gets_no_privacy_note(monkeypatch, tmp_path, capsys):
    run_main(monkeypatch, tmp_path, "--brain-url", "http://localhost:1234/v1", "--brain-model", "m")

    out = capsys.readouterr().out
    assert "assistant brain: localhost:1234, model m" in out
    assert "sent to that endpoint" not in out


def test_a_local_run_says_so_and_passes_no_brain(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(eval_run.config, "BRAIN_URL", "")

    brains, document, _ = run_main(monkeypatch, tmp_path)

    assert brains == [None] * 4
    assert "assistant brain: the local llama-server" in capsys.readouterr().out
    assert document["config"]["brain"] == {"host": None, "model": None}


def test_a_run_without_the_assistant_path_records_no_brain(monkeypatch, tmp_path):
    monkeypatch.setattr(eval_run, "server_info", lambda url: {"url": url})
    monkeypatch.setattr(eval_run, "audio_for", lambda case, scratch: (16000, None))
    monkeypatch.setattr(eval_run, "run_text", lambda *args: {"error": None, "calls": [], "transcript": "x",
                                                            "latency_s": 0.1})
    out = tmp_path / "out.json"

    eval_run.main(["--paths", "text", "--repeats", "1", "--cases", "know-sky", "--out", str(out)])

    assert "brain" not in json.loads(out.read_text())["config"]


def test_the_summary_and_the_table_report_answer_and_command_checks_apart_from_the_verdict(
        monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(eval_run.config, "BRAIN_URL", "")

    _, document, _ = run_main(monkeypatch, tmp_path, answer="Plenty of room.")  # right call, 160 not in the answer

    out = capsys.readouterr().out
    assert "  correct" in out and "answer faithful 0/2" in out  # two disk-free runs, neither says 160
    assert "command ok" not in out  # no case asked for it
    table = out.split("case ", 1)[1]
    assert "answer" in table.splitlines()[0] and "command" not in table.splitlines()[0]
    (disk_row,) = [line for line in table.splitlines() if line.startswith("disk-free")]
    assert disk_row.split() == ["disk-free", "2/2", "0/2"]  # the call verdict is right, the answer is not
    (goodbye_row,) = [line for line in table.splitlines() if line.startswith("chat-goodbye")]
    assert goodbye_row.split() == ["chat-goodbye", "0/2", "-"]
    first = document["runs"][0]["score"]
    assert first["verdict"] == "correct" and first["answer_ok"] is False and first["command_ok"] is None
    assert document["summary"]["assistant"]["answer_faithful"] == {"ok": 0, "of": 2}


def test_the_command_check_is_reported_for_the_shell_cases(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(eval_run.config, "BRAIN_URL", "")
    monkeypatch.setattr(eval_run, "server_info", lambda url: {"url": url})
    monkeypatch.setattr(eval_run, "audio_for", lambda case, scratch: (16000, None))
    commands = iter(["ls ~/Downloads", "ls /Users/me/Downloads"])
    monkeypatch.setattr(eval_run, "run_assistant", lambda *args: {
        "error": None, "calls": [["run_shell", json.dumps({"command": next(commands)})]], "transcript": "x",
        "answer": "Shall I run ls?", "pending": True, "latency_s": 0.1})

    eval_run.main(["--paths", "assistant", "--shell", "--cases-file", "assistant_shell_cases.json", "--repeats", "2",
                   "--cases", "shell-ls-downloads", "--out", str(tmp_path / "out.json")])

    out = capsys.readouterr().out
    assert "command ok 1/2" in out and "answer faithful" not in out
    (row,) = [line for line in out.splitlines() if line.startswith("shell-ls-downloads")]
    assert row.split() == ["shell-ls-downloads", "2/2", "1/2"]  # the verdict is right twice, the command once


def test_run_assistant_records_a_failure_instead_of_raising(monkeypatch):
    def broken(url, timeout, audio):
        raise ConnectionError("server gone")

    monkeypatch.setattr(eval_run, "_transcribe", broken)

    record = eval_run.run_assistant("http://x", "prompt", eval_run.stub_skills(), object(), [], brain=FakeBrain())

    assert record["error"] == "ConnectionError: server gone"
    assert record["latency_s"] is None
