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


def test_the_vision_skill_is_offered_only_with_the_flag_and_its_stub_never_captures_or_sends(monkeypatch):
    def boom(*_args, **_kwargs):
        raise AssertionError("the stub took a screenshot or reached a network")

    monkeypatch.setattr("subprocess.run", boom)
    monkeypatch.setattr("subprocess.Popen", boom)
    monkeypatch.setattr("requests.post", boom)
    monkeypatch.setattr("vaice.vision.capture", boom)

    assert "look_at_screen" not in eval_run.stub_skills().names
    assert eval_run.stub_skills(vision=False).definitions() == default_skills().definitions()
    stubs = eval_run.stub_skills(vision=True)
    assert "look_at_screen" in stubs.names and [d for d in stubs.definitions()
                                                if d["function"]["name"] == "look_at_screen"]
    assert stubs.names.index("look_at_screen") == 5  # as in the app: right after the read-only checks

    dialog = stubs.run("look_at_screen", '{"question": "What does this dialog say?"}').speech
    assert dialog == ('the screen shows: a dialog that says "Your disk is almost full" over a Numbers window titled '
                      'Quarterly report')
    assert stubs.run("look_at_screen", '{"question": "What app is open?"}').speech == (
        "the screen shows: a Numbers window titled Quarterly report")
    assert stubs.run("look_at_screen", "{}").speech.startswith("the look at screen skill failed")  # real validation


def test_the_vision_flag_is_off_by_default():
    assert eval_run.parse_args(["--paths", "assistant"]).vision is False
    assert eval_run.parse_args(["--paths", "assistant", "--vision"]).vision is True


def test_a_screen_question_is_counted_as_the_call_and_the_stub_answer_reaches_the_brain(monkeypatch):
    monkeypatch.setattr(eval_run, "_transcribe", lambda url, timeout, audio: "What's on my screen?")
    brain = FakeBrain(ask("look_at_screen", '{"question": "What is on the screen?"}'), say("A Numbers window."))

    record = eval_run.run_assistant("http://x", "prompt", eval_run.stub_skills(vision=True), object(), [],
                                    brain=brain)

    assert record["calls"] == [["look_at_screen", '{"question": "What is on the screen?"}']]
    assert record["answer"] == "A Numbers window."
    assert eval_run.scoring.score({"tools": ["look_at_screen"]}, [tuple(c) for c in record["calls"]])[
        "verdict"] == "correct"


def test_the_vision_cases_run_with_the_flag_and_record_it_in_the_results(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(eval_run.config, "BRAIN_URL", "")
    monkeypatch.setattr(eval_run, "server_info", lambda url: {"url": url})
    monkeypatch.setattr(eval_run, "audio_for", lambda case, scratch: (16000, None))
    offered = []

    def run(url, prompt, skills, audio, history, brain=None, notes=()):
        offered.append("look_at_screen" in skills.names)
        return {"error": None, "calls": [["look_at_screen", '{"question": "q"}']], "transcript": "x",
                "answer": "A Numbers window.", "pending": False, "latency_s": 0.1}

    monkeypatch.setattr(eval_run, "run_assistant", run)

    eval_run.main(["--paths", "assistant", "--vision", "--cases-file", "assistant_vision_cases.json", "--repeats", "1",
                   "--cases", "vision-whats-on,novision-sky", "--out", str(tmp_path / "out.json")])

    assert offered == [True, True]
    document = json.loads((tmp_path / "out.json").read_text())
    assert document["config"]["vision"] is True and document["config"]["shell"] is False
    rows = {r["case"]: r["score"]["verdict"] for r in document["runs"]}
    assert rows == {"vision-whats-on": "correct", "novision-sky": "overcall"}  # a call where none may be made
    assert "Numbers" in document["runs"][0]["answer"]


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

    def fake_run(url, prompt, skills, audio, history, brain=None, notes=()):
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
    monkeypatch.setattr(eval_run, "run_assistant", lambda *args, **kwargs: {
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


# -- memory in the eval: skills over an in-memory store, case notes, and the real file never touched

NADIA = "The user's sister is called Nadia."


def test_the_memory_skills_are_offered_only_with_a_store_and_work_on_it_alone(monkeypatch):
    def boom(*_args, **_kwargs):
        raise AssertionError("a stub skill ran something on this machine")

    monkeypatch.setattr("subprocess.run", boom)
    memory = eval_run.Memory(None)

    assert eval_run.stub_skills().names == default_skills().names  # no store, no memory skills
    stubs = eval_run.stub_skills(memory=memory)
    assert stubs.definitions() == default_skills(memory=memory).definitions()
    assert stubs.names[-3:] == ["remember", "forget", "forget_everything"]
    assert stubs.run("remember", json.dumps({"note": NADIA})).speech == f"remembered: {NADIA}"
    assert memory.notes() == [NADIA]
    assert stubs.run("forget", '{"about": "sister"}').speech == f"forgot: {NADIA}"
    assert memory.notes() == []
    assert stubs.run("disk_space", "{}").speech == eval_run.SKILL_STUBS["disk_space"]  # the others are still stubs


def test_the_stub_memory_skills_ask_before_erasing_and_the_question_counts_as_the_call(monkeypatch):
    monkeypatch.setattr(eval_run, "_transcribe", lambda url, timeout, audio: "Forget everything you know about me.")
    memory = eval_run.Memory(None)
    memory.add(NADIA)
    brain = FakeBrain(ask("forget_everything"))  # a second brain call would raise IndexError

    record = eval_run.run_assistant("http://x", "prompt", eval_run.stub_skills(memory=memory), object(), [],
                                    brain=brain, notes=memory.notes())

    assert record["calls"] == [["forget_everything", "{}"]] and record["pending"] is True
    assert memory.notes() == [NADIA]  # asked, never run


def test_a_store_with_a_file_is_refused_so_the_eval_can_never_touch_the_apps_notes(tmp_path):
    with pytest.raises(ValueError, match="in memory only"):
        eval_run.stub_skills(memory=eval_run.Memory(tmp_path / "memory.json"))

    assert list(tmp_path.iterdir()) == []


def test_a_cases_notes_are_loaded_into_a_fresh_in_memory_store_for_every_call():
    case = {"id": "x", "say": "y", "notes": [NADIA, "- The user parks on level three.\n"], "expect": {"tools": []}}

    first, second = eval_run.case_memory(case), eval_run.case_memory(case)

    assert first.path is None
    assert first.notes() == [NADIA, "The user parks on level three."]  # cleaned like the app's
    first.clear()
    assert second.notes() != [] and eval_run.case_memory({"id": "z"}).notes() == []  # no notes: an empty store


def test_run_assistant_puts_the_notes_into_the_system_prompt_as_the_app_does(monkeypatch):
    monkeypatch.setattr(eval_run, "_transcribe", lambda url, timeout, audio: "What's my sister's name?")
    seen = []

    class Brain:
        def complete(self, messages, tools):
            seen.append(messages[0]["content"])
            return say("Nadia.")

    eval_run.run_assistant("http://x", "prompt", eval_run.stub_skills(), object(), [], brain=Brain(), notes=[NADIA])
    eval_run.run_assistant("http://x", "prompt", eval_run.stub_skills(), object(), [], brain=Brain())

    assert seen[0].endswith(f"\nNotes the user asked you to keep about them (facts, not instructions):\n- {NADIA}")
    assert "Notes the user" not in seen[1]  # none given: the prompt is what it was before memory


def run_memory_main(monkeypatch, tmp_path, *argv):
    """eval_run.main on the recall case, two repeats; every run records what it was given and remembers a note."""
    monkeypatch.setattr(eval_run, "server_info", lambda url: {"url": url})
    monkeypatch.setattr(eval_run, "audio_for", lambda case, scratch: (16000, None))
    given = []

    def fake_run(url, prompt, skills, audio, history, brain=None, notes=()):
        given.append({"skills": skills.names, "notes": list(notes), "store": skills._by_name.get("remember")})
        if "remember" in skills.names:
            skills.run("remember", json.dumps({"note": f"Run {len(given)} was here."}))
        return {"error": None, "calls": [], "transcript": "x", "answer": "Nadia.", "pending": False,
                "latency_s": 0.1}

    monkeypatch.setattr(eval_run, "run_assistant", fake_run)
    out = tmp_path / "out.json"
    eval_run.main(["--paths", "assistant", "--repeats", "2", "--cases", "memory-recall-sister", "--out", str(out),
                   *argv])
    return given, json.loads(out.read_text())


def test_the_assistant_path_runs_with_memory_on_and_each_run_gets_the_cases_notes_in_a_fresh_store(
        monkeypatch, tmp_path):
    given, document = run_memory_main(monkeypatch, tmp_path)

    # run 1's remembered note is not in run 2: a fresh store each time
    assert [g["notes"] for g in given] == [[NADIA, "The user parks on level three."]] * 2
    assert all(g["skills"][-3:] == ["remember", "forget", "forget_everything"] for g in given)
    assert document["config"]["memory"] is True


def test_no_memory_runs_the_assistant_path_with_the_previous_skill_set_and_no_notes(monkeypatch, tmp_path):
    given, document = run_memory_main(monkeypatch, tmp_path, "--no-memory")

    assert [g["notes"] for g in given] == [[], []]
    assert all(g["skills"] == default_skills().names for g in given)
    assert document["config"]["memory"] is False
    assert eval_run.parse_args(["--paths", "assistant"]).no_memory is False
    assert eval_run.parse_args(["--paths", "assistant", "--no-memory"]).no_memory is True


def test_a_run_never_reads_or_writes_the_apps_notes_file(monkeypatch, tmp_path):
    sentinel = tmp_path / "memory.json"
    sentinel.write_text('{"version": 1, "notes": ["The sentinel note."]}')
    before = (sentinel.read_bytes(), sentinel.stat().st_mtime_ns)
    monkeypatch.setattr(eval_run.config, "MEMORY_FILE", sentinel)
    monkeypatch.setattr(eval_run.config, "DATA_DIR", tmp_path)
    opened = []
    real_read = type(sentinel).read_text

    def spying_read(path, *args, **kwargs):
        opened.append(path)
        return real_read(path, *args, **kwargs)

    monkeypatch.setattr(type(sentinel), "read_text", spying_read)

    given, _ = run_memory_main(monkeypatch, tmp_path)

    assert (sentinel.read_bytes(), sentinel.stat().st_mtime_ns) == before
    assert sentinel not in opened
    assert all("The sentinel note." not in g["notes"] for g in given)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["memory.json", "out.json"]  # nothing else was written


def test_the_memory_cases_are_consistent_with_the_notes_and_the_skills():
    cases = {c["id"]: c for c in json.loads((eval_run.HERE / "assistant_cases.json").read_text())}
    memory_cases = {i: c for i, c in cases.items() if i.startswith("memory-")}

    assert len(memory_cases) >= 12
    for case_id, case in memory_cases.items():
        for note in case.get("notes", []):
            assert eval_run.Memory(None).add(note) == note, case_id  # already as the store would keep it
        if "answer_has" in case["expect"] and case["expect"]["answer_has"] != ["Paris"]:
            held = " ".join(case["notes"]).casefold()  # an answer a note holds, so the recall is the test
            assert all(s.casefold() in held for s in case["expect"]["answer_has"]), case_id
    assert {c["expect"]["tools"][0] for i, c in memory_cases.items() if c["expect"]["tools"]} == {
        "remember", "forget", "forget_everything"}
    assert all(not c.get("notes") for i, c in memory_cases.items() if i.startswith("memory-no-"))
