"""The eval harness's one path: argument handling, the stub skills and `run_case` against a fake realtime server."""
import json
import threading

import pytest
import run as eval_run
from fake_realtime import pcm

from vaice.skills import Question, default_skills

AUDIO = (16000, pcm(1, 2, 3).reshape(1, -1))
NADIA = "The user's sister is called Nadia."
SPOKEN = [("text", "Sorry."), ("audio", pcm(1, 2, 3, 4)), ("done",)]
RECORD = {"error": None, "calls": [], "transcript": "x", "answer": "a", "pending": False, "first_audio_s": 0.1,
          "total_s": 0.2, "audio_s": 1.0, "latency_s": 0.1}


def test_the_defaults_are_the_apps_own():
    args = eval_run.parse_args([])
    assert args.cases_file == "cases.json"
    assert (args.shell, args.vision, args.browse, args.memory, args.history) == (False, False, False, False, True)


def test_the_paths_option_is_gone(capsys):
    with pytest.raises(SystemExit):
        eval_run.parse_args(["--paths", "assistant"])
    assert "--paths" in capsys.readouterr().err


def test_the_flags_turn_the_optional_skills_on():
    args = eval_run.parse_args(["--shell", "--vision", "--browse", "--memory", "--no-history"])
    assert (args.shell, args.vision, args.browse, args.memory, args.history) == (True, True, True, True, False)


# -- the stub skills


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


def test_the_system_info_stub_has_a_neutral_hostname():
    assert eval_run.SKILL_STUBS["system_info"] == (
        "macOS 15.6 on Example-MacBook, up 3 days and 4 hours, 48 gigabytes of memory")


def test_the_shell_skill_is_offered_only_with_the_flag_and_its_stub_never_runs_a_command(monkeypatch):
    def boom(*_args, **_kwargs):
        raise AssertionError("the stub started a process")

    monkeypatch.setattr("subprocess.Popen", boom)
    monkeypatch.setattr("subprocess.run", boom)

    assert "run_shell" not in eval_run.stub_skills().names
    stubs = eval_run.stub_skills(shell=True)
    assert stubs.definitions() == default_skills(shell=True).definitions()
    assert stubs.names[-1] == "run_shell"
    (skill,) = [s for s in stubs.skills if s.name == "run_shell"]
    assert skill.run_stoppable is None  # the real runner cannot be reached

    assert stubs.prepare("run_shell", '{"command": "rm -rf /"}') == Question("Shall I run run shell?")
    assert stubs.run("run_shell", '{"command": "rm -rf /"}', lambda: False).speech == "the command finished. Output: ok"


def test_the_vision_skill_is_offered_only_with_the_flag_and_its_stub_never_captures_or_sends(monkeypatch):
    def boom(*_args, **_kwargs):
        raise AssertionError("the stub took a screenshot or reached a network")

    monkeypatch.setattr("subprocess.run", boom)
    monkeypatch.setattr("subprocess.Popen", boom)
    monkeypatch.setattr("requests.post", boom)
    monkeypatch.setattr("vaice.vision.capture", boom)

    assert "look_at_screen" not in eval_run.stub_skills().names
    stubs = eval_run.stub_skills(vision=True)
    assert stubs.names.index("look_at_screen") == 5  # as in the app: right after the read-only checks

    dialog = stubs.run("look_at_screen", '{"question": "What does this dialog say?"}').speech
    assert dialog == ('the screen shows: a dialog that says "Your disk is almost full" over a Numbers window titled '
                      'Quarterly report')
    assert stubs.run("look_at_screen", '{"question": "What app is open?"}').speech == (
        "the screen shows: a Numbers window titled Quarterly report")
    assert stubs.run("look_at_screen", "{}").speech.startswith("the look at screen skill failed")  # real validation


def test_the_browse_skill_is_offered_only_with_the_flag_and_nothing_is_started():
    from vaice.skills import Started

    assert "browse" not in eval_run.stub_skills().names
    skills = eval_run.stub_skills(browse=True)
    assert {"browse", "browser_status", "stop_browsing"} <= set(skills.names)
    assert isinstance(skills.prepare("browse", '{"goal": "open python.org"}'), Started)  # validated, nothing run
    assert skills.run("browser_status", "{}").speech == eval_run.BROWSER_STATUS_STUB  # canned, nothing asked
    assert skills.run("stop_browsing", "{}").speech == "stopped the browser task"


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


# -- run_case: the real responder over a fake realtime server


def test_run_case_records_the_calls_the_transcript_the_answer_and_the_timings(realtime):
    server = realtime(
        reply=[
            [("call", "item_1", "disk_space", "call_1", "{}"), ("done",)],
            [("text", "About 160 gigabytes."), ("audio", pcm(1, 2)), ("done",)],
        ],
        transcription=[("transcript", "How much disk space?")],
    )

    record = eval_run.run_case(server.url, "prompt", eval_run.stub_skills(), None, AUDIO, [])

    assert record["error"] is None
    assert record["calls"] == [["disk_space", "{}"]]
    assert record["pending"] is False
    assert record["transcript"] == "How much disk space?"
    assert record["answer"] == "About 160 gigabytes."
    assert 0 <= record["first_audio_s"] <= record["total_s"] and record["latency_s"] == record["first_audio_s"]
    assert record["audio_s"] > 0
    (output,) = [e["item"]["output"] for e in server.conversation().events
                 if e["type"] == "conversation.item.create"]
    assert output == eval_run.SKILL_STUBS["disk_space"]  # the stub's result went back to the session


def test_a_confirm_skill_is_counted_as_the_call_and_nothing_runs(realtime, monkeypatch):
    def boom(*_args, **_kwargs):
        raise AssertionError("a confirm skill ran")

    monkeypatch.setattr("subprocess.run", boom)
    server = realtime(
        reply=[
            [("call", "item_1", "quit_app", "call_1", '{"name": "Safari"}'), ("done",)],
            [("text", "Press Run."), ("audio", pcm(1)), ("done",)],
        ],
    )

    record = eval_run.run_case(server.url, "prompt", eval_run.stub_skills(), None, AUDIO, [])

    assert record["calls"] == [["quit_app", '{"name": "Safari"}']] and record["pending"] is True
    score = eval_run.scoring.score({"tools": ["quit_app"], "args": {"quit_app": {"name": "Safari"}}},
                                   [tuple(c) for c in record["calls"]])
    assert score["verdict"] == "correct"


def test_calls_in_one_reply_are_all_kept_in_order(realtime):
    server = realtime(
        reply=[
            [("call", "item_1", "disk_space", "call_1", "{}"),
             ("call", "item_2", "open_app", "call_2", '{"name": "Calculator"}'), ("done",)],
            [("text", "Done."), ("audio", pcm(1)), ("done",)],
        ],
    )

    record = eval_run.run_case(server.url, "prompt", eval_run.stub_skills(), None, AUDIO, [])

    assert record["calls"] == [["disk_space", "{}"], ["open_app", '{"name": "Calculator"}']]


def test_a_background_call_is_scored_as_the_call_and_marked_pending(realtime):
    server = realtime(
        reply=[
            [("call", "item_1", "browse", "call_1", '{"goal": "open python.org"}'), ("done",)],
            [("text", "On it."), ("audio", pcm(1)), ("done",)],
        ],
    )

    record = eval_run.run_case(server.url, "prompt", eval_run.stub_skills(browse=True), None, AUDIO, [])

    assert record["calls"] == [["browse", '{"goal": "open python.org"}']] and record["pending"] is True
    assert record["error"] is None


def browse_server(realtime):
    return realtime(reply=[
        [("call", "item_1", "browse", "call_1", '{"goal": "open python.org"}'), ("done",)],
        [("text", "Here are some results."), ("audio", pcm(1)), ("done",)],  # what the old continuation would say
    ])


def test_a_browse_run_starts_a_task_on_a_fake_agent_ends_the_reply_and_reads_the_calls_before_stopping_it(realtime):
    server = browse_server(realtime)

    record = eval_run.run_case(server.url, "prompt", eval_run.stub_skills(browse=True), None, AUDIO, [])

    assert record["error"] is None and record["browse_start"] is True and record["pending"] is True
    assert record["calls"] == [["browse", '{"goal": "open python.org"}']]  # not the task's own `stopped` event
    assert record["audio_s"] == 0 and record["first_audio_s"] is None and record["answer"] == ""  # it ended there
    assert server.conversation().types.count("response.create") == 1  # the model was not asked to continue
    assert not [t for t in threading.enumerate() if t.name == eval_run.BROWSE_THREAD]  # stopped and joined


def test_a_run_without_browse_has_no_task_and_no_browse_start(realtime):
    record = eval_run.run_case(realtime().url, "prompt", eval_run.stub_skills(), None, AUDIO, [])

    assert record["browse_start"] is False
    assert not [t for t in threading.enumerate() if t.name == eval_run.BROWSE_THREAD]


def test_a_browse_call_that_did_not_start_is_not_a_browse_start(realtime):
    # a bad goal is a failed call, not a started task
    server = realtime(reply=[[("call", "item_1", "browse", "call_1", "{}"), ("done",)], SPOKEN])

    record = eval_run.run_case(server.url, "prompt", eval_run.stub_skills(browse=True), None, AUDIO, [])

    assert record["browse_start"] is False and record["pending"] is False and record["audio_s"] > 0


def test_the_notes_go_into_the_instructions_as_the_app_does(realtime):
    with_notes, without = realtime(), realtime()
    memory = eval_run.Memory(None)
    memory.add(NADIA)

    eval_run.run_case(with_notes.url, "prompt", eval_run.stub_skills(memory=memory), memory, AUDIO, [])
    eval_run.run_case(without.url, "prompt", eval_run.stub_skills(), None, AUDIO, [])

    assert NADIA in with_notes.conversation().session["instructions"]
    assert "Notes the user" not in without.conversation().session["instructions"]


def test_run_case_records_a_failure_instead_of_raising(realtime):
    server = realtime(reply=[("error", "boom")])

    record = eval_run.run_case(server.url, "prompt", eval_run.stub_skills(), None, AUDIO, [])

    assert record["error"].startswith("RuntimeError") and "boom" in record["error"]
    assert record["first_audio_s"] is None and record["latency_s"] is None


def test_a_case_history_is_sent_with_the_reply(realtime):
    server = realtime()

    eval_run.run_case(server.url, "prompt", eval_run.stub_skills(), None, AUDIO,
                      [("user", "Hi"), ("assistant", "Hello.")])

    instructions = server.conversation().session["instructions"]
    assert "<|im_start|>user\nHi" in instructions and "<|im_start|>assistant\nHello." in instructions


# -- main, with no server, `say` or model


@pytest.fixture
def offline(monkeypatch):
    monkeypatch.setattr(eval_run, "server_info", lambda url: {"url": url})
    monkeypatch.setattr(eval_run, "audio_for", lambda case, scratch: (16000, None))


def test_the_summary_and_the_table_report_answer_checks_apart_from_the_verdict(offline, monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(eval_run, "run_case", lambda *args: {
        **RECORD, "calls": [["disk_space", "{}"]], "answer": "Plenty of room.", "first_audio_s": 0.5, "audio_s": 2.0})
    out = tmp_path / "out.json"

    eval_run.main(["--repeats", "2", "--cases", "disk-free,chat-goodbye", "--out", str(out)])

    printed = capsys.readouterr().out
    assert "  correct" in printed and "answer faithful 0/2" in printed  # two disk-free runs, neither says 160
    assert "command ok" not in printed  # no case asked for it
    assert "audio per reply: median 2.00s" in printed
    table = printed.split("case ", 1)[1]
    (disk_row,) = [line for line in table.splitlines() if line.startswith("disk-free")]
    assert disk_row.split() == ["disk-free", "2/2", "0/2"]  # the call verdict is right, the answer is not
    document = json.loads(out.read_text())
    first = document["runs"][0]["score"]
    assert first["verdict"] == "correct" and first["answer_ok"] is False and first["command_ok"] is None
    assert document["summary"]["answer_faithful"] == {"ok": 0, "of": 2}
    assert document["config"]["cases_file"] == "cases.json"


def test_browser_starts_are_left_out_of_the_silent_latency_and_audio_numbers_and_counted_apart(
        offline, monkeypatch, tmp_path, capsys):
    started = {**RECORD, "calls": [["browse", '{"goal": "g"}']], "pending": True, "browse_start": True,
               "audio_s": 0.0, "first_audio_s": None, "latency_s": None}
    answered = {**RECORD, "calls": [["disk_space", "{}"]], "answer": "Plenty."}
    runs = iter([started,
                 {**started, "audio_s": 3.0, "first_audio_s": 9.0, "latency_s": 9.0},  # a start with some audio too
                 {**answered, "audio_s": 2.0, "latency_s": 0.5},
                 {**answered, "audio_s": 0.1, "latency_s": 0.7}])
    monkeypatch.setattr(eval_run, "run_case", lambda *args: next(runs))
    out = tmp_path / "out.json"

    eval_run.main(["--repeats", "4", "--cases", "disk-free", "--out", str(out)])

    summary = json.loads(out.read_text())["summary"]
    printed = capsys.readouterr().out
    assert summary["silent"] == 1 and summary["audio_median_s"] == 1.05  # only the two answered runs
    assert summary["browse_starts"] == 2
    assert summary["latency_median_s"] == 0.6 and summary["latency_max_s"] == 0.7
    assert "browser starts: 2" in printed and "under 0.5 s: 1" in printed
    assert json.loads(out.read_text())["runs"][0]["browse_start"] is True


def test_without_browser_starts_the_summary_prints_no_extra_line(offline, monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(eval_run, "run_case", lambda *args: RECORD)

    eval_run.main(["--repeats", "1", "--cases", "disk-free", "--out", str(tmp_path / "out.json")])

    assert "browser starts" not in capsys.readouterr().out


def test_the_command_check_is_reported_for_the_shell_cases(offline, monkeypatch, tmp_path, capsys):
    commands = iter(["ls ~/Downloads", "ls /Users/me/Downloads"])
    monkeypatch.setattr(eval_run, "run_case", lambda *args: {
        **RECORD, "calls": [["run_shell", json.dumps({"command": next(commands)})]], "answer": "Press Run.",
        "pending": True})

    eval_run.main(["--shell", "--cases-file", "shell_cases.json", "--repeats", "2", "--cases", "shell-ls-downloads",
                   "--out", str(tmp_path / "out.json")])

    out = capsys.readouterr().out
    assert "command ok 1/2" in out and "answer faithful" not in out
    (row,) = [line for line in out.splitlines() if line.startswith("shell-ls-downloads")]
    assert row.split() == ["shell-ls-downloads", "2/2", "1/2"]  # the verdict is right twice, the command once


def test_the_vision_cases_run_with_the_flag_and_record_it_in_the_results(offline, monkeypatch, tmp_path):
    offered = []

    def run(url, prompt, skills, memory, audio, history):
        offered.append("look_at_screen" in skills.names)
        return {**RECORD, "calls": [["look_at_screen", '{"question": "q"}']], "answer": "A Numbers window."}

    monkeypatch.setattr(eval_run, "run_case", run)

    eval_run.main(["--vision", "--cases-file", "vision_cases.json", "--repeats", "1",
                   "--cases", "vision-whats-on,novision-sky", "--out", str(tmp_path / "out.json")])

    assert offered == [True, True]
    document = json.loads((tmp_path / "out.json").read_text())
    assert document["config"]["vision"] is True and document["config"]["shell"] is False
    rows = {r["case"]: r["score"]["verdict"] for r in document["runs"]}
    assert rows == {"vision-whats-on": "correct", "novision-sky": "overcall"}  # a call where none may be made


def test_a_case_with_its_own_history_runs_with_it_even_without_the_history_flag(offline, monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr(eval_run, "run_case", lambda url, prompt, skills, memory, audio, history:
                        seen.append(history) or RECORD)

    eval_run.main(["--no-history", "--vision", "--repeats", "1", "--cases-file", "vision_cases.json",
                   "--cases", "vision-look-again,vision-whats-on", "--out", str(tmp_path / "out.json")])

    with_own, without = seen
    assert [role for role, _ in with_own] == ["user", "assistant"]  # the case's own turns, as tuples
    assert all(isinstance(entry, tuple) for entry in with_own)
    assert without == []  # a case with none follows --no-history


def test_memory_is_off_by_default_and_each_run_gets_the_cases_notes_in_a_fresh_store(offline, monkeypatch, tmp_path):
    given = []

    def run(url, prompt, skills, memory, audio, history):
        given.append({"skills": skills.names, "notes": [] if memory is None else memory.notes()})
        if memory is not None:
            skills.run("remember", json.dumps({"note": f"Run {len(given)} was here."}))
        return RECORD

    monkeypatch.setattr(eval_run, "run_case", run)
    argv = ["--repeats", "2", "--cases", "memory-recall-sister", "--out", str(tmp_path / "out.json")]

    eval_run.main(argv)
    assert [g["notes"] for g in given] == [[], []]
    assert all(g["skills"] == default_skills().names for g in given)
    assert json.loads((tmp_path / "out.json").read_text())["config"]["memory"] is False

    given.clear()
    eval_run.main([*argv, "--memory"])
    # run 1's remembered note is not in run 2: a fresh store each time
    assert [g["notes"] for g in given] == [[NADIA, "The user parks on level three."]] * 2
    assert all(g["skills"][-3:] == ["remember", "forget", "forget_everything"] for g in given)
    assert json.loads((tmp_path / "out.json").read_text())["config"]["memory"] is True


def test_a_run_never_reads_or_writes_the_apps_notes_file(offline, monkeypatch, tmp_path):
    sentinel = tmp_path / "memory.json"
    sentinel.write_text('{"version": 1, "notes": ["The sentinel note."]}')
    before = (sentinel.read_bytes(), sentinel.stat().st_mtime_ns)
    monkeypatch.setattr(eval_run.config, "DATA_DIR", tmp_path)
    notes = []

    def run(url, prompt, skills, memory, audio, history):
        notes.extend(memory.notes())
        return RECORD

    monkeypatch.setattr(eval_run, "run_case", run)

    eval_run.main(["--memory", "--repeats", "1", "--cases", "memory-recall-sister",
                   "--out", str(tmp_path / "out.json")])

    assert (sentinel.read_bytes(), sentinel.stat().st_mtime_ns) == before
    assert "The sentinel note." not in notes
    assert sorted(p.name for p in tmp_path.iterdir()) == ["memory.json", "out.json"]  # nothing else was written


# -- the case files


def test_the_memory_cases_are_consistent_with_the_notes_and_the_skills():
    cases = {c["id"]: c for c in json.loads((eval_run.HERE / "cases.json").read_text())}
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


def test_the_shipped_case_histories_are_plain_spoken_turns():
    seen = 0
    for name in ("cases.json", "vision_cases.json", "shell_cases.json", "browse_cases.json"):
        for case in json.loads((eval_run.HERE / name).read_text()):
            if "history" in case:
                seen += 1
                turns = eval_run.case_history(case["history"])
                assert turns and all(role in ("user", "assistant") and isinstance(text, str) for role, text in turns)

    assert seen == 5
