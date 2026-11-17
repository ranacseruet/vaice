"""The eval harness's pure scoring logic and its case files; nothing here needs a server, `say` or the network."""
import json
from pathlib import Path

import pytest
import scoring

EVALS = Path(__file__).parent.parent / "evals"
OPEN_SAFARI = {"tools": ["open_app"], "args": {"open_app": {"name": "Safari"}}}


def call(tool, /, **arguments):  # positional-only: `name` is also an argument of open_app
    return (tool, json.dumps(arguments))


def verdict(expect, calls):
    return scoring.score(expect, calls)["verdict"]


# --- expect: {"tools": [...]} ---


def test_expected_tool_with_matching_args_is_correct():
    assert verdict(OPEN_SAFARI, [call("open_app", name="Safari")]) == "correct"


def test_no_call_when_one_was_expected_is_missed():
    result = scoring.score(OPEN_SAFARI, [])
    assert result["verdict"] == "missed"
    assert "no tool called" in result["reason"]


def test_wrong_tool_is_missed_and_names_both_sides():
    result = scoring.score(OPEN_SAFARI, [call("web_search", query="safari")])
    assert result["verdict"] == "missed"
    assert "missing ['open_app']" in result["reason"]
    assert "extra ['web_search']" in result["reason"]


def test_an_extra_different_tool_is_missed_and_named():
    result = scoring.score(OPEN_SAFARI, [call("open_app", name="Safari"), call("run_shell", command="ls")])
    assert result["verdict"] == "missed"
    assert result["reason"] == "extra ['run_shell']"


def test_two_step_needs_both_tools_in_any_order():
    expect = {"tools": ["open_app", "set_volume"]}
    assert verdict(expect, [call("set_volume", level=50), call("open_app", name="Terminal")]) == "correct"
    assert verdict(expect, [call("open_app", name="Terminal")]) == "missed"


def test_duplicates_of_an_expected_tool_do_not_change_the_verdict():
    calls = [call("open_app", name="Safari"), call("open_app", name="Safari")]
    result = scoring.score(OPEN_SAFARI, calls)
    assert result["verdict"] == "correct"
    assert result["duplicate"] is True


# --- expect: {"tools": []} ---


def test_no_call_expected_and_none_made_is_correct():
    assert verdict({"tools": []}, []) == "correct"


def test_any_call_when_none_expected_is_overcall():
    result = scoring.score({"tools": []}, [call("web_search", query="sky")])
    assert result["verdict"] == "overcall"
    assert "web_search" in result["reason"]


# --- expect: {"any_of": [...]} ---


def test_any_of_accepts_any_listed_tool():
    expect = {"any_of": ["run_shell", "disk_space"]}
    assert verdict(expect, [call("disk_space")]) == "correct"
    assert verdict(expect, [call("run_shell", command="df -h")]) == "correct"
    assert verdict(expect, [call("run_shell", command="df -h"), call("disk_space")]) == "correct"


def test_any_of_needs_a_call():
    assert verdict({"any_of": ["run_shell"]}, []) == "missed"


def test_any_of_rejects_a_call_outside_the_set():
    result = scoring.score({"any_of": ["run_shell", "disk_space"]}, [call("disk_space"), call("web_search", query="x")])
    assert result["verdict"] == "missed"
    assert "web_search" in result["reason"]


def test_an_expectation_without_tools_or_any_of_is_an_error():
    with pytest.raises(ValueError, match="tools"):
        scoring.score({}, [])


# --- argument matching ---


@pytest.mark.parametrize("given", ["safari", "SAFARI", "  Safari  "])
def test_string_args_compare_case_insensitively_after_stripping(given):
    assert verdict(OPEN_SAFARI, [call("open_app", name=given)]) == "correct"


def test_a_different_string_is_wrong_args():
    result = scoring.score(OPEN_SAFARI, [call("open_app", name="Chrome")])
    assert result["verdict"] == "wrong_args"
    assert "name" in result["reason"]


def test_a_list_holds_accepted_alternatives():
    expect = {"tools": ["open_app"], "args": {"open_app": {"name": ["Notes", "Notes app"]}}}
    assert verdict(expect, [call("open_app", name="notes app")]) == "correct"
    assert verdict(expect, [call("open_app", name="Reminders")]) == "wrong_args"


@pytest.mark.parametrize("given", [20, 20.0, "20"])
def test_numbers_compare_by_value(given):
    expect = {"tools": ["set_volume"], "args": {"set_volume": {"level": 20}}}
    assert verdict(expect, [call("set_volume", level=given)]) == "correct"


def test_a_different_number_is_wrong_args():
    expect = {"tools": ["set_volume"], "args": {"set_volume": {"level": 20}}}
    assert verdict(expect, [call("set_volume", level=25)]) == "wrong_args"
    assert verdict(expect, [call("set_volume", level="loud")]) == "wrong_args"
    assert verdict(expect, [call("set_volume", level=True)]) == "wrong_args"


def test_zero_is_a_number_not_a_missing_value():
    expect = {"tools": ["set_volume"], "args": {"set_volume": {"level": 0}}}
    assert verdict(expect, [call("set_volume", level=0)]) == "correct"
    assert verdict(expect, [call("set_volume")]) == "wrong_args"


def test_a_missing_key_is_wrong_args():
    result = scoring.score(OPEN_SAFARI, [call("open_app")])
    assert result["verdict"] == "wrong_args"
    assert "missing 'name'" in result["reason"]


def test_extra_keys_are_ignored_because_args_are_a_subset_match():
    assert verdict(OPEN_SAFARI, [call("open_app", name="Safari", background=True)]) == "correct"


def test_unparseable_arguments_on_an_expected_tool_are_wrong_args():
    for bad in ("{not json", "[1, 2]", '"Safari"'):
        assert verdict(OPEN_SAFARI, [("open_app", bad)]) == "wrong_args"
    assert verdict({"tools": ["open_app"]}, [("open_app", "{not json")]) == "wrong_args"


def test_empty_arguments_are_an_empty_object():
    assert verdict({"tools": ["disk_space"]}, [("disk_space", "")]) == "correct"
    assert verdict({"tools": ["disk_space"]}, [("disk_space", "{}")]) == "correct"


def test_args_for_one_tool_are_checked_per_tool_in_a_two_step_request():
    expect = {"tools": ["open_app", "set_volume"],
              "args": {"open_app": {"name": "Terminal"}, "set_volume": {"level": 50}}}
    assert verdict(expect, [call("open_app", name="Terminal"), call("set_volume", level=50)]) == "correct"
    assert verdict(expect, [call("open_app", name="Terminal"), call("set_volume", level=5)]) == "wrong_args"


def test_one_matching_call_is_enough_when_a_tool_is_called_twice():
    calls = [call("open_app", name="Chrome"), call("open_app", name="Safari")]
    result = scoring.score(OPEN_SAFARI, calls)
    assert result["verdict"] == "correct"
    assert result["duplicate"] is False  # same tool, different arguments


# --- duplicates ---


def test_duplicate_means_same_tool_and_same_arguments_more_than_once():
    assert scoring.score({"tools": []}, [])["duplicate"] is False
    assert scoring.score({"tools": ["disk_space"]}, [call("disk_space")])["duplicate"] is False
    twice = [("set_volume", '{"level": 5}'), ("set_volume", '{"level":5}')]
    assert scoring.score({"tools": ["set_volume"]}, twice)["duplicate"] is True  # formatting does not matter


def test_duplicate_is_reported_next_to_a_failing_verdict():
    result = scoring.score({"tools": []}, [call("disk_space"), call("disk_space")])
    assert result["verdict"] == "overcall"
    assert result["duplicate"] is True


# --- aggregation ---


def record(case, verdict_name=None, *, error=None, latency=None, duplicate=False):
    out = {"case": case, "error": error, "latency_s": latency}
    if verdict_name:
        out["score"] = {"verdict": verdict_name, "reason": "", "duplicate": duplicate}
    return out


def test_aggregate_counts_percentages_and_latency():
    records = [
        record("a", "correct", latency=1.0),
        record("a", "correct", latency=3.0, duplicate=True),
        record("b", "missed", latency=2.0),
        record("b", error="boom"),
    ]
    agg = scoring.aggregate(records)
    assert agg["runs"] == 4
    assert agg["counts"] == {"correct": 2, "missed": 1, "overcall": 0, "wrong_args": 0, "error": 1}
    assert agg["percent"]["correct"] == 50.0
    assert agg["percent"]["error"] == 25.0
    assert agg["duplicate"] == 1
    assert agg["duplicate_percent"] == 25.0
    assert agg["latency_median_s"] == 2.0
    assert agg["latency_max_s"] == 3.0


def test_aggregate_with_zero_runs_has_zero_percentages_and_no_latency():
    agg = scoring.aggregate([])
    assert agg["runs"] == 0
    assert set(agg["percent"].values()) == {0.0}
    assert agg["duplicate_percent"] == 0.0
    assert agg["latency_median_s"] is None
    assert agg["latency_max_s"] is None


def test_aggregate_ignores_errored_runs_for_latency_and_duplicates():
    agg = scoring.aggregate([record("a", error="boom", latency=99.0)])
    assert agg["counts"]["error"] == 1
    assert agg["latency_max_s"] is None


def test_per_case_counts_correct_over_all_runs():
    records = [record("a", "correct"), record("a", "overcall"), record("b", error="boom"), record("b", "correct")]
    assert scoring.per_case(records) == {"a": (1, 2), "b": (1, 2)}
    assert scoring.per_case([]) == {}


# --- transcripts ---


def test_normalize_transcript_ignores_case_punctuation_and_spacing():
    assert scoring.normalize_transcript("  Open   Safari, for me! ") == "open safari for me"
    assert scoring.normalize_transcript("What's the capital of France?") == "whats the capital of france"


def test_misheard_flags_only_real_differences():
    assert not scoring.misheard("Open Safari for me.", "open safari for me")
    assert scoring.misheard("Open Safari for me.", "Open Safari for free.")
    assert scoring.misheard("Mute the sound.", "")


# --- answer and command checks (apart from the call verdict) ---


def checks(expect, calls=(), answer=None, pending=False):
    return scoring.check_answer(expect, list(calls), answer, pending)


def test_no_checks_asked_for_means_none_for_both():
    assert checks({"tools": ["disk_space"]}, [call("disk_space")], "About 160 gigabytes.") == {
        "answer_ok": None, "command_ok": None}


def test_answer_has_needs_every_string_in_the_answer_ignoring_case():
    expect = {"tools": ["battery_status"], "answer_has": ["78", "Percent"]}
    assert checks(expect, answer="You have 78 PERCENT left.")["answer_ok"] is True
    assert checks(expect, answer="You have 78 left.")["answer_ok"] is False  # one string missing
    assert checks(expect, answer="")["answer_ok"] is False
    assert checks(expect, answer=None)["answer_ok"] is False


def test_answer_has_is_not_applicable_when_the_run_ended_in_a_confirmation_question():
    expect = {"tools": ["quit_app"], "answer_has": ["Safari"]}
    assert checks(expect, answer="Shall I quit Safari?", pending=True)["answer_ok"] is None


def test_the_answer_check_is_independent_of_the_verdict():
    expect = {"tools": ["disk_space"], "answer_has": ["160"]}
    calls = [call("disk_space")]
    assert verdict(expect, calls) == "correct"
    assert checks(expect, calls, "Plenty of room.")["answer_ok"] is False  # right call, wrong answer


def test_command_matches_and_command_not_apply_to_the_run_shell_command():
    expect = {"tools": ["run_shell"], "command_matches": "Downloads", "command_not": "/Users/me"}
    assert checks(expect, [call("run_shell", command="ls ~/Downloads")])["command_ok"] is True
    assert checks(expect, [call("run_shell", command="ls ~/downloads")])["command_ok"] is False  # case-sensitive
    assert checks(expect, [call("run_shell", command="ls /Users/me/Downloads")])["command_ok"] is False
    assert checks(expect, [call("run_shell", command="ls ~/Desktop")])["command_ok"] is False


def test_command_not_alone_and_regular_expressions():
    expect = {"tools": ["run_shell"], "command_not": "netstat -[a-z]*p"}
    assert checks(expect, [call("run_shell", command="lsof -i :8080")])["command_ok"] is True
    assert checks(expect, [call("run_shell", command="netstat -anp | grep 8080")])["command_ok"] is False
    assert checks({"command_matches": "^ping .*google"}, [call("run_shell", command="ping -c 3 google.com")])[
        "command_ok"] is True


def test_the_command_check_fails_without_a_run_shell_call_or_a_command_and_judges_every_call():
    expect = {"tools": ["run_shell"], "command_matches": "ping"}
    assert checks(expect, [])["command_ok"] is False
    assert checks(expect, [call("disk_space")])["command_ok"] is False
    assert checks(expect, [("run_shell", "not json")])["command_ok"] is False
    assert checks(expect, [call("run_shell", command=7)])["command_ok"] is False
    assert checks(expect, [call("run_shell", command="ping a"), call("run_shell", command="ls")])["command_ok"] is False
    assert checks(expect, [call("run_shell", command="ping a"), call("run_shell", command="ping b")])[
        "command_ok"] is True


def checked(case, answer_ok=None, command_ok=None, error=None):
    out = {"case": case, "error": error, "latency_s": None}
    if not error:
        out["score"] = {"verdict": "correct", "reason": "", "duplicate": False, "answer_ok": answer_ok,
                        "command_ok": command_ok}
    return out


def test_aggregate_adds_the_check_totals_only_when_some_run_asked_for_them():
    assert "answer_faithful" not in scoring.aggregate([record("a", "correct")])  # old records have no such keys
    records = [checked("a", True), checked("a", False), checked("b"), checked("c", command_ok=True),
               checked("c", error="boom")]

    agg = scoring.aggregate(records)

    assert agg["answer_faithful"] == {"ok": 1, "of": 2}  # the run without the expectation is not counted
    assert agg["command_ok"] == {"ok": 1, "of": 1}  # an errored run has no score at all
    assert agg["counts"]["correct"] == 4  # the verdicts are as before


def test_per_case_check_counts_per_case_and_leaves_out_cases_it_never_applied_to():
    records = [checked("a", True), checked("a", False), checked("b"), checked("c", True, error="boom")]
    assert scoring.per_case_check(records, "answer_ok") == {"a": (1, 2)}


# --- the case files ---


def load(name):
    return json.loads((EVALS / name).read_text())


def test_case_and_tool_files_are_consistent():
    cases, tools = load("cases.json"), load("tools.json")
    names = [t["definition"]["name"] for t in tools]
    assert len(names) == len(set(names)) == 6
    for tool in tools:
        assert tool["definition"]["type"] == "function"
        assert tool["definition"]["parameters"]["type"] == "object"
        assert isinstance(tool["result"], str)

    ids = [c["id"] for c in cases]
    assert len(ids) == len(set(ids))
    for c in cases:
        assert c["say"].strip()
        expected = c["expect"].get("tools", c["expect"].get("any_of"))
        assert isinstance(expected, list), c["id"]
        assert set(expected) <= set(names), c["id"]
        assert set(c["expect"].get("args", {})) <= set(expected), c["id"]
        scoring.score(c["expect"], [])  # the expectation is well formed

    assert sum(1 for c in cases if c["expect"].get("tools") == []) >= 8


def test_the_required_sentences_are_cases():
    said = {c["say"] for c in load("cases.json")}
    assert {
        "Open Safari for me.",
        "How much free disk space do I have?",
        "Turn the volume down to twenty percent.",
        "Remind me to call mom tomorrow at nine.",
        "What's the latest news about SpaceX?",
        "Thanks, that's all for now.",
        "Why is the sky blue?",
        "Open the terminal and then set the volume to fifty.",
    } <= said


def test_the_assistant_cases_are_consistent_with_the_skills():
    from vaice.skills import default_skills

    cases = load("assistant_cases.json")
    skills = default_skills()
    ids = [c["id"] for c in cases]
    assert len(ids) == len(set(ids))
    assert len(cases) >= 30
    for c in cases:
        assert c["say"].strip()
        expected = c["expect"]["tools"]
        assert isinstance(expected, list), c["id"]
        assert set(expected) <= set(skills.names), c["id"]
        assert set(c["expect"].get("args", {})) <= set(expected), c["id"]
        for tool, wanted in c["expect"].get("args", {}).items():
            properties = next(s for s in skills.skills if s.name == tool).parameters["properties"]
            assert set(wanted) <= set(properties), c["id"]
        scoring.score(c["expect"], [])  # the expectation is well formed

    assert sum(1 for c in cases if c["expect"]["tools"] == []) >= 10
    assert {name for c in cases for name in c["expect"]["tools"]} == set(skills.names)  # every skill is exercised
    assert sum(1 for c in cases if len(c["expect"]["tools"]) == 2) >= 3


def test_the_answer_and_command_expectations_are_well_formed_and_where_the_stub_result_allows_them():
    import re

    from run import SKILL_STUBS

    cases = {c["id"]: c["expect"] for name in ("assistant_cases.json", "assistant_shell_cases.json")
             for c in load(name)}
    for case_id, expect in cases.items():
        if "answer_has" in expect:
            assert expect["tools"] and all(isinstance(s, str) and s for s in expect["answer_has"]), case_id
            stub = SKILL_STUBS[expect["tools"][0]].casefold()
            assert all(s.casefold() in stub for s in expect["answer_has"]), case_id  # a fact the stub result holds
        for key in ("command_matches", "command_not"):
            if key in expect:
                assert expect["tools"] == ["run_shell"], case_id
                re.compile(expect[key])

    assert {case_id: cases[case_id]["answer_has"] for case_id in (
        "disk-free", "battery-level", "battery-left", "system-memory", "system-os", "system-uptime",
        "procs-memory")} == {"disk-free": ["160"], "battery-level": ["78"], "battery-left": ["4 hours"],
                             "system-memory": ["48"], "system-os": ["15.6"], "system-uptime": ["3 days"],
                             "procs-memory": ["Chrome"]}
    assert cases["shell-ls-downloads"] == {"tools": ["run_shell"], "command_matches": "Downloads",
                                           "command_not": "/Users/me"}
    assert cases["shell-terminal-list"]["command_matches"] == "Desktop"
    assert cases["shell-port"]["command_matches"] == "8080"
    assert cases["shell-port"]["command_not"] == "netstat -[a-z]*p|sudo"
    assert {c: cases[c]["command_matches"] for c in ("shell-git-status", "shell-whoami", "shell-ping")} == {
        "shell-git-status": "git .*status", "shell-whoami": "whoami", "shell-ping": "ping"}


def test_the_shell_cases_are_consistent_with_the_skills_when_run_shell_is_on():
    from vaice.skills import default_skills

    cases = load("assistant_shell_cases.json")
    skills = default_skills(shell=True)
    ids = [c["id"] for c in cases]
    assert len(ids) == len(set(ids))
    assert len(cases) >= 10
    for c in cases:
        assert c["say"].strip()
        expected = c["expect"]["tools"]
        assert isinstance(expected, list), c["id"]
        assert set(expected) <= set(skills.names), c["id"]
        assert not c["expect"].get("args", {}).get("run_shell"), c["id"]  # only the tool is expected, not the command
        scoring.score(c["expect"], [])  # the expectation is well formed

    assert sum(1 for c in cases if c["expect"]["tools"] == ["run_shell"]) >= 4
    assert sum(1 for c in cases if c["expect"]["tools"] == []) >= 4  # tempting ones that must not call it
    # without --shell the skill does not exist: this file is only for runs that pass it
    assert "run_shell" not in default_skills().names
