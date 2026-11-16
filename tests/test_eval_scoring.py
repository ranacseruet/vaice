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
