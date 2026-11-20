"""Skill kinds, and the skills that act on the Mac, against a fake `_command`: nothing here runs open, osascript
or lsappinfo."""
import json

import pytest

from vaice import skills as skills_module
from vaice.skills import Question, Skill, SkillError, SkillResult, Skills, default_skills

LSAPPINFO = """ 1) "loginwindow" ASN:0x0-0x3003:
    bundleID="com.apple.loginwindow"
    pid = 434 type="UIElement" flavor=3 Version="3085.6.3" fileType="APPL" creator="lgnw" Arch=ARM64
 9) "Finder" ASN:0x0-0x15015:
    bundleID="com.apple.finder"
    pid = 520 type="Foreground" flavor=3 Version="15.6" fileType="APPL" creator="MACS" Arch=ARM64
38) "Google Chrome" ASN:0x0-0x1424423: (in front)
    bundleID="com.google.Chrome"
    pid = 71197 type="Foreground" flavor=3 Version="8078.81" fileType="APPL" creator="rimZ" Arch=ARM64
39) "AutoFill (Google Chrome)" ASN:0x0-0x1427426:
    pid = 71200 type="BackgroundOnly" flavor=2 Version="1" fileType="APPL" creator="????" Arch=ARM64
41) "Safari" ASN:0x0-0x1454453:
    pid = 97959 type="Foreground" flavor=3 Version="618" fileType="APPL" creator="sfri" Arch=ARM64
"""


@pytest.fixture(autouse=True)
def on_a_mac(monkeypatch):
    monkeypatch.setattr(skills_module.sys, "platform", "darwin")


@pytest.fixture
def script(monkeypatch):
    """Replaces `_command`: records every argv; `script.answer(argv)` is what it prints (default nothing),
    or the exception it raises."""

    class Script:
        def __init__(self):
            self.ran = []
            self.answer = lambda argv: ""

        def __call__(self, argv):
            self.ran.append(argv)
            result = self.answer(argv)
            if isinstance(result, Exception):
                raise result
            return result

    fake = Script()
    monkeypatch.setattr(skills_module, "_command", fake)
    return fake


@pytest.fixture
def apps(script):
    """`script`, with Finder, Google Chrome and Safari running."""
    script.answer = lambda argv: LSAPPINFO if argv[0] == "lsappinfo" else ""
    return script


def test_the_kinds_of_the_default_skills():
    skills = default_skills()

    assert {s.name: s.kind for s in skills.skills} == {
        "disk_space": "read", "battery_status": "read", "system_info": "read", "top_processes": "read",
        "find_files": "read", "open_app": "action", "quit_app": "confirm", "set_volume": "action",
        "volume_up": "action", "volume_down": "action", "mute_sound": "action"}
    assert [s.name for s in skills.skills] == [
        "disk_space", "battery_status", "system_info", "top_processes", "find_files", "open_app", "quit_app",
        "set_volume", "volume_up", "volume_down", "mute_sound"]
    assert [s.name for s in skills.skills if s.ask is not None] == ["quit_app"]


def test_every_skill_has_a_one_sentence_description_and_a_simple_schema():
    for s in default_skills().skills:
        assert s.description.endswith(".") and len(s.description) < 200, s.name
        assert s.parameters["type"] == "object"
        assert len(s.parameters["properties"]) <= 2, s.name


def test_parse_foreground_apps_keeps_only_apps_with_a_dock_icon():
    assert skills_module.parse_foreground_apps(LSAPPINFO) == ["Finder", "Google Chrome", "Safari"]
    assert skills_module.parse_foreground_apps("") == []


# -- prepare: the checks before a confirm skill runs


def test_prepare_is_nothing_for_skills_that_need_no_confirmation_and_for_unknown_ones():
    skills = default_skills()

    for name, arguments in [("disk_space", "{}"), ("open_app", '{"name": "Safari"}'), ("nope", "{}"),
                            ("open_app", "{not json")]:
        assert skills.prepare(name, arguments) is None, name


def test_prepare_returns_the_question_of_a_confirm_skill_and_quits_nothing(apps):
    asked = default_skills().prepare("quit_app", '{"name": "safari"}')

    assert asked == Question("Shall I quit Safari?")  # the running app's own name, whatever the model's case
    assert apps.ran == [["lsappinfo", "list"]]  # looked, did not quit


@pytest.mark.parametrize(("arguments", "reason"), [
    ("{not json", "its arguments are not valid JSON"),
    ("[]", "its arguments are not a JSON object"),
    ("{}", "the argument name is missing"),
    ('{"name": ""}', "no app name was given"),
    ('{"name": "Calculator"}', "Calculator is not running"),
])
def test_prepare_reports_what_is_wrong_instead_of_asking(apps, arguments, reason):
    result = default_skills().prepare("quit_app", arguments)

    assert result == SkillResult(f"the quit app skill failed: {reason}")
    assert not any(argv[0] == "osascript" for argv in apps.ran)


def test_prepare_never_raises_and_a_confirm_skill_without_a_question_is_a_failure():
    def broken(_args):
        raise ZeroDivisionError("oops")

    schema = {"type": "object", "properties": {}}
    skills = Skills([Skill("a", "A.", schema, lambda _: SkillResult("ran"), kind="confirm", ask=broken),
                     Skill("b", "B.", schema, lambda _: SkillResult("ran"), kind="confirm")])

    assert skills.prepare("a", "{}") == SkillResult("the a skill failed: ZeroDivisionError: oops")
    assert skills.prepare("b", "{}") == SkillResult("the b skill failed: it has no question to ask")


def test_run_runs_a_confirm_skill_too_because_its_caller_holds_the_approval():
    skills = Skills([Skill("a", "A.", {"type": "object", "properties": {}}, lambda _: SkillResult("ran"),
                           kind="confirm", ask=lambda _: "Shall I?")])
    assert skills.run("a", "{}") == SkillResult("ran")


# -- open_app


def test_open_app_opens_the_app_by_name_and_the_result_names_it(script):
    result = default_skills().run("open_app", '{"name": " Google Chrome "}')

    assert script.ran == [["open", "-a", "Google Chrome"]]
    assert result == SkillResult("opened Google Chrome")


@pytest.mark.parametrize("name", ["Safari", "Google Chrome", "AT&T Connect", "Notes 2", "Mail.app",
                                  "Dr. D'Arcy-Smith", "C++ Builder", "Ünï Çode"])
def test_open_app_accepts_ordinary_app_names(script, name):
    assert default_skills().run("open_app", json.dumps({"name": name})).speech == f"opened {name}"


@pytest.mark.parametrize("name", ["", "   ", "x" * 61, "-a", "--help", "Safari; rm -rf ~", "a/b", "../x", 'Say "hi"',
                                  "$(whoami)", "a`b`", "line\nbreak", "tab\tname", "a|b", "~", "null\x00"])
def test_open_app_refuses_names_that_are_not_app_names_and_runs_nothing(script, name):
    result = default_skills().run("open_app", json.dumps({"name": name}))

    assert result.speech.startswith("the open app skill failed: ")
    assert script.ran == []


def test_open_app_says_when_there_is_no_such_app(script):
    script.answer = lambda argv: SkillError("open failed with status 1: Unable to find application named 'Foo'")

    assert default_skills().run("open_app", '{"name": "Foo"}').speech == (
        "the open app skill failed: there is no app called Foo")


def test_open_app_reports_any_other_failure_of_open_as_it_is(script):
    script.answer = lambda argv: SkillError("open failed with status 1: something else")

    assert default_skills().run("open_app", '{"name": "Foo"}').speech == (
        "the open app skill failed: open failed with status 1: something else")


# -- quit_app


def test_quit_app_quits_gracefully_with_the_name_passed_as_an_argument_never_as_source(apps):
    result = default_skills().run("quit_app", '{"name": "google chrome"}')

    *_, quit_call = apps.ran
    assert quit_call == ["osascript", "-e", "on run argv", "-e", "tell application (item 1 of argv) to quit",
                         "-e", "end run", "Google Chrome"]
    assert not any("chrome" in part.lower() for part in quit_call[:-1])  # the source never holds the name
    assert result == SkillResult("quit Google Chrome")


def test_quit_app_does_not_launch_an_app_that_stopped_running_before_the_approval(apps):
    result = default_skills().run("quit_app", '{"name": "Calculator"}')

    assert result.speech == "the quit app skill failed: Calculator is not running"
    assert all(argv[0] == "lsappinfo" for argv in apps.ran)  # AppleScript's quit would have started it


@pytest.mark.parametrize("name", ["", "-a", 'Safari" to quit --', "a\nb", "x" * 61])
def test_quit_app_refuses_bad_names_before_looking_at_anything(apps, name):
    skills = default_skills()

    assert skills.prepare("quit_app", json.dumps({"name": name})).speech.startswith("the quit app skill failed: ")
    assert skills.run("quit_app", json.dumps({"name": name})).speech.startswith("the quit app skill failed: ")
    assert apps.ran == []


# -- set_volume


def test_set_volume_sets_the_level_and_the_result_says_which(script):
    result = default_skills().run("set_volume", '{"level": 40}')

    assert script.ran == [["osascript", "-e", "set volume output volume 40"]]
    assert result == SkillResult("volume set to 40 percent")


@pytest.mark.parametrize(("level", "expected"), [(0, 0), (100, 100), ("35", 35), (60.0, 60)])
def test_set_volume_takes_whole_numbers_in_range(script, level, expected):
    result = default_skills().run("set_volume", json.dumps({"level": level}))

    assert script.ran == [["osascript", "-e", f"set volume output volume {expected}"]]
    assert result.speech == f"volume set to {expected} percent"


@pytest.mark.parametrize("level", [101, -1, 12.5, "loud", "-5", True, [40], {"a": 1}, ""])
def test_set_volume_refuses_a_level_that_is_not_a_whole_number_from_0_to_100(script, level):
    result = default_skills().run("set_volume", json.dumps({"level": level}))

    assert result.speech.startswith("the set volume skill failed: ")
    assert script.ran == []


def test_set_volume_needs_a_level(script):
    result = default_skills().run("set_volume", "{}")

    assert result.speech == "the set volume skill failed: the argument level is missing"
    assert script.ran == []


@pytest.mark.parametrize(("name", "current", "expected"), [
    ("volume_up", 31, 41), ("volume_down", 31, 21), ("volume_up", 95, 100), ("volume_down", 5, 0),
    ("volume_up", 0, 10)])
def test_volume_up_and_down_move_ten_points_from_the_current_volume_clamped(script, name, current, expected):
    script.answer = lambda argv: f"{current}\n" if "get volume settings" in argv[-1] else ""

    result = default_skills().run(name, "{}")

    assert script.ran == [["osascript", "-e", "output volume of (get volume settings)"],
                          ["osascript", "-e", f"set volume output volume {expected}"]]
    assert result.speech == f"volume set to {expected} percent"


def test_volume_up_and_down_take_no_arguments():
    for name in ("volume_up", "volume_down"):
        (skill,) = [s for s in default_skills().skills if s.name == name]
        assert skill.parameters == {"type": "object", "properties": {}}


@pytest.mark.parametrize("name", ["volume_up", "volume_down"])
def test_volume_up_and_down_say_when_the_output_has_no_volume_to_read(script, name):
    script.answer = lambda argv: "missing value\n"

    result = default_skills().run(name, "{}")

    assert result.speech == f"the {name.replace('_', ' ')} skill failed: the volume of this output cannot be read"
    assert len(script.ran) == 1  # nothing was set


# -- mute_sound


@pytest.mark.parametrize(("arguments", "word", "speech"), [
    ("{}", "true", "sound muted"), ('{"muted": true}', "true", "sound muted"),
    ('{"muted": false}', "false", "sound unmuted"), ('{"muted": "False"}', "false", "sound unmuted"),
    ('{"muted": "true"}', "true", "sound muted")])
def test_mute_sound_mutes_by_default_and_unmutes_on_request(script, arguments, word, speech):
    result = default_skills().run("mute_sound", arguments)

    assert script.ran == [["osascript", "-e", f"set volume output muted {word}"]]
    assert result == SkillResult(speech)


@pytest.mark.parametrize("muted", [1, "maybe", None, [True]])
def test_mute_sound_refuses_anything_else(script, muted):
    result = default_skills().run("mute_sound", json.dumps({"muted": muted}))

    assert result.speech.startswith("the mute sound skill failed: muted must be true or false")
    assert script.ran == []


def test_web_search_is_not_a_skill():
    assert "web_search" not in default_skills().names  # measured: the 2B model searched for everything


def test_the_new_skills_say_so_off_a_mac(monkeypatch, script):
    monkeypatch.setattr(skills_module.sys, "platform", "linux")
    skills = default_skills()

    for name, arguments in [("open_app", '{"name": "Safari"}'), ("quit_app", '{"name": "Safari"}'),
                            ("set_volume", '{"level": 5}'), ("volume_up", "{}"), ("volume_down", "{}"),
                            ("mute_sound", "{}")]:
        assert "only available on macOS" in skills.run(name, arguments).speech, name
    assert "only available on macOS" in skills.prepare("quit_app", '{"name": "Safari"}').speech
    assert script.ran == []
