"""Skills against sample command output and fake commands: nothing here runs pmset, ps, mdfind or sysctl."""
import json
import logging
import subprocess
from collections import namedtuple
from pathlib import Path

import pytest

from vaice import skills as skills_module
from vaice.skills import (
    Skill,
    SkillError,
    SkillResult,
    Skills,
    default_skills,
    describe_disk,
    describe_files,
    describe_processes,
    describe_system,
    parse_battery,
    parse_boot_time,
    parse_memory,
    parse_processes,
    spoken_duration,
)

GIB = 1024**3


@pytest.fixture(autouse=True)
def on_a_mac(monkeypatch):
    monkeypatch.setattr(skills_module.sys, "platform", "darwin")


@pytest.fixture
def commands(monkeypatch):
    """Replaces `_command`: `commands.output[argv[0]]` is what each command prints; `commands.ran` records argv."""

    class Commands:
        def __init__(self):
            self.output = {}
            self.ran = []

        def __call__(self, argv):
            self.ran.append(argv)
            result = self.output[argv[0]]
            if isinstance(result, Exception):
                raise result
            return result

    fake = Commands()
    monkeypatch.setattr(skills_module, "_command", fake)
    return fake


# -- parsers

PMSET_DISCHARGING = """Now drawing from 'Battery Power'
 -InternalBattery-0 (id=24379491)\t78%; discharging; 4:12 remaining present: true
"""
PMSET_CHARGING = """Now drawing from 'AC Power'
 -InternalBattery-0 (id=24379491)\t45%; charging; 1:05 remaining present: true
"""
PMSET_CHARGED = """Now drawing from 'AC Power'
 -InternalBattery-0 (id=24379491)\t100%; charged; 0:00 remaining present: true
"""
PMSET_PLUGGED_NOT_CHARGING = """Now drawing from 'AC Power'
 -InternalBattery-0 (id=24379491)\t80%; AC attached; not charging present: true
"""
PMSET_NO_ESTIMATE = """Now drawing from 'Battery Power'
 -InternalBattery-0 (id=24379491)\t62%; discharging; (no estimate) present: true
"""
PMSET_DESKTOP = "Now drawing from 'AC Power'\n"


@pytest.mark.parametrize(("output", "spoken"), [
    (PMSET_DISCHARGING, "battery at 78 percent, running on battery power, about 4 hours and 12 minutes left"),
    (PMSET_CHARGING, "battery at 45 percent, charging, about 1 hour and 5 minutes until full"),
    (PMSET_CHARGED, "battery at 100 percent, fully charged and plugged in"),
    (PMSET_PLUGGED_NOT_CHARGING, "battery at 80 percent, plugged in but not charging"),
    (PMSET_NO_ESTIMATE, "battery at 62 percent, running on battery power"),
    (PMSET_DESKTOP, "this Mac has no battery"),
    ("", "no battery was found"),
    ("\t55%; finishing charge; 0:08 remaining present: true",
     "battery at 55 percent, almost fully charged, about 8 minutes until full"),
    ("\t55%; charged; 0:00 remaining present: true", "battery at 55 percent, fully charged"),
    ("\t55%; ; present: true\nNow drawing from 'AC Power'", "battery at 55 percent, plugged in"),
    ("\t55%; mystery; present: true", "battery at 55 percent, running on battery power"),
])
def test_parse_battery(output, spoken):
    assert parse_battery(output) == spoken


def test_parse_boot_time_reads_the_seconds():
    assert parse_boot_time("{ sec = 1731000000, usec = 123456 } Fri Nov  8 00:00:00 2024\n") == 1731000000
    with pytest.raises(SkillError, match=r"unexpected kern\.boottime"):
        parse_boot_time("garbage")


def test_parse_memory_reads_bytes():
    assert parse_memory("51539607552\n") == 48 * GIB
    with pytest.raises(SkillError, match=r"unexpected hw\.memsize"):
        parse_memory("lots")


@pytest.mark.parametrize(("seconds", "spoken"), [
    (0, "less than a minute"),
    (59, "less than a minute"),
    (60, "1 minute"),
    (3600, "1 hour"),
    (4 * 3600 + 12 * 60, "4 hours and 12 minutes"),
    (86400, "1 day"),
    (3 * 86400 + 4 * 3600 + 5 * 60, "3 days and 4 hours"),  # two units at most
    (2 * 86400 + 30 * 60, "2 days and 30 minutes"),  # a zero unit in the middle is skipped
])
def test_spoken_duration(seconds, spoken):
    assert spoken_duration(seconds) == spoken


def test_describe_system():
    now = 1731000000 + 3 * 86400 + 4 * 3600
    assert describe_system("15.6", "Ranas-MacBook.local", 1731000000, 48 * GIB, now) == (
        "macOS 15.6 on Ranas-MacBook, up 3 days and 4 hours, 48 gigabytes of memory")
    assert "up less than a minute" in describe_system("15.6", "Mac", now + 10, 8 * GIB, now)  # a clock step back


def test_describe_disk_uses_whole_binary_gigabytes():
    assert describe_disk(926 * GIB, 160 * GIB) == "160 gigabytes free of 926 gigabytes"
    assert describe_disk(int(926.4 * GIB), int(159.6 * GIB)) == "160 gigabytes free of 926 gigabytes"


PS_OUTPUT = """  %CPU %MEM COMM
  12.3  9.8 Google Chrome
   4.0  4.5 Slack
   0.0  0.1 launchd
   9.1  2.0 Google Chrome Helper
   2.2  3.9 Xcode
broken line
  x.y  1.0 Bad
"""


def test_parse_processes_skips_the_header_and_keeps_names_with_spaces():
    assert parse_processes(PS_OUTPUT) == [
        (12.3, 9.8, "Google Chrome"), (4.0, 4.5, "Slack"), (0.0, 0.1, "launchd"),
        (9.1, 2.0, "Google Chrome Helper"), (2.2, 3.9, "Xcode")]


def test_describe_processes_by_memory_and_by_cpu():
    rows = parse_processes(PS_OUTPUT)
    assert describe_processes(rows, "memory", 3) == (
        "Google Chrome 10 percent of memory, Slack 4 percent, Xcode 4 percent")
    assert describe_processes(rows, "cpu", 2) == (
        "Google Chrome 12 percent of the processor, Google Chrome Helper 9 percent")
    assert describe_processes(rows, "memory", 1) == "Google Chrome 10 percent of memory"
    assert describe_processes([], "memory", 3) == "no processes were listed"


def test_describe_processes_adds_up_helpers_that_share_a_name():
    rows = [(1.0, 3.0, "Chrome Helper"), (1.0, 3.0, "Chrome Helper"), (1.0, 4.0, "Slack")]
    assert describe_processes(rows, "memory", 2) == "Chrome Helper 6 percent of memory, Slack 4 percent"


def test_describe_files_without_results():
    assert describe_files("") == SkillResult("no files found")
    assert describe_files("\n  \n") == SkillResult("no files found")


def test_describe_files_names_up_to_three_files_and_keeps_the_paths_in_the_detail():
    paths = ["/Users/x/Documents/report.pdf", "/Users/x/Downloads/report-final.pdf",
             "/Users/x/Desktop/old report.docx"]

    result = describe_files("\n".join(paths) + "\n")

    assert result.speech == "3 files: report.pdf, report-final.pdf, old report.docx"
    assert result.detail == "\n".join(paths)
    assert describe_files(paths[0]).speech == "1 file: report.pdf"
    assert describe_files("\n".join(paths[:2])).speech == "2 files: report.pdf, report-final.pdf"


def test_describe_files_with_more_than_three_names_three_and_says_how_many():
    paths = [f"/Users/x/Documents/note {i}.txt" for i in range(7)]
    result = describe_files("\n".join(paths))
    assert result.speech == "7 files, including note 0.txt, note 1.txt, note 2.txt"
    assert result.detail == "\n".join(paths)


def test_describe_files_caps_the_list_at_twenty():
    paths = [f"/Users/x/f{i}.txt" for i in range(35)]
    result = describe_files("\n".join(paths))
    assert result.speech == "more than 20 files, including f0.txt, f1.txt, f2.txt"
    assert result.detail.splitlines() == paths[:20]
    assert describe_files("\n".join(paths[:20])).speech.startswith("20 files, including")


# -- Skills.run never raises


def skill(run, name="demo", required=()):
    schema = {"type": "object", "properties": {"x": {"type": "string"}}, "required": list(required)}
    return Skill(name, "Do the demo.", schema, run)


def test_definitions_use_the_chat_completions_tool_shape():
    definitions = default_skills().definitions()

    assert [d["function"]["name"] for d in definitions] == [
        "disk_space", "battery_status", "system_info", "top_processes", "find_files"]
    for definition in definitions:
        assert definition["type"] == "function"
        assert definition["function"]["description"].endswith(".")
        assert definition["function"]["parameters"]["type"] == "object"
    by_name = {d["function"]["name"]: d["function"]["parameters"] for d in definitions}
    assert by_name["find_files"]["required"] == ["name"]
    assert by_name["find_files"]["properties"]["folder"]["enum"] == ["home", "desktop", "documents", "downloads"]
    assert by_name["top_processes"]["properties"]["by"]["enum"] == ["memory", "cpu"]
    json.dumps(definitions)  # serializable as sent


def test_a_skill_runs_with_its_parsed_arguments():
    got = []
    skills = Skills([skill(lambda args: got.append(args) or SkillResult("done", "details"))])

    assert skills.run("demo", '{"x": "1"}') == SkillResult("done", "details")
    assert skills.run("demo", "") == SkillResult("done", "details")  # no arguments is {}
    assert got == [{"x": "1"}, {}]


def test_an_unknown_skill_is_a_result_not_an_error():
    assert Skills([]).run("nope", "{}") == SkillResult("there is no skill called nope")


@pytest.mark.parametrize(("arguments", "reason"), [
    ("{not json", "its arguments are not valid JSON"),
    ("[1, 2]", "its arguments are not a JSON object"),
    ('"text"', "its arguments are not a JSON object"),
    ("{}", "the argument x is missing"),
])
def test_bad_arguments_come_back_as_plain_words(arguments, reason):
    ran = []
    skills = Skills([skill(lambda args: ran.append(args), required=["x"])])

    result = skills.run("demo", arguments)

    assert result == SkillResult(f"the demo skill failed: {reason}")
    assert ran == []


def test_a_skill_error_is_spoken_and_logged_without_a_traceback(caplog):
    def run(_):
        raise SkillError("pmset is not available")

    with caplog.at_level(logging.WARNING, logger="vaice.skills"):
        result = Skills([skill(run, name="battery_status")]).run("battery_status", "{}")

    assert result.speech == "the battery status skill failed: pmset is not available"
    assert "pmset is not available" in caplog.text
    assert not any(r.exc_info for r in caplog.records)


def test_an_unexpected_exception_is_spoken_and_logged_with_its_traceback(caplog):
    def run(_):
        raise ZeroDivisionError("division by zero")

    with caplog.at_level(logging.ERROR, logger="vaice.skills"):
        result = Skills([skill(run)]).run("demo", "{}")

    assert result.speech == "the demo skill failed: ZeroDivisionError: division by zero"
    assert any(r.exc_info for r in caplog.records)


# -- the commands

Completed = namedtuple("Completed", "returncode stdout stderr")


def test_commands_run_as_an_argv_list_without_a_shell_or_stdin_and_with_a_timeout(monkeypatch):
    seen = {}

    def run(argv, **kwargs):
        seen.update(argv=argv, **kwargs)
        return Completed(0, "out", "")

    monkeypatch.setattr(skills_module.subprocess, "run", run)

    assert skills_module._command(["pmset", "-g", "batt"]) == "out"
    assert seen["argv"] == ["pmset", "-g", "batt"]
    assert seen["stdin"] == subprocess.DEVNULL
    assert seen["timeout"] == 10
    assert seen["capture_output"] is True
    assert not seen.get("shell")


def test_a_missing_command_a_failing_command_and_a_timeout_are_skill_errors(monkeypatch):
    def run_missing(argv, **kwargs):
        raise FileNotFoundError(argv[0])

    monkeypatch.setattr(skills_module.subprocess, "run", run_missing)
    with pytest.raises(SkillError, match="pmset is not available"):
        skills_module._command(["pmset"])

    monkeypatch.setattr(skills_module.subprocess, "run", lambda argv, **kw: Completed(2, "", " bad flag \n"))
    with pytest.raises(SkillError, match="ps failed with status 2: bad flag"):
        skills_module._command(["ps"])

    def run_slow(argv, **kwargs):
        raise subprocess.TimeoutExpired(argv, kwargs["timeout"])

    monkeypatch.setattr(skills_module.subprocess, "run", run_slow)
    with pytest.raises(SkillError, match="mdfind did not answer within 10 seconds"):
        skills_module._command(["mdfind"])


def test_a_hung_command_comes_back_as_a_spoken_failure_through_the_skill(monkeypatch):
    def run_slow(argv, **kwargs):
        raise subprocess.TimeoutExpired(argv, kwargs["timeout"])

    monkeypatch.setattr(skills_module.subprocess, "run", run_slow)

    result = default_skills().run("battery_status", "{}")

    assert result.speech == "the battery status skill failed: pmset did not answer within 10 seconds"


def test_every_skill_but_disk_space_says_so_off_a_mac(monkeypatch):
    monkeypatch.setattr(skills_module.sys, "platform", "linux")
    skills = default_skills()

    for name, arguments in [("battery_status", "{}"), ("system_info", "{}"), ("top_processes", "{}"),
                            ("find_files", '{"name": "x"}')]:
        assert "only available on macOS" in skills.run(name, arguments).speech, name


def test_disk_space(monkeypatch):
    usage = namedtuple("usage", "total used free")
    monkeypatch.setattr(skills_module.shutil, "disk_usage", lambda path: usage(926 * GIB, 766 * GIB, 160 * GIB))

    assert default_skills().run("disk_space", "{}") == SkillResult("160 gigabytes free of 926 gigabytes")


def test_battery_status(commands):
    commands.output["pmset"] = PMSET_DISCHARGING

    result = default_skills().run("battery_status", "{}")

    assert commands.ran == [["pmset", "-g", "batt"]]
    assert result.speech == "battery at 78 percent, running on battery power, about 4 hours and 12 minutes left"
    assert result.detail == PMSET_DISCHARGING.strip()


def test_system_info(commands, monkeypatch):
    answers = {"kern.boottime": "{ sec = 1731000000, usec = 0 } Fri Nov  8 2024", "hw.memsize": "51539607552\n"}
    monkeypatch.setattr(skills_module, "_command", lambda argv: answers[argv[-1]])
    monkeypatch.setattr(skills_module.platform, "mac_ver", lambda: ("15.6", ("", "", ""), "arm64"))
    monkeypatch.setattr(skills_module.platform, "node", lambda: "Ranas-MacBook.local")
    monkeypatch.setattr(skills_module.time, "time", lambda: 1731000000 + 3 * 86400 + 4 * 3600)

    result = default_skills().run("system_info", "{}")

    assert result.speech == "macOS 15.6 on Ranas-MacBook, up 3 days and 4 hours, 48 gigabytes of memory"


def test_top_processes_ranks_by_memory_by_default_and_by_cpu_on_request(commands):
    commands.output["ps"] = PS_OUTPUT
    skills = default_skills()

    by_memory = skills.run("top_processes", "{}")
    by_cpu = skills.run("top_processes", '{"by": "CPU", "count": 2}')

    assert commands.ran == [["ps", "-Aceo", "pcpu,pmem,comm", "-m"], ["ps", "-Aceo", "pcpu,pmem,comm", "-r"]]
    assert by_memory.speech == "Google Chrome 10 percent of memory, Slack 4 percent, Xcode 4 percent"
    assert by_cpu.speech == "Google Chrome 12 percent of the processor, Google Chrome Helper 9 percent"
    assert "12.3% cpu  9.8% mem  Google Chrome" in by_memory.detail


def test_top_processes_clamps_the_count_and_refuses_what_it_cannot_read(commands):
    commands.output["ps"] = PS_OUTPUT
    skills = default_skills()

    assert skills.run("top_processes", '{"count": 99}').speech.count("percent") == 5  # capped at 5
    assert skills.run("top_processes", '{"count": 0}').speech.count("percent") == 1
    assert "by must be memory or cpu" in skills.run("top_processes", '{"by": "disk"}').speech
    assert "count must be a whole number" in skills.run("top_processes", '{"count": "many"}').speech


def test_find_files_searches_the_chosen_folder_by_name(commands):
    commands.output["mdfind"] = "/Users/x/Downloads/report.pdf\n/Users/x/Downloads/old report.docx\n"
    skills = default_skills()

    result = skills.run("find_files", '{"name": " report ", "folder": "Downloads"}')
    skills.run("find_files", '{"name": "report"}')

    assert commands.ran[0] == ["mdfind", "-name", "report", "-onlyin", str(Path.home() / "Downloads")]
    assert commands.ran[1][-1] == str(Path.home())  # home is the default
    assert result.speech == "2 files: report.pdf, old report.docx"
    assert "/Users/x/Downloads/report.pdf" in result.detail


@pytest.mark.parametrize("folder", ["desktop", "documents"])
def test_find_files_knows_the_other_folders(commands, folder):
    commands.output["mdfind"] = ""

    result = default_skills().run("find_files", json.dumps({"name": "x", "folder": folder}))

    assert commands.ran[0][-1] == str(Path.home() / folder.capitalize())
    assert result.speech == "no files found"


def test_find_files_refuses_a_made_up_folder_and_an_empty_name(commands):
    skills = default_skills()

    assert "folder must be one of home, desktop, documents, downloads" in skills.run(
        "find_files", '{"name": "x", "folder": "/etc"}').speech
    assert "no file name to look for" in skills.run("find_files", '{"name": "  "}').speech
    assert commands.ran == []  # nothing ran for either
