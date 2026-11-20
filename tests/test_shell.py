"""The `run_shell` skill: which commands are refused (no process), and the runner against real, harmless
subprocesses (`/bin/echo`, `/bin/sleep`, `/bin/zsh -c`) run in a scratch home."""
import itertools
import json
import os
import subprocess
import time
from pathlib import Path

import pytest

from vaice import shell
from vaice.skills import Question, SkillError, SkillResult, default_skills


def never():
    return False


@pytest.fixture
def home(tmp_path, monkeypatch):
    """A scratch home: commands run there."""
    monkeypatch.setenv("HOME", str(tmp_path))
    return tmp_path


# --- check: what is refused before any question


@pytest.mark.parametrize("command", ["sudo ls", "ls; sudo ls", "ls | sudo tee x", "  SUDO ls", "echo 'sudo'",
                                     "/usr/bin/sudo ls", "ls && Sudo rm x", "ls\nsudo x"])
def test_sudo_is_refused_before_any_question(command):
    with pytest.raises(SkillError, match="administrator password"):
        shell.check(command)
    assert default_skills(shell=True).prepare("run_shell", json.dumps({"command": command})).speech.startswith(
        "the run shell skill failed: sudo needs an administrator password")


def test_a_word_that_only_contains_sudo_is_not_refused():
    assert shell.check("ls pseudo") == "ls pseudo"  # no word boundary inside "pseudo"


@pytest.mark.parametrize("command", ["", "   ", "\n"])
def test_an_empty_command_is_refused(command):
    with pytest.raises(SkillError, match="no command was given"):
        shell.check(command)


def test_a_command_over_the_limit_is_refused_and_one_at_the_limit_is_not():
    assert shell.check("ls " + "a" * (shell.MAX_COMMAND - 3))
    with pytest.raises(SkillError, match="longer than 500 characters"):
        shell.check("ls " + "a" * (shell.MAX_COMMAND - 2))


@pytest.mark.parametrize("command", [None, 5, ["ls"]])
def test_a_command_that_is_not_text_is_refused(command):
    with pytest.raises(SkillError, match="must be text"):
        shell.check(command)


def test_a_null_character_is_refused():
    with pytest.raises(SkillError, match="null character"):
        shell.check("echo a\0b")


# --- the skill: the question and the arguments


def prepared(command):
    return default_skills(shell=True).prepare("run_shell", json.dumps({"command": command}))


def test_every_command_is_asked_with_the_fixed_sentence_and_never_names_the_command():
    question = prepared("cat ~/.ssh/config")

    assert question == Question(shell.ASK)
    assert "cat" not in question.text.split() and ".ssh" not in question.text
    assert prepared("ls") == question  # even a harmless one: the button is the only approval


@pytest.mark.parametrize("arguments", ["{}", '{"command": 5}', '{"command": ""}', '{"command": null}'])
def test_a_missing_or_bad_command_is_a_failed_call_and_asks_nothing(arguments):
    result = default_skills(shell=True).prepare("run_shell", arguments)

    assert result.speech.startswith("the run shell skill failed: ")


def test_the_skill_is_a_confirm_skill_with_a_narrow_description_and_one_required_argument():
    (skill,) = [s for s in default_skills(shell=True).skills if s.name == "run_shell"]

    assert skill.kind == "confirm" and skill.parameters["required"] == ["command"]
    assert set(skill.parameters["properties"]) == {"command"}
    assert "The user must approve it first." in skill.description


def test_the_skill_exists_only_with_the_flag():
    assert "run_shell" not in default_skills().names
    assert "run_shell" not in default_skills(shell=False).names
    assert default_skills(shell=True).names == [*default_skills().names, "run_shell"]


# --- run: real subprocesses


def test_a_command_runs_through_zsh_in_the_home_directory(home, monkeypatch):
    started = []
    real_popen = subprocess.Popen

    def spy(argv, **kwargs):
        started.append((argv, kwargs))
        return real_popen(argv, **kwargs)

    monkeypatch.setattr(shell.subprocess, "Popen", spy)

    result = shell.run("pwd", never)

    assert started[0][0] == ["/bin/zsh", "-c", "pwd"]
    kwargs = started[0][1]
    assert kwargs["start_new_session"] is True and kwargs["stdin"] == subprocess.DEVNULL
    assert kwargs["stderr"] == subprocess.STDOUT and Path(kwargs["cwd"]) == home
    assert result.speech == f"the command finished. Output: {home.resolve()}"
    assert result.detail == f"$ pwd\n(the command finished)\n{home.resolve()}"


def test_stdout_and_stderr_are_merged_and_the_output_is_collapsed_for_the_model(home):
    result = shell.run("echo a; echo b >&2; printf 'c\\n\\n  d'", never)

    assert result.speech == "the command finished. Output: a b c d"
    assert result.detail.endswith("a\nb\nc\n\n  d")  # the screen gets it raw


def test_a_command_with_no_output_says_only_how_it_ended(home):
    assert shell.run("true; true", never).speech == "the command finished"


def test_a_failing_command_reports_its_status_and_its_output(home):
    result = shell.run("echo oops >&2; exit 3", never)

    assert result.speech == "the command failed with status 3. Output: oops"


def test_stdin_is_empty(home):
    assert shell.run("cat; echo done", never).speech == "the command finished. Output: done"


def test_a_missing_shell_is_a_spoken_failure_not_a_crash(home, monkeypatch):
    def missing(*_args, **_kwargs):
        raise FileNotFoundError(2, "No such file")

    monkeypatch.setattr(shell.subprocess, "Popen", missing)

    with pytest.raises(SkillError, match="/bin/zsh is not available"):
        shell.run("pwd", never)
    assert default_skills(shell=True).run("run_shell", '{"command": "pwd"}').speech == (
        "the run shell skill failed: /bin/zsh is not available")


def test_a_command_that_cannot_be_started_for_another_reason_is_a_spoken_failure(home, monkeypatch):
    def refuse(*_args, **_kwargs):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(shell.subprocess, "Popen", refuse)

    with pytest.raises(SkillError, match="could not be started: Permission denied"):
        shell.run("pwd", never)


def test_run_checks_again_so_a_refused_command_never_starts(monkeypatch):
    def boom(*_args, **_kwargs):
        raise AssertionError("a refused command was started")

    monkeypatch.setattr(shell.subprocess, "Popen", boom)

    with pytest.raises(SkillError, match="administrator password"):
        shell.run("sudo ls", never)
    with pytest.raises(SkillError, match="no command"):
        shell.run("", never)


def test_the_output_given_to_the_model_is_cut_at_600_characters_and_says_so(home):
    result = shell.run("printf 'x%.0s' {1..700}", never)

    assert result.speech == f"the command finished. Output: {'x' * 600} (more on screen)"
    assert result.detail.endswith("x" * 700)


def test_the_output_kept_is_capped_and_a_command_that_prints_megabytes_does_not_hang(home):
    started = time.monotonic()
    result = shell.run("head -c 5000000 /dev/zero | tr '\\0' 'y'", never)

    assert time.monotonic() - started < 10
    assert result.speech.startswith("the command finished. Output: yyyy")
    assert result.speech.endswith("(more on screen)")
    assert result.detail.split("\n", 2)[2] == "y" * shell.SHELL_OUTPUT_CHARS


def test_output_exactly_at_the_cap_is_not_marked_as_cut(home, monkeypatch):
    monkeypatch.setattr(shell, "SHELL_OUTPUT_CHARS", 10)

    assert shell.run("printf 'abcdefghij'", never).speech == "the command finished. Output: abcdefghij"
    cut = shell.run("printf 'abcdefghijk'", never)
    assert cut.speech == "the command finished. Output: abcdefghij (more on screen)"


def test_output_that_is_not_utf8_is_replaced_not_raised(home):
    result = shell.run("printf 'a\\xffb'", never)

    assert result.speech == "the command finished. Output: a�b"


def test_a_command_past_the_time_limit_is_killed_and_says_so(home, monkeypatch):
    monkeypatch.setattr(shell, "SHELL_TIMEOUT", 0.4)
    started = time.monotonic()

    result = shell.run("echo before; sleep 30", never)

    assert time.monotonic() - started < 5
    assert result.speech == "the command was stopped after 0.4 seconds. Output: before"


def test_the_timeout_is_30_seconds_by_default():
    assert shell.SHELL_TIMEOUT == 30 and shell.SHELL_OUTPUT_CHARS == 20000


def alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def wait_gone(pid, seconds=5.0):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if not alive(pid):
            return True
        time.sleep(0.02)
    return False


def test_stop_ends_the_whole_process_group_not_just_the_shell(home):
    calls = [0]

    def stop():
        calls[0] += 1
        return calls[0] > 6  # the command has printed its child's pid by then

    started = time.monotonic()
    result = shell.run("sleep 30 & echo $!; sleep 30", stop)

    assert time.monotonic() - started < 5
    assert result.speech.startswith("the command was stopped. Output: ")
    child = int(result.speech.rsplit(" ", 1)[1])
    assert wait_gone(child), "the background child survived the stop"


def test_a_child_that_ignores_sigterm_is_killed_too(home, monkeypatch):
    monkeypatch.setattr(shell, "KILL_WAIT", 0.3)
    calls = [0]

    def stop():
        calls[0] += 1
        return calls[0] > 6

    result = shell.run("(trap '' TERM; sleep 30) & echo $!; trap '' TERM; sleep 30", stop)

    assert wait_gone(int(result.speech.rsplit(" ", 1)[1]))


def test_stop_is_polled_often_even_while_the_command_floods_its_output(home):
    seen = []

    def stop():
        seen.append(time.monotonic())
        return len(seen) > 5

    started = time.monotonic()
    result = shell.run("yes", stop)  # `yes` never ends and never stops writing

    assert time.monotonic() - started < 5
    assert result.speech.startswith("the command was stopped. Output: y y y")
    assert max(b - a for a, b in itertools.pairwise(seen)) < 0.2


def test_a_background_process_holding_the_output_open_does_not_hang_the_command(home, monkeypatch):
    monkeypatch.setattr(shell, "LEFTOVER_WAIT", 0.2)
    started = time.monotonic()

    result = shell.run("sleep 30 & echo $!", never)  # the shell exits at once, its child keeps the pipe

    assert time.monotonic() - started < 5
    assert result.speech.startswith("the command finished. Output: ")
    assert wait_gone(int(result.speech.rsplit(" ", 1)[1]))  # ended with the command


def test_a_background_process_that_closed_the_output_ends_with_the_command_too(home):
    result = shell.run("sleep 30 >/dev/null 2>&1 & echo $!", never)  # the pipe reaches EOF, the child lives on

    assert result.speech.startswith("the command finished. Output: ")
    assert wait_gone(int(result.speech.rsplit(" ", 1)[1]))


def test_a_command_stopped_before_it_started_is_never_started(home, tmp_path):
    marker = tmp_path / "ran"

    result = shell.run(f"touch {marker}", lambda: True)

    assert result == SkillResult("the command was stopped", f"$ touch {marker}\n(the command was stopped)")
    assert not marker.exists()


def test_the_registry_is_empty_after_a_run(home):
    shell.run("true", never)

    assert shell._live == set()


def test_the_exit_hook_ends_a_command_that_is_still_running(home):
    proc = subprocess.Popen(["sleep", "30"], start_new_session=True)
    with shell._live_lock:
        shell._live.add(proc)
    try:
        shell._stop_all()

        assert proc.poll() is not None
    finally:
        with shell._live_lock:
            shell._live.discard(proc)
        if proc.poll() is None:
            proc.kill()


def test_stopping_a_group_that_is_already_gone_is_quiet(home):
    proc = subprocess.Popen(["true"], start_new_session=True)
    proc.wait()

    shell._stop_group(proc)  # nothing to signal: no error


def test_a_command_that_survives_sigkill_is_logged_not_waited_for(home, monkeypatch, caplog):
    class Stuck:
        pid = 2**22 + 12345  # no such group

        def wait(self, timeout=None):
            raise subprocess.TimeoutExpired("x", timeout)

    monkeypatch.setattr(shell, "KILL_WAIT", 0.01)

    shell._stop_group(Stuck())

    assert "did not end after SIGKILL" in caplog.text


def test_an_error_while_a_command_runs_never_leaves_it_running(home, monkeypatch):
    started = []
    real_popen = subprocess.Popen
    monkeypatch.setattr(shell.subprocess, "Popen",
                        lambda argv, **kw: started.append(real_popen(argv, **kw)) or started[-1])

    def broken(*_args):
        raise RuntimeError("select failed")

    monkeypatch.setattr(shell.select, "select", broken)

    with pytest.raises(RuntimeError, match="select failed"):
        shell.run("sleep 30", never)

    assert started[0].poll() is not None


def test_a_skill_run_by_name_gets_the_stop_callable(home):
    skills = default_skills(shell=True)

    result = skills.run("run_shell", '{"command": "sleep 30"}', lambda: True)

    assert result.speech == "the command was stopped"
    assert skills.run("run_shell", '{"command": "echo hi"}').speech == "the command finished. Output: hi"


def test_the_skills_own_run_is_the_unstoppable_fallback(home):
    (skill,) = [s for s in default_skills(shell=True).skills if s.name == "run_shell"]

    assert skill.run({"command": "echo hi"}).speech == "the command finished. Output: hi"
