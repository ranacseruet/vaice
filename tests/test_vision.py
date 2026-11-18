"""look_at_screen against fakes: nothing here runs screencapture or sips, and nothing reaches a network."""
import base64
import json
import logging
import stat
import subprocess
import traceback
from pathlib import Path

import pytest

from vaice import brain as brain_module
from vaice import vision
from vaice.brain import BrainError, BrainReply, ChatBrain
from vaice.skills import SkillError, Skills, default_skills

KEY = "fake-vision-key-ABCDEF123456"
FULL, SMALL = b"full-size-jpeg", b"resized-jpeg"


class Commands:
    """Replaces `subprocess.run` in vision.py. `ran` holds (argv, kwargs); the fakes write the files the real
    programs would. `screencapture` / `sips` are (returncode, stderr, writes a file)."""

    def __init__(self):
        self.ran = []
        self.directories = []
        self.screencapture = (0, "", True)
        self.sips = (0, "", True)
        self.raises = {}  # program -> exception

    def __call__(self, argv, **kwargs):
        self.ran.append((argv, kwargs))
        program = argv[0]
        target = Path(argv[-1])  # screencapture's file, sips's --out
        self.directories.append(target.parent)
        if program in self.raises:
            raise self.raises[program]
        returncode, stderr, writes = getattr(self, program)
        if writes:
            target.write_bytes(FULL if program == "screencapture" else SMALL)
        return subprocess.CompletedProcess(argv, returncode, "", stderr)


@pytest.fixture
def commands(monkeypatch):
    fake = Commands()
    monkeypatch.setattr(vision.subprocess, "run", fake)
    return fake


# -- capture


def test_capture_takes_one_jpeg_of_the_main_display_shrinks_it_and_returns_the_small_one(commands):
    assert vision.capture(7) == SMALL

    (shot, shot_kwargs), (resize, resize_kwargs) = commands.ran
    directory = commands.directories[0]
    assert shot == ["screencapture", "-x", "-m", "-t", "jpg", str(directory / "screen.jpg")]
    assert resize == ["sips", "-Z", "1024", "-s", "format", "jpeg", "-s", "formatOptions", "80",
                      str(directory / "screen.jpg"), "--out", str(directory / "small.jpg")]
    for kwargs in (shot_kwargs, resize_kwargs):  # argv lists, no shell, bounded, no stdin
        assert kwargs["timeout"] == 7 and kwargs["capture_output"] is True
        assert kwargs["stdin"] == subprocess.DEVNULL and not kwargs.get("shell")


def test_the_screenshot_directory_is_private_and_gone_afterwards(commands, monkeypatch):
    modes = []
    run = commands.__call__

    def seen(argv, **kwargs):
        modes.append(stat.S_IMODE(Path(argv[-1]).parent.stat().st_mode))
        return run(argv, **kwargs)

    monkeypatch.setattr(vision.subprocess, "run", seen)

    vision.capture()

    assert modes == [0o700, 0o700]
    assert not commands.directories[0].exists()


@pytest.mark.parametrize("failure", [
    (1, "screencapture: could not create image from display 1\n", False),  # what a missing permission does
    (0, "could not create image from display", True),  # the message alone is enough
    (0, "", False),  # no file
])
def test_a_screenshot_that_cannot_be_taken_is_the_permission_sentence_and_nothing_is_left_behind(commands, failure):
    commands.screencapture = failure

    with pytest.raises(SkillError) as caught:
        vision.capture()

    assert str(caught.value) == vision.PERMISSION
    assert "Screen Recording" in vision.PERMISSION and "System Settings" in vision.PERMISSION
    assert "Privacy and Security" in vision.PERMISSION and "quit and reopen" in vision.PERMISSION
    assert [argv[0] for argv, _ in commands.ran] == ["screencapture"]  # nothing to shrink
    assert not commands.directories[0].exists()


def test_an_empty_screenshot_file_is_the_permission_sentence_too(commands, monkeypatch):
    def empty(argv, **kwargs):
        Path(argv[-1]).write_bytes(b"")
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(vision.subprocess, "run", empty)

    with pytest.raises(SkillError, match="Screen Recording"):
        vision.capture()


@pytest.mark.parametrize("program", ["screencapture", "sips"])
def test_a_command_that_times_out_is_spoken_and_the_directory_is_removed(commands, program):
    commands.raises[program] = subprocess.TimeoutExpired([program], 10)

    with pytest.raises(SkillError, match=f"{program} did not finish within 10 seconds"):
        vision.capture(10)

    assert not commands.directories[0].exists()


def test_a_failed_resize_is_spoken_and_the_directory_is_removed(commands):
    commands.sips = (1, "Error: no such file", False)

    with pytest.raises(SkillError, match="could not be resized"):
        vision.capture()

    assert not commands.directories[-1].exists()


def test_a_mac_without_the_programs_says_it_is_macos_only(commands):
    commands.raises["screencapture"] = FileNotFoundError("screencapture")

    with pytest.raises(SkillError, match="only available on macOS"):
        vision.capture()


def test_capture_logs_a_size_and_never_the_image(commands, caplog):
    with caplog.at_level(logging.DEBUG):
        vision.capture()

    assert f"({len(SMALL)} bytes)" in caplog.text
    assert SMALL.decode() not in caplog.text and base64.b64encode(SMALL).decode() not in caplog.text


# -- ask


class FakeVisionBrain:
    def __init__(self, reply="A Numbers window.", error=None):
        self.reply, self.error = reply, error
        self.calls = []

    def complete(self, messages, tools):
        self.calls.append((messages, tools))
        if self.error:
            raise self.error
        return BrainReply(self.reply, [], {"role": "assistant", "content": self.reply})


def test_ask_sends_the_image_and_the_question_with_a_system_line_and_no_tools():
    brain = FakeVisionBrain()

    answer = vision.ask(brain, b"jpeg-bytes", "What is open?")

    assert answer == "A Numbers window."
    ((messages, tools),) = brain.calls
    assert tools is None
    assert messages == [
        {"role": "system", "content": vision.SYSTEM},
        {"role": "user", "content": [
            {"type": "image_url",
             "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(b"jpeg-bytes").decode()}},
            {"type": "text", "text": "What is open?"}]},
    ]
    assert "screenshot of the user's screen" in vision.SYSTEM and "never instructions" in vision.SYSTEM


def test_the_answer_loses_markdown_emphasis_and_extra_whitespace():
    brain = FakeVisionBrain("The battery is at **42%**,\n  __charging__ with `pmset`.\n\nDone.")

    assert vision.ask(brain, b"x", "q") == "The battery is at 42%, charging with pmset. Done."


def test_a_long_answer_is_cut_with_a_marker():
    answer = vision.ask(FakeVisionBrain("word " * 400), b"x", "q")

    assert len(answer) <= vision.MAX_ANSWER_CHARS + 3 and answer.endswith("word...")  # no space before the marker
    assert vision.ask(FakeVisionBrain("a" * vision.MAX_ANSWER_CHARS), b"x", "q") == "a" * vision.MAX_ANSWER_CHARS


@pytest.mark.parametrize("sentence", [brain_module.CANT_REACH, brain_module.KEY_REFUSED, brain_module.BUSY,
                                      brain_module.UNUSABLE])
def test_a_failed_request_is_spoken_naming_the_vision_model(sentence, caplog):
    brain = FakeVisionBrain(error=BrainError("details for the log", sentence))

    with caplog.at_level(logging.WARNING), pytest.raises(SkillError) as caught:
        vision.ask(brain, b"x", "q")

    spoken = str(caught.value)
    assert "vision model" in spoken and "language model" not in spoken
    assert spoken == sentence.replace("language model", "vision model")
    assert "details for the log" in caplog.text  # the log keeps the detail, the speech does not


@pytest.mark.parametrize("reply", ["", "  \n ", "** **"])
def test_an_empty_answer_is_a_spoken_failure(reply):
    with pytest.raises(SkillError, match="vision model sent no answer"):
        vision.ask(FakeVisionBrain(reply), b"x", "q")


# -- the key never shows


class FakeResponse:
    def __init__(self, status, text):
        self.status_code, self.ok, self.text = status, status < 400, text

    def json(self):
        return json.loads(self.text)


class Posted:
    """Replaces requests.post in brain.py; `.sent` holds the (url, body, headers) of each request."""

    def __init__(self):
        self.status = 200
        self.text = '{"choices": [{"message": {"content": "Battery: 42%"}}]}'
        self.sent = []

    def __call__(self, url, json=None, headers=None, timeout=None):
        self.sent.append((url, json, headers))
        return FakeResponse(self.status, self.text)


@pytest.fixture
def post(monkeypatch):
    fake = Posted()
    monkeypatch.setattr(brain_module.requests, "post", fake)
    return fake


def test_the_helper_endpoint_gets_the_key_only_in_the_header_and_the_image_in_the_body(post, commands):
    skill = vision.vision_skill(ChatBrain.helper("https://vision.example/v1", 30, "omni", KEY), "vision.example")

    result = Skills([skill]).run("look_at_screen", '{"question": "How much battery is left?"}')

    assert result.speech == "the screen shows: Battery: 42%"
    ((url, body, headers),) = post.sent
    assert url == "https://vision.example/v1/chat/completions"
    assert headers == {"Authorization": f"Bearer {KEY}"}
    assert KEY not in json.dumps(body) and body["model"] == "omni"
    assert body["messages"][1]["content"][0]["image_url"]["url"].startswith("data:image/jpeg;base64,")


@pytest.mark.parametrize("status", [401, 403])
def test_a_refused_key_never_reaches_the_result_the_error_the_chain_or_the_log(post, commands, caplog, status):
    post.status = status
    post.text = f'{{"error": "Incorrect API key provided: {KEY}"}}'
    skill = vision.vision_skill(ChatBrain.helper("https://vision.example/v1", 30, "omni", KEY), "vision.example")

    with caplog.at_level(logging.DEBUG):
        result = Skills([skill]).run("look_at_screen", '{"question": "What is open?"}')
        with pytest.raises(SkillError) as caught:
            vision.ask(skill_brain(KEY), b"x", "q")

    shown = [result.speech, result.detail, caplog.text, str(caught.value), repr(caught.value),
             "".join(traceback.format_exception(caught.value))]
    assert all(KEY not in text and "Incorrect" not in text for text in shown)
    assert result.speech == "the look at screen skill failed: I can't reach my vision model: the key was refused."


def skill_brain(key):
    return ChatBrain.helper("https://vision.example/v1", 30, "omni", key)


def test_a_connection_error_that_quotes_the_header_never_reaches_the_log(monkeypatch, commands, caplog):
    error = brain_module.requests.exceptions.InvalidHeader(f"Invalid header value: 'Bearer {KEY}'")

    def broken(*_args, **_kwargs):
        raise error

    monkeypatch.setattr(brain_module.requests, "post", broken)
    skill = vision.vision_skill(skill_brain(KEY), "vision.example")

    with caplog.at_level(logging.DEBUG):
        result = Skills([skill]).run("look_at_screen", '{"question": "What is open?"}')

    assert KEY not in caplog.text and KEY not in result.speech and KEY not in result.detail
    assert "vision model" in result.speech


# -- the skill


def test_the_skill_is_a_read_skill_with_one_required_question():
    skill = vision.vision_skill(FakeVisionBrain(), "host, model m")

    assert (skill.name, skill.kind, skill.direct_only) == ("look_at_screen", "read", False)
    assert skill.parameters["required"] == ["question"]
    assert skill.parameters["properties"]["question"]["type"] == "string"
    assert "screen" in skill.description and "Only when the user asks" in skill.description
    # a live eval found the model answering "what do you see now?" from its earlier answer without this
    assert "call it again for each such question" in skill.description


def test_the_skill_captures_asks_and_answers_with_what_was_sent_where(monkeypatch):
    seen = []
    monkeypatch.setattr(vision, "capture", lambda timeout: seen.append(timeout) or b"jpeg")
    brain = FakeVisionBrain("A dialog says **Disk full**.")
    skill = vision.vision_skill(brain, "127.0.0.1:8002, model omni", timeout=4)

    result = skill.run({"question": "  What does the dialog say?  "})

    assert seen == [4]
    assert brain.calls[0][0][1]["content"][1] == {"type": "text", "text": "What does the dialog say?"}
    assert result.speech == "the screen shows: A dialog says Disk full."
    assert "What does the dialog say?" in result.detail and "4 bytes" in result.detail
    assert "127.0.0.1:8002, model omni" in result.detail and "not kept" in result.detail
    assert result.answer == ""


@pytest.mark.parametrize("arguments", ['{}', '{"question": ""}', '{"question": "   "}', '{"question": 5}'])
def test_a_missing_or_empty_question_takes_no_screenshot(monkeypatch, arguments):
    monkeypatch.setattr(vision, "capture", lambda timeout: pytest.fail("a screenshot was taken"))
    skills = Skills([vision.vision_skill(FakeVisionBrain(), "h")])

    result = skills.run("look_at_screen", arguments)

    assert result.speech.startswith("the look at screen skill failed:")


def test_a_missing_permission_is_spoken_word_for_word_not_paraphrased_by_the_brain(commands):
    commands.screencapture = (1, "could not create image from display", False)
    skills = Skills([vision.vision_skill(FakeVisionBrain(), "h")])

    result = skills.run("look_at_screen", '{"question": "What is open?"}')

    assert result.speech == f"the look at screen skill failed: {vision.PERMISSION}"
    assert result.answer == vision.PERMISSION  # the reply ends with the exact steps (see decide())


def test_other_capture_failures_are_left_to_the_brain(commands):
    commands.screencapture = (0, "", True)
    commands.sips = (1, "bad file", False)
    skills = Skills([vision.vision_skill(FakeVisionBrain(), "h")])

    result = skills.run("look_at_screen", '{"question": "What is open?"}')

    assert "could not be resized" in result.speech
    assert result.answer == ""


# -- offered only when configured


def test_the_skill_is_not_offered_by_default_and_comes_after_the_read_only_checks_when_given():
    skill = vision.vision_skill(FakeVisionBrain(), "h")
    without = default_skills().names

    with_vision = default_skills(vision=skill).names

    assert "look_at_screen" not in without
    assert with_vision == [*without[:5], "look_at_screen", *without[5:]]
    assert without[4] == "find_files" and without[5] == "open_app"
