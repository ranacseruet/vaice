"""The pure parts of the browse skill: the definition, the goal, what is said, how a step is asked, the result."""
import re

import pytest

from vaice import browse, browse_task
from vaice.confirm import NO, YES, spoken_answer
from vaice.skills import SkillError, Skills, Started, default_skills


def test_the_skill_is_a_background_skill_the_brain_can_call_with_a_goal():
    skill = browse_task.browse_skill()

    assert skill.name == "browse" and skill.kind == "background"
    assert skill.parameters["required"] == ["goal"]
    assert Skills([skill]).definitions()[0]["function"]["name"] == "browse"


def test_it_is_offered_only_when_asked_for():
    assert "browse" not in default_skills().names
    assert "browse" in default_skills(browse=True).names


def test_prepare_returns_the_fixed_line_to_say_now_and_validates_the_goal():
    skills = Skills([browse_task.browse_skill()])

    assert skills.prepare("browse", '{"goal": "open python.org"}') == Started(browse_task.WORKING)
    for arguments in ("{}", '{"goal": ""}', '{"goal": "  "}', '{"goal": 3}', "not json", "[]"):
        prepared = skills.prepare("browse", arguments)
        assert not isinstance(prepared, Started), arguments  # a failed call, answered to the brain like any other


def test_the_goal_is_one_line_of_bounded_length():
    assert browse_task.goal_of('{"goal": "open\\npython.org\\t now"}') == "open python.org now"
    assert len(browse_task.goal_of({"goal": "x" * 5000})) == browse_task.GOAL_MAX_CHARS
    assert browse_task.goal_of({"goal": None}) == "" and browse_task.goal_of("[]") == ""


@pytest.mark.parametrize(("tool", "args", "said"), [
    ("mcp__chrome__navigate_page", {"url": "https://www.python.org/downloads"}, "Opening python.org."),
    ("mcp__chrome__new_page", {"url": "http://news.ycombinator.com"}, "Opening news.ycombinator.com."),
    ("mcp__chrome__navigate_page", {"url": "file:///etc/passwd"}, None),
    ("mcp__chrome__navigate_page", {"url": "https://"}, None),
    ("mcp__chrome__navigate_page", {"type": "reload"}, None),
    ("mcp__chrome__take_snapshot", {}, None),
    ("mcp__chrome__take_screenshot", {"url": "https://a.test"}, None),
    ("mcp__chrome__list_pages", {}, None),
    ("mcp__chrome__select_page", {"pageId": 1}, None),
    ("mcp__chrome__click", {"uid": "1_2"}, None)])
def test_only_opening_a_page_is_said(tool, args, said):
    assert browse_task.progress_phrase(tool, args) == said


def test_progress_is_at_most_one_line_per_interval():
    now = [10.0]
    limiter = browse_task.ProgressLimiter(8.0, clock=lambda: now[0])

    assert limiter.allow() is True  # the first is allowed at once
    now[0] += 7.9
    assert limiter.allow() is False
    now[0] += 0.1
    assert limiter.allow() is True and limiter.allow() is False


def test_every_gated_tool_has_a_spoken_name_and_none_holds_an_answer_word():
    assert set(browse._MUTATING_TOOLS) == set(browse_task.STEP_ACTIONS)  # every tool the gate asks about is named
    for action in (*browse_task.STEP_ACTIONS.values(), browse_task.OTHER_STEP):
        sentence = browse_task.step_question("mcp__chrome__click", None).replace("click something", action)
        words = set(re.findall(r"[a-z0-9]+", sentence.casefold()))
        assert not words & browse_task._ANSWER_WORDS, sentence


def test_the_question_never_says_an_answer_word_and_is_no_answer_itself():
    for host in (None, "python.org", "news.ycombinator.com", "stop.example.com", "yes.test", "go-ahead.dev"):
        for tool in ("mcp__chrome__click", "mcp__chrome__fill", "mcp__chrome__something_new"):
            text = browse_task.step_question(tool, host)
            words = set(re.findall(r"[a-z0-9]+", text.casefold()))
            assert not words & browse_task._ANSWER_WORDS, text
            assert spoken_answer(text) is None


def test_the_answer_words_cover_every_word_of_every_yes_and_no_phrase():
    assert {"yes", "no", "stop", "cancel", "go", "ahead"} <= browse_task._ANSWER_WORDS
    for phrase in YES | NO:
        assert set(re.sub(r"[^\w\s]", "", phrase.casefold()).split()) <= browse_task._ANSWER_WORDS


def test_the_question_names_the_action_and_the_host_but_never_the_arguments():
    text = browse_task.step_question("mcp__chrome__fill", "python.org")

    assert text == "Shall I fill in a field on python.org? The details are on the screen."
    assert browse_task.step_question("mcp__chrome__fill", None) == "Shall I fill in a field? " \
                                                                   "The details are on the screen."
    assert browse_task.step_question("mcp__chrome__unknown", None).startswith("Shall I take a step?")


def test_a_host_that_holds_an_answer_word_is_left_out_because_a_page_could_pick_it():
    assert browse_task.step_question("mcp__chrome__click", "stop.example.com") == \
        browse_task.step_question("mcp__chrome__click", None)


def test_the_result_is_one_clean_line_for_the_brain_and_keeps_the_whole_answer_as_detail():
    result = browse_task.result_of("**Welcome to Python.org**\n\nThe `title` is __this__.")

    assert result.speech == "the browser task finished: Welcome to Python.org The title is this."
    assert result.detail == "**Welcome to Python.org**\n\nThe `title` is __this__."


def test_a_long_answer_is_cut_for_the_brain_but_not_in_the_detail():
    result = browse_task.result_of("word " * 1000)

    assert result.speech.endswith(browse_task.CUT) and len(result.speech) < browse_task.RESULT_CHARS + 100
    assert len(result.detail) == 5000


def test_the_fixed_sentences_hold_no_answer_word():
    for text in (browse_task.WORKING, browse_task.FAILED):
        assert not set(re.findall(r"[a-z0-9]+", text.casefold())) & browse_task._ANSWER_WORDS, text


def test_a_background_call_without_a_goal_fails_as_a_skill_error():
    with pytest.raises(SkillError):
        browse_task.browse_skill().ask({})
