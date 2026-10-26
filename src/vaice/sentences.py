"""Group a stream of text fragments into complete sentences."""
import re
from collections.abc import Iterable, Iterator

# Sentence ends at . ! ? followed by whitespace, so "3.5" is not split. A line break also ends one
# (list items often have no final punctuation).
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+|\n+")
# Periods that never end a sentence: titles before a name, initials ("U.S.", "e.g."), and a number
# that would otherwise be spoken on its own (a list item, or "fig. 3.")
_NEVER_AN_END = re.compile(r"(?:\b(?:mr|mrs|ms|dr|prof|st|vs)\.|\b(?:[a-z]\.)+|^\s*\d+\.)\Z", re.IGNORECASE)
# "etc." ends a sentence only when the next word is capitalized: "dogs, etc. What about you?"
_END_IF_CAPITAL = re.compile(r"\betc\.\Z", re.IGNORECASE)
# "a.m."/"p.m." end a sentence only when the next word is capitalized: "at 7 a.m. and then" is one
# sentence, "at 3 p.m. Tomorrow" is two. (Without this they never end one, so the whole reply was
# spoken as a single TTS utterance and the first audio only came at the end of it.)
_AM_PM = re.compile(r"\b(?:a|p)\.m\.\Z", re.IGNORECASE)
# Markdown a TTS engine would read aloud literally
_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_BULLET = re.compile(r"^\s*[-•]\s+", re.MULTILINE)
_MARKDOWN = re.compile(r"[*_#`~]")


def split_sentences(fragments: Iterable[str]) -> Iterator[str]:
    """Yield each sentence as soon as the next one starts, then the remainder."""
    buffer = ""
    for fragment in fragments:
        buffer += fragment
        start = 0
        for end in _SENTENCE_END.finditer(buffer):
            sentence = buffer[start : end.start()]
            if _AM_PM.search(sentence):
                # Ends the sentence only before a capitalized word, like "etc.": "at 3 p.m. Tomorrow"
                # is two sentences, "at 7 a.m. and then" is one.
                if end.end() == len(buffer):  # next word not streamed yet: decide on the next fragment
                    break
                if not buffer[end.end()].isupper():
                    continue
            elif _NEVER_AN_END.search(sentence):
                continue
            elif _END_IF_CAPITAL.search(sentence):
                if end.end() == len(buffer):  # next word not streamed yet: decide on the next fragment
                    break
                if not buffer[end.end()].isupper():
                    continue
            if sentence:
                yield sentence
            start = end.end()
        buffer = buffer[start:]
    yield buffer


def clean_for_speech(text: str) -> str:
    text = _BULLET.sub("", _LINK.sub(r"\1", text))
    return " ".join(_MARKDOWN.sub("", text).split())  # also folds line breaks left inside a sentence
