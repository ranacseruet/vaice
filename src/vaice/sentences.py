"""Group a stream of text fragments into complete sentences."""
import re
from collections.abc import Iterable, Iterator

# Sentence ends at . ! ? followed by whitespace, so "3.5" is not split. A line break also ends one
# (list items often have no final punctuation).
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+|\n+")
# Periods that don't end a sentence: common abbreviations, and a list number at the start of a line
_NOT_AN_END = re.compile(r"(?:\b(?:mr|mrs|ms|dr|prof|st|vs|etc|e\.g|i\.e)|^\s*\d+)\.\Z", re.IGNORECASE | re.MULTILINE)
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
            if _NOT_AN_END.search(buffer, start, end.start()):
                continue
            if sentence := buffer[start : end.start()]:
                yield sentence
            start = end.end()
        buffer = buffer[start:]
    yield buffer


def clean_for_speech(text: str) -> str:
    text = _BULLET.sub("", _LINK.sub(r"\1", text))
    return _MARKDOWN.sub("", text).strip()
