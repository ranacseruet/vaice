"""Group a stream of text fragments into complete sentences."""
import re
from collections.abc import Iterable, Iterator

# Sentence ends at . ! ? followed by whitespace, so "3.5" is not split.
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
# Characters a TTS engine would read aloud literally.
_MARKDOWN = re.compile(r"[*_#`]")


def split_sentences(fragments: Iterable[str]) -> Iterator[str]:
    """Yield each sentence as soon as the next one starts, then the remainder."""
    buffer = ""
    for fragment in fragments:
        buffer += fragment
        *done, buffer = _SENTENCE_END.split(buffer)
        yield from done
    yield buffer


def clean_for_speech(text: str) -> str:
    return _MARKDOWN.sub("", text).strip()
