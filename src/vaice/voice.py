"""Text fit for speaking: markup, paths and units made plain, and the reply cut into sentences.

Pure functions, no I/O. The assistant path sends each sentence to llama-server's /tts on its own,
so what the model wrote has to read aloud sensibly and split where a speaker would pause.
"""
import re

_UNICODE_SPACES = re.compile(r"[\u00a0\u1680\u2000-\u200b\u202f\u205f\u3000]")
_CODE_FENCE = re.compile(r"```[\w+-]*")
_MD_LINK = re.compile(r"\[([^\]\n]+)\]\([^)\s]+\)")
# the last character of a URL is never sentence punctuation or a closing bracket
_URL = re.compile(r"(?:https?://|www\.)[^\s<>]*[^\s<>.,;:!?'\")\]]")
# `/Users/x/report.pdf` or `~/Downloads/`: starts at a boundary, so "and/or", "24/7" and "3/4" stay as they are
_PATH = re.compile(r"(?<![\w/~.:-])(?:~/|/(?=[\w.@+%-]))[\w.@+%-]*(?:/[\w.@+%-]*)*")
_INLINE_CODE = re.compile(r"`([^`\n]*)`")
_BOLD = re.compile(r"(\*\*|__)(?=\S)(.+?)(?<=\S)\1")
_EM_STAR = re.compile(r"(?<![\w*])\*(?=[^\s*])([^*\n]+?)(?<=[^\s*])\*(?![\w*])")
_EM_UNDERSCORE = re.compile(r"(?<![\w_])_(?=[^\s_])([^_\n]+?)(?<=[^\s_])_(?![\w_])")
_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+")
_BULLET = re.compile(r"^\s*(?:[-*+•]|\d{1,3}[.)])\s+")
_QUOTE = re.compile(r"^\s*>\s?")
_NUMBER_UNIT = re.compile(r"(?<![\w.])(\d+(?:[,.]\d+)*)\s*(GiB|GB|MiB|MB|KiB|KB|TiB|TB)\b")
_PERCENT = re.compile(r"(\d)\s*%")
_UNITS = {"GiB": "gigabyte", "GB": "gigabyte", "MiB": "megabyte", "MB": "megabyte",
          "KiB": "kilobyte", "KB": "kilobyte", "TiB": "terabyte", "TB": "terabyte"}


def _path_name(match: re.Match) -> str:
    """A path as it is said: its last component (a directory's own name when it ends in `/`)."""
    path = match.group(0)
    trailing = ""
    while path and path[-1] in ".,;:!?":  # sentence punctuation after the path is not part of it
        trailing = path[-1] + trailing
        path = path[:-1]
    parts = [part for part in path.split("/") if part and part != "~"]
    return (parts[-1] if parts else "the home folder") + trailing


def _unit(match: re.Match) -> str:
    number, unit = match.groups()
    word = _UNITS[unit]
    return f"{number} {word}" if number == "1" else f"{number} {word}s"


def _plain_line(line: str) -> str:
    """One line without heading, quote or list markers; a list item or heading becomes a sentence."""
    line = _QUOTE.sub("", line)
    marked = bool(_HEADING.match(line) or _BULLET.match(line))
    line = _BULLET.sub("", _HEADING.sub("", line)).strip()
    if marked and line and line[-1] not in ".!?:;":
        line += "."
    return line


def speakable(text: str) -> str:
    """`text` made fit for TTS: no markdown, links or file paths, spelled-out units, single spaces.

    Prose, decimals (3.5), times (3:30 p.m.) and abbreviations are left as they are.
    """
    text = _UNICODE_SPACES.sub(" ", text)
    text = _CODE_FENCE.sub("", text)
    text = _MD_LINK.sub(r"\1", text)
    text = _URL.sub("a link", text)
    text = _INLINE_CODE.sub(r"\1", text)
    text = _PATH.sub(_path_name, text)
    text = "\n".join(_plain_line(line) for line in text.split("\n"))
    text = _BOLD.sub(r"\2", text)
    text = _EM_STAR.sub(r"\1", text)
    text = _EM_UNDERSCORE.sub(r"\1", text)
    text = _NUMBER_UNIT.sub(_unit, text)
    text = _PERCENT.sub(r"\1 percent", text)
    return " ".join(text.split())


# A "." after one of these is not a sentence end ("Mr. Smith", "3 p.m. tomorrow", "e.g. this")
_ABBREVIATIONS = frozenset({
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "mt", "no", "inc", "ltd", "co",
    "a.m", "p.m", "e.g", "i.e", "u.s", "u.k",
})
_BOUNDARY = re.compile(r"""([.!?]+|…)["')\]]*\s+(?=["'(\[]?[A-Z0-9])""")
MIN_SENTENCE_CHARS = 12  # shorter fragments are joined to a neighbour: a clip that short sounds clipped


def _is_boundary(text: str, match: re.Match) -> bool:
    mark = match.group(1)
    if mark == "…" or (len(mark) > 1 and set(mark) == {"."}):  # an ellipsis goes on mid-sentence
        return False
    if mark != ".":
        return True
    words = text[: match.start(1)].split()
    word = words[-1].lstrip("\"'([").lower() if words else ""
    return word not in _ABBREVIATIONS and not (len(word) == 1 and word.isalpha())  # not an initial either


def sentences(text: str) -> list[str]:
    """`text` split into sentences, for TTS one at a time; fragments under 12 characters are merged."""
    pieces: list[str] = []
    start = 0
    for match in _BOUNDARY.finditer(text):
        if _is_boundary(text, match):
            pieces.append(text[start: match.end()])
            start = match.end()
    pieces.append(text[start:])
    merged: list[str] = []
    carry = ""  # a short opening fragment, waiting for the sentence after it
    for piece in (p.strip() for p in pieces):
        if not piece:
            continue
        if carry:
            piece, carry = f"{carry} {piece}", ""
        if len(piece) < MIN_SENTENCE_CHARS and merged:
            merged[-1] += " " + piece
        elif len(piece) < MIN_SENTENCE_CHARS:
            carry = piece
        else:
            merged.append(piece)
    if carry:  # nothing came after the short fragment: it is the whole reply
        merged.append(carry)
    return merged
