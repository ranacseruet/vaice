import pytest

from vaice.voice import sentences, speakable


@pytest.mark.parametrize(
    ("text", "spoken"),
    [
        # markdown
        ("This is **important** news.", "This is important news."),
        ("This is __also__ important.", "This is also important."),
        ("It is *really* fast and _quite_ quiet.", "It is really fast and quite quiet."),
        ("Run `ls -la` now.", "Run ls -la now."),
        ("```bash\nls -la\n```", "ls -la"),
        ("# Summary\nAll is well.", "Summary. All is well."),
        ("## Disk\nIt is fine.", "Disk. It is fine."),
        ("- one thing\n- another thing", "one thing. another thing."),
        ("* starred item\n+ plus item\n• dotted item", "starred item. plus item. dotted item."),
        ("1. open it\n2) close it", "open it. close it."),
        ("A list:\n- first.\n- second!", "A list: first. second!"),
        ("> quoted text", "quoted text"),
        ("see [the docs](https://example.com/docs) now", "see the docs now"),
        # a snake_case name and a product of numbers are not emphasis
        ("the snake_case_name is fine and 2*3*4 too", "the snake_case_name is fine and 2*3*4 too"),
        # unicode spaces
        ("160\u00a0GB free", "160 gigabytes free"),
        ("a\u202fb\u2009c\u2003d", "a b c d"),
        # urls
        ("Open https://example.com/a?b=1 please.", "Open a link please."),
        ("Go to https://example.com.", "Go to a link."),
        ("Try www.example.org, or not.", "Try a link, or not."),
        # file paths
        ("It is /Users/rana/Downloads/report.pdf.", "It is report.pdf."),
        ("It is at ~/Documents/notes.txt now", "It is at notes.txt now"),
        ("Look in /Users/rana/Downloads/ for it.", "Look in Downloads for it."),
        ("Look in ~/Desktop/ for it.", "Look in Desktop for it."),
        ("Look in ~/ for it.", "Look in the home folder for it."),
        ("Saved as `/tmp/out/result.txt`, done.", "Saved as result.txt, done."),
        # units
        ("160 GB free of 926 GiB", "160 gigabytes free of 926 gigabytes"),
        ("1 GB free", "1 gigabyte free"),
        ("512 MB and 3 MiB", "512 megabytes and 3 megabytes"),
        ("20KB", "20 kilobytes"),
        ("2 TB or 1.5 TiB", "2 terabytes or 1.5 terabytes"),
        ("Memory is 45% used and 100 % full.", "Memory is 45 percent used and 100 percent full."),
        # whitespace
        ("  too   many\n\nspaces\there  ", "too many spaces here"),
        # things that must be left alone
        ("Paris is the capital of France.", "Paris is the capital of France."),
        ("It costs 3.5 dollars.", "It costs 3.5 dollars."),
        ("Come at 3:30 p.m. or 9 a.m. sharp.", "Come at 3:30 p.m. or 9 a.m. sharp."),
        ("Mr. Smith, e.g. the one on St. James, said and/or did it 24/7 in 3/4 time.",
         "Mr. Smith, e.g. the one on St. James, said and/or did it 24/7 in 3/4 time."),
        ("My GBA has 100gbps?", "My GBA has 100gbps?"),
        ("Gigabytes are fine, and so is 5 gigabytes.", "Gigabytes are fine, and so is 5 gigabytes."),
        ("", ""),
    ],
)
def test_speakable(text, spoken):
    assert speakable(text) == spoken


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Hello there. How are you? I am fine, thank you!", ["Hello there.", "How are you?", "I am fine, thank you!"]),
        ("This is the first one. This is the second one.", ["This is the first one.", "This is the second one."]),
        ("Is it really so? Yes it is really so! Good.", ["Is it really so?", "Yes it is really so! Good."]),
        # decimals, times, abbreviations are not ends
        ("It costs 3.5 dollars today. Then it costs more.", ["It costs 3.5 dollars today.", "Then it costs more."]),
        ("Meet at 3:30 p.m. Tomorrow works too.", ["Meet at 3:30 p.m. Tomorrow works too."]),
        ("Dr. Smith said so. Then he left the room.", ["Dr. Smith said so.", "Then he left the room."]),
        ("Use a tool, e.g. This one works. Then stop it.", ["Use a tool, e.g. This one works.", "Then stop it."]),
        ("Mr. and Mrs. Jones arrived late. They sat down.", ["Mr. and Mrs. Jones arrived late.", "They sat down."]),
        ("J. Smith wrote this long letter. It was short.", ["J. Smith wrote this long letter.", "It was short."]),
        # an ellipsis goes on, and so does a quote closing the sentence
        ("Well... I think so, but maybe not. Let us see how it goes.",
         ["Well... I think so, but maybe not.", "Let us see how it goes."]),
        ("Well\u2026 I think so, but maybe not. Let us see how it goes.",
         ["Well\u2026 I think so, but maybe not.", "Let us see how it goes."]),
        ('He said "go away now." Then he left.', ['He said "go away now."', "Then he left."]),
        # a lowercase word after the full stop is not a new sentence
        ("It is e.g. like that. no it is not", ["It is e.g. like that. no it is not"]),
        # short fragments join their neighbour
        ("Sure. The disk has 160 gigabytes free.", ["Sure. The disk has 160 gigabytes free."]),
        ("The disk is almost full. Yes. Really.", ["The disk is almost full. Yes. Really."]),
        ("Yes. No.", ["Yes. No."]),
        # text with no ending
        ("The disk is fine. And the battery has plenty left", ["The disk is fine.", "And the battery has plenty left"]),
        ("no punctuation here at all", ["no punctuation here at all"]),
        ("Hi.", ["Hi."]),
        ("", []),
        ("   ", []),
    ],
)
def test_sentences(text, expected):
    assert sentences(text) == expected
