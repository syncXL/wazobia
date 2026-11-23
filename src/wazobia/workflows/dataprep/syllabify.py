#!/usr/bin/env python
"""Syllable pre-processing for the "Frugal Tonal ASR for Fongbe" leaderboard (standard library only).

The leaderboard metric is the usual word error rate (WER), computed on syllables instead of words.
The organisers apply `syllabify()` to the ground truth; participants MUST apply the same function to
their transcriptions before submitting. The WER then counts one error for each wrong syllable:
a wrong tone makes one syllable wrong (not the whole word), and the way words are split no longer matters.

    from syllabify import syllabify
    syllabify("Un nɔ́ ɖu nǔ")          # -> 'un nɔ1 ɖu nu3'

    python syllabify.py my_transcriptions.csv my_submission.csv
    # input: columns id, transcription (plain Fongbe text) or ID, Target
    # output: the submission file, columns ID, Target (format of SampleSubmission.csv)

What the function does:
1. normalisation (same as normalize.py): Unicode NFD, lower case, look-alike characters fixed,
   punctuation, digits and elision apostrophes removed; letters and tones kept;
2. each word is cut into syllables: onset consonants + vowel (+ syllable-final n), or a syllabic consonant
   carrying a tone (ń, m̀...);
3. the tone of each syllable is written as a digit after it, so that it survives any further text cleaning:
   1 = high (acute), 2 = low (grave), 3 = rising (caron), 4 = falling (circumflex), 5 = mid (macron);
   no digit = no tone mark;
4. the syllables are joined with single spaces.
"""
from __future__ import annotations

import csv
import re
import sys
import unicodedata

# ---- normalisation: identical to normalize.tonal_target(text, digits=False) ----
ACUTE, CARON, GRAVE, CIRCUMFLEX, MACRON = "́", "̌", "̀", "̂", "̄"
TONES = frozenset({ACUTE, CARON, GRAVE, CIRCUMFLEX, MACRON})
SPURIOUS = frozenset({"̧", "̈", "̣"})     # cedilla, diaeresis, dot below
LETTERS = frozenset("abcdefghijklmnopqrstuvwxyz") | {"ɔ", "ɖ", "ɛ"}
MARKUP = frozenset("<>+*")
LOOK_ALIKES = {
    "ε": "ɛ", "Ε": "ɛ", "ð": "ɖ", "đ": "ɖ", "ɗ": "ɖ",
    "‚": ",", "“": '"', "”": '"', "‘": "'", "’": "'", "œ": "oe",
}
_SPACES = re.compile(r" {2,}")


def normalize(text: str) -> str:
    t = unicodedata.normalize("NFD", text).lower()
    t = unicodedata.normalize("NFD", t)
    for wrong, right in LOOK_ALIKES.items():
        t = t.replace(wrong, right)
    return unicodedata.normalize("NFD", t)


def tonal_target(text: str) -> str:
    out: list[str] = []
    for c in normalize(text):
        if c in TONES:
            if out and out[-1] != " ":
                out.append(c)
        elif c in LETTERS or c == " ":
            out.append(c)
        elif c.isdigit() or c in SPURIOUS or c in MARKUP or unicodedata.category(c).startswith("P"):
            continue
        elif c.isspace():
            out.append(" ")
    return _SPACES.sub(" ", "".join(out)).strip()


# ---- syllables: identical to score.syllables() ----
VOWELS = set("aeiouɛɔə")
TONE_DIGIT = {ACUTE: "1", GRAVE: "2", CARON: "3", CIRCUMFLEX: "4", MACRON: "5"}


def _units(word):
    out = []
    for c in word:
        if unicodedata.category(c).startswith("M") and out:
            out[-1][1].append(c)
        elif not unicodedata.category(c).startswith("M"):
            out.append([c, []])
    return out


def syllables(text: str) -> list[tuple[str, str]]:
    """List of (base, tone marks) for a text."""
    res = []
    for word in tonal_target(text or "").split():
        u = _units(word); i = 0; onset = ""
        while i < len(u):
            b, m = u[i]
            tone = "".join(sorted(x for x in m if x in TONES)); other = "".join(x for x in m if x not in TONES)
            if b in VOWELS:
                base = onset + b + other; onset = ""
                if i + 1 < len(u) and u[i + 1][0] == "n" and not u[i + 1][1] and (i + 2 >= len(u) or u[i + 2][0] not in VOWELS):
                    base += "n"; i += 1
                res.append((base, tone)); i += 1
            elif tone:
                res.append((onset + b + other, tone)); onset = ""; i += 1
            else:
                onset += b + other; i += 1
        if onset:
            res.append((onset, ""))
    return res


def syllabify(text: str) -> str:
    """Text -> space-separated syllables with tone digits, e.g. 'Un nɔ́ ɖu nǔ' -> 'un nɔ1 ɖu nu3'."""
    return " ".join(unicodedata.normalize("NFC", base) + "".join(TONE_DIGIT[t] for t in tone)
                    for base, tone in syllables(text))


def tone_numbers_only(text: str) -> str:
    """Normalize text and mark tones with digits without adding syllable spaces."""
    words = []
    for word in tonal_target(text or "").split():
        word_syllables = syllables(word)
        words.append("".join(
            unicodedata.normalize("NFC", base) + "".join(TONE_DIGIT[t] for t in tone)
            for base, tone in word_syllables
        ))
    return " ".join(words)


def desyllabify(text: str) -> str:
    """Replace trailing tone digits with tone marks without changing whitespace."""
    digit_tone = {digit: mark for mark, digit in TONE_DIGIT.items()}

    def restore_token(match: re.Match[str]) -> str:
        restored = []
        syllable_start = 0
        for char in unicodedata.normalize("NFD", match.group(0)):
            tone_mark = digit_tone.get(char)
            if tone_mark is None or not restored:
                restored.append(char)
                continue

            tone_index = next(
                (index for index in range(len(restored) - 1, syllable_start - 1, -1)
                 if restored[index].lower() in VOWELS),
                next((index for index in range(len(restored) - 1, syllable_start - 1, -1)
                      if not unicodedata.category(restored[index]).startswith("M")), None),
            )
            if tone_index is None:
                restored.append(char)
            else:
                restored.insert(tone_index + 1, tone_mark)
                syllable_start = len(restored)

        return unicodedata.normalize("NFC", "".join(restored))

    return re.sub(r"\S+", restore_token, text)


def _self_test() -> None:
    assert syllabify("Un nɔ́ ɖu nǔ") == "un nɔ1 ɖu nu3"
    assert syllabify("Kɔ́xojí, ɖ'é mɛ !") == syllabify("kɔ́ xo jí ɖé mɛ") == "kɔ1 xo ji1 ɖe1 mɛ"
    assert syllabify("Azɔ̌mɛ 2026") == "a zɔ3 mɛ"
    assert syllabify("") == "" and syllabify("?!") == ""
    s = syllabify("Mǐ ná yi gbɛ̀ ń")
    assert s == "mi3 na1 yi gbɛ2 n1", s
    print("syllabify.py: self-test OK")


def _convert(src: str, dst: str) -> None:
    with open(src, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    with open(dst, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ID", "Target"])
        for r in rows:
            if "ID" in r:
                w.writerow([r["ID"], syllabify(r.get("Target") or "")])
            else:
                w.writerow([r["id"], syllabify(r.get("transcription") or "")])
    print(f"{len(rows)} rows written to {dst}")


if __name__ == "__main__":
    if len(sys.argv) == 3:
        _convert(sys.argv[1], sys.argv[2])
    else:
        _self_test()
