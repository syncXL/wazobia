# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import re
import unicodedata
from pathlib import Path
import yaml


from wazobia.workflows.dataprep import norm_config_module as norm_config_module
from unidecode import unidecode

norm_config = norm_config_module.norm_config  # type: ignore
DIGITS = set("0123456789")
BASE = {
    "yor": set("abdefghijklmnoprstuwy"),
    "ibo": set("abcdefghijklmnoprstuvwyz"),
    "hau": set("abcdefghijklmnorstuwyz") | set("ɓɗƙƴ"),
    "pcm": set("abcdefghijklmnopqrstuvwxyz"),
    "eng": set("abcdefghijklmnopqrstuvwxyz"),
}
MARKS = {
    "yor": set("\u0323\u0300\u0301\u0304"),
    "ibo": set("\u0323\u0307\u0300\u0301\u0304"),   # dot below, dot above (ṅ), tones
    "hau": set("\u0300\u0301\u0302\u0304"),
    "pcm": set(), "eng": set(),
}
ALLOWED = {l: BASE[l] | MARKS[l] | DIGITS | {"'", " "} for l in BASE}

APOS = "[\u2018\u2019\u201B\u02BC\u02BB]"

for code in BASE:
    cfg = copy.deepcopy(norm_config["*"])
    cfg["mapping"].pop("ٱ", None)                      # Arabic leak from the file above
    cfg["mapping"][APOS] = "'"
    cfg["unicode_norm"] = "NFC"
    cfg["punc_set"] = cfg["punc_set"].replace("'", "")  # keep apostrophe
    norm_config[code] = cfg

class UniqueKeyLoader(yaml.BaseLoader):
    """BaseLoader keeps every scalar a string; also fails on duplicate keys."""
    def construct_mapping(self, node, deep=False):
        seen = set()
        for k_node, _ in node.value:
            k = self.construct_scalar(k_node)
            if k in seen:
                raise ValueError(f"duplicate YAML key: {k!r}")
            seen.add(k)
        return super().construct_mapping(node, deep)


class HomophoneMapper:
    def __init__(self, path, lang="pcm"):
        raw = yaml.load(Path(path).read_text(encoding="utf-8"), Loader=UniqueKeyLoader)
        self.map = {}
        for k, v in raw.items():
            k = wazobia_normalize(k, lang)
            v = wazobia_normalize(v, lang)
            if not k or k == v:
                continue
            if k in self.map and self.map[k] != v:
                raise ValueError(f"conflict after normalization: {k!r} -> {self.map[k]!r} vs {v!r}")
            self.map[k] = v
        self.max_n = max(len(k.split()) for k in self.map)

        chained = [(k, v) for k, v in self.map.items() if v in self.map]
        if chained:
            print(f"[HomophoneMapper] {len(chained)} chained entries, e.g. {chained[:5]}")

    def __call__(self, text: str) -> str:
        toks, out, i = text.split(), [], 0
        while i < len(toks):
            for n in range(min(self.max_n, len(toks) - i), 0, -1):
                v = self.map.get(" ".join(toks[i:i + n]))
                if v is not None:
                    out.append(v)
                    i += n
                    break
            else:
                out.append(toks[i])
                i += 1
        return " ".join(out)

def text_normalize(
    text, iso_code, lower_case=True, remove_numbers=True, remove_brackets=False
):
    """Given a text, normalize it by changing to lower case, removing punctuations, removing words that only contain digits and removing extra spaces

    Args:
        text : The string to be normalized
        iso_code :
        remove_numbers : Boolean flag to specify if words containing only digits should be removed

    Returns:
        normalized_text : the string after all normalization

    """

    config = copy.deepcopy(norm_config.get(iso_code, norm_config["*"]))

    for field in [
        "lower_case",
        "punc_set",
        "del_set",
        "mapping",
        "digit_set",
        "unicode_norm",
    ]:
        if field not in config:
            config[field] = norm_config["*"][field]

    text = unicodedata.normalize(config["unicode_norm"], text)

    # Convert to lower case

    if config["lower_case"] and lower_case:
        text = text.lower()

    # brackets

    # always text inside brackets with numbers in them. Usually corresponds to "(Sam 23:17)"
    text = re.sub(r"\([^\)]*\d[^\)]*\)", " ", text)
    if remove_brackets:
        text = re.sub(r"\([^\)]*\)", " ", text)

    # Apply mappings

    for old, new in config["mapping"].items():
        text = re.sub(old, new, text)

    # Replace punctutations with space

    punct_pattern = r"[" + config["punc_set"]

    punct_pattern += "]"

    normalized_text = re.sub(punct_pattern, " ", text)

    # remove characters in delete list

    delete_patten = r"[" + config["del_set"] + "]"

    normalized_text = re.sub(delete_patten, "", normalized_text)

    # Remove words containing only digits
    # We check for 3 cases  a)text starts with a number b) a number is present somewhere in the middle of the text c) the text ends with a number
    # For each case we use lookaround regex pattern to see if the digit pattern in preceded and followed by whitespaces, only then we replace the numbers with space
    # The lookaround enables overlapping pattern matches to be replaced

    if remove_numbers:

        digits_pattern = "[" + config["digit_set"]

        digits_pattern += "]+"

        complete_digit_pattern = (
            r"^"
            + digits_pattern
            + r"(?=\s)|(?<=\s)"
            + digits_pattern
            + r"(?=\s)|(?<=\s)"
            + digits_pattern
            + "$"
        )

        normalized_text = re.sub(complete_digit_pattern, " ", normalized_text)

    if config["rm_diacritics"]:
        normalized_text = unidecode(normalized_text)

    # Remove extra spaces
    normalized_text = re.sub(r"\s+", " ", normalized_text).strip()

    return normalized_text

def _lowercase_sentence_initial(match):
    """Lowercase a sentence-initial letter.

    Keeps the letter uppercase when the next letter is also uppercase, which marks
    the start of an acronym or initialism.

    Input: a `re.Match` from the SENTENCE_INITIAL pattern with three groups:
        1. sentence delimiter (start-of-string or `.!?—` plus optional space)
        2. first letter of the sentence
        3. next letter (may be empty)
    Output: the joined string with group 2 possibly lowercased.
    """
    delimiter, first, second = match.group(1), match.group(2), match.group(3)
    if first.isupper() and not (second and second.isupper()):
        first = first.lower()
    return delimiter + first + second


def wazobia_normalize(text, lang, remove_numbers=True):
    assert lang in ALLOWED, lang
    text = text_normalize(text, lang, remove_numbers=remove_numbers) 
    allowed = ALLOWED[lang]
    text = "".join(
        ch if all(c in allowed for c in unicodedata.normalize("NFD", ch)) else " "
        for ch in text
    )
    text = re.sub(r"(^|\s)[\u0300-\u036f]+", r"\1", text)     # orphaned tone marks
    return re.sub(r"\s+", " ", text).strip()
