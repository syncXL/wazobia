import re
from pathlib import Path

import pyarrow as pa

from wazobia.workflows.dataprep import text_tools
from wazobia.workflows.dataprep.syllabify import syllabify


class TextProcessor:
    def __init__(self, lang : str, remove_numbers=False, unuttered_words:list[str] | None = None, normalizer: str | None = None):
        self.pattern = None
        if unuttered_words:
            self.pattern = re.compile(r"\b(?:" + "|".join(map(re.escape, unuttered_words)) + r")\b")
        self.lang = lang
        self.lang_mapping = {
            "yor_ng" : "yor",
            "ibo_ng" : "ibo",
            "hau_ng" : "hau",
            "pcm_ng" : "pcm",
            "eng_ng" : "eng",
            "fon" : "fon",
        }
        self.remove_numbers = remove_numbers
        self.normalizer = normalizer
        self.homophones = text_tools.HomophoneMapper(Path(__file__).parent / "pcm_homophones.yaml") if lang.startswith("pcm") else None

    def __call__(self, batch: pa.Table) -> pa.Table:
        transcriptions = batch["transcript"].to_pylist()
        processed_transcriptions = []
        iso_lang = self.lang_mapping[self.lang]
        for text in transcriptions:
            if self.homophones:
                text = self.homophones(text)
            if self.pattern:
                cleaned = self.pattern.sub("", text)
                text = re.sub(r"\s+", " ", cleaned).strip()
            if self.normalizer in {"fon", "fongbe_syllabify"}:
                processed_text = syllabify(text)
            elif iso_lang == "fon":
                processed_text = text_tools.text_normalize(text, "*", remove_numbers=self.remove_numbers)
            else:
                processed_text = text_tools.wazobia_normalize(text, iso_lang, self.remove_numbers)
            processed_transcriptions.append(processed_text)

        batch = batch.drop(["transcript"]).append_column(
            "transcription", pa.array(processed_transcriptions, type=pa.string())
        )

        if "language" in batch.column_names:
            batch = batch.drop(["language"])

        language_values = [self.lang_mapping.get(self.lang, self.lang)] * len(batch)
        batch = batch.append_column(
            "language", pa.array(language_values, type=pa.string())
        )

        return batch
