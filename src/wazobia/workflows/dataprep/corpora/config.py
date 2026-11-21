from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True, slots=True)
class Corpus:
    repo_id: str
    output_repo_id: str | None = None
    include_lang_as_arg: bool = False
    ingest_strategy: Literal['base', 'parquet'] = 'base'
    splits: Mapping[str, str | None] = field(default_factory=dict)
    languages: Mapping[str, str | None] = field(default_factory=dict)
    columns: Mapping[str, str] = field(default_factory=dict)
    corpus_name: str | None = None
    normalize: str | None = None
    compile_text: bool | Sequence[str] = False
    split_weight: Sequence[float] | None = None
    exclude_cols: Sequence[str] | None = None
    dir_format: str | None = None
    exclude_annotations: Mapping[str, Sequence[str]] | None = None
    annotation: Mapping[str, Sequence[str]] | None = None
    exclude_annotation: Mapping[str, Sequence[str]] | None = None

@dataclass(frozen=True, slots=True)
class Corpora:
    corpora: Mapping[str, Corpus]
