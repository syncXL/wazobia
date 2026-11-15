import hashlib
from collections.abc import Sequence
from dataclasses import is_dataclass
from pathlib import Path
from typing import TypeVar

import yaml


def assign_split(file_path: str, weights=(0.7, 0.1, 0.2)) -> str:
        # deterministic: same file always gets same split, across reruns
        h = int(hashlib.sha256(file_path.encode()).hexdigest(), 16)
        r = (h % 10_000) / 10_000  # -> [0, 1)
        cum = 0.0
        for split, w in zip(("train", "validation", "test"), weights):
            cum += w
            if r < cum:
                return split
        return "test"  # float rounding fallback

def staging_root(file: Path) -> Path | None:
    return next((candidate for candidate in [file.parent, *file.parents] if candidate.parent.name == "staging"), None)

def remove_files(files: Sequence[str | Path]) -> None:
    """Remove the given files, ignoring files that do not exist."""
    for file in files:
        Path(file).unlink(missing_ok=True)

def shard_id(filename: str) -> str:
    return hashlib.sha256(filename.encode("utf-8")).hexdigest()[:20]

T = TypeVar("T")

def load_yaml[T](path: str | Path, schema: type[T]) -> T:
	"""Load a YAML mapping into an instance of the given dataclass schema."""
	if not is_dataclass(schema):
		raise TypeError("schema must be a dataclass type")

	with Path(path).open(encoding="utf-8") as file:
		content = yaml.safe_load(file)

	if content is None:
		content = {}
	if not isinstance(content, dict):
		raise ValueError("YAML content must be a mapping")  # noqa: TRY004

	if schema.__name__ == "Corpora":
		entries = content.get("corpora", {})
		if isinstance(entries, list):
			normalized = {}
			for entry in entries:
				if not isinstance(entry, dict) or len(entry) != 1:
					raise ValueError("Each corpus entry must contain one named mapping")
				name, values = next(iter(entry.items()))
				normalized[name] = values
			entries = normalized
		from .corpora.config import Corpus
		corpora = {}
		for name, values in entries.items():
			values = dict(values)
			def clean_none(value):
				if value == "None":
					return None
				if isinstance(value, dict):
					return {k: clean_none(v) for k, v in value.items()}
				if isinstance(value, list):
					return [clean_none(v) for v in value]
				return value
			values = clean_none(values)
			for key, value in list(values.items()):
				if value == "None":
					values[key] = None
			for key in ("splits", "languages", "columns", "exclude_annotations"):
				value = values.get(key)
				if isinstance(value, list):
					values[key] = {k: v for item in value if isinstance(item, dict) for k, v in item.items()}
				elif value is None:
					values[key] = {}
			for key in ("exclude_cols", "split_weight"):
				if values.get(key) is None:
					values[key] = None
			corpora[name] = Corpus(**values)
		return schema(corpora=corpora)
	return schema(**content)
