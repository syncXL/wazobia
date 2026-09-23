import pickle
from dataclasses import dataclass, field
from pathlib import Path
import polars as pl

METRIC_COLS = [
    "snr_db", "noise_level_db", "speech_level_db",
    "speech_ratio", "duration_sec", "spectral_bandwidth_hz",
]

@dataclass
class GroupStats:
    count: int = 0
    hours: float = 0.0
    sum: dict = field(default_factory=lambda: {c: 0.0 for c in METRIC_COLS})
    sumsq: dict = field(default_factory=lambda: {c: 0.0 for c in METRIC_COLS})
    n: dict = field(default_factory=lambda: {c: 0 for c in METRIC_COLS})  # non-null count per col


class MetricsAccumulator:
    def __init__(self, state_path: str):
        self.state_path = Path(state_path)
        self.groups: dict[tuple, GroupStats] = {}
        self.counted_files: set[str] = set()
        if self.state_path.exists():
            with open(self.state_path, "rb") as f:
                self.groups, self.counted_files = pickle.load(f)

    def update_from_local(self, local_parquet_path: str, corpus: str, split: str, language: str):
        if local_parquet_path in self.counted_files:
            return  # already counted, e.g. a retried shard

        df = pl.read_parquet(local_parquet_path, columns=METRIC_COLS)
        key = (corpus, split, language)
        gs = self.groups.setdefault(key, GroupStats())
        gs.count += df.height
        gs.hours += (df["duration_sec"].sum() or 0.0) / 3600

        for col in METRIC_COLS:
            vals = df[col].drop_nulls().to_numpy()
            if vals.size == 0:
                continue
            gs.sum[col] += float(vals.sum())
            gs.sumsq[col] += float((vals ** 2).sum())
            gs.n[col] += vals.size

        self.counted_files.add(local_parquet_path)
        self._save()

    def _save(self):
        tmp = self.state_path.with_suffix(".tmp")
        with open(tmp, "wb") as f:
            pickle.dump((self.groups, self.counted_files), f)
        tmp.replace(self.state_path)

    def finalize(self) -> pl.DataFrame:
        rows = []
        for (corpus, split, language), gs in self.groups.items():
            row = {"corpus": corpus, "split": split, "language": language,
                   "hours": gs.hours, "count": gs.count}
            for col in METRIC_COLS:
                n = gs.n[col]
                mean = gs.sum[col] / n if n else None
                var = (gs.sumsq[col] / n - mean ** 2) if n else None
                std = var ** 0.5 if var and var > 0 else 0.0
                row[f"{col}_mean"] = mean
                row[f"{col}_std"] = std
            rows.append(row)
        return pl.DataFrame(rows)

    def write_tsv(self, out_path: str):
        self.finalize().write_csv(out_path, separator="\t")