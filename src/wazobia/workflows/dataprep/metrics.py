import pickle
import math
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
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

    def update_from_local(self, local_parquet_path: str, corpus: str, split: str, language: str, identity: str | None = None):
        counted_key = identity or str(Path(local_parquet_path).resolve())
        if counted_key in self.counted_files:
            return  # already counted, e.g. a retried shard

        df = pl.read_parquet(local_parquet_path, columns=METRIC_COLS)
        key = (corpus, split, language)
        gs = self.groups.setdefault(key, GroupStats())
        gs.count += df.height
        duration_values = df["duration_sec"].drop_nulls().to_numpy()
        duration_values = duration_values[np.isfinite(duration_values)]
        gs.hours += float(duration_values.sum()) / 3600 if duration_values.size else 0.0

        for col in METRIC_COLS:
            vals = df[col].drop_nulls().to_numpy()
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            gs.sum[col] += float(vals.sum())
            gs.sumsq[col] += float((vals ** 2).sum())
            gs.n[col] += vals.size

        self.counted_files.add(counted_key)
        self._save()

    def update_shard_from_local(self, paths: list[str | Path], corpus: str, split: str, language: str, identity: str):
        """Count every Parquet file for one source shard as one idempotent update."""
        if identity in self.counted_files:
            return
        key = (corpus, split, language)
        row_count = 0
        shard_hours = 0.0
        shard_sum = {c: 0.0 for c in METRIC_COLS}
        shard_sumsq = {c: 0.0 for c in METRIC_COLS}
        shard_n = {c: 0 for c in METRIC_COLS}
        for path in paths:
            df = pl.read_parquet(str(path), columns=METRIC_COLS)
            row_count += df.height
            durations = df["duration_sec"].drop_nulls().to_numpy()
            durations = durations[np.isfinite(durations)]
            if durations.size:
                shard_hours += float(durations.sum()) / 3600
            for col in METRIC_COLS:
                values = df[col].drop_nulls().to_numpy()
                values = values[np.isfinite(values)]
                if values.size:
                    shard_sum[col] += float(values.sum())
                    shard_sumsq[col] += float((values ** 2).sum())
                    shard_n[col] += int(values.size)
        gs = self.groups.setdefault(key, GroupStats())
        gs.count += row_count
        gs.hours += shard_hours
        for col in METRIC_COLS:
            gs.sum[col] += shard_sum[col]
            gs.sumsq[col] += shard_sumsq[col]
            gs.n[col] += shard_n[col]
        self.counted_files.add(identity)
        self._save()

    def _save(self):
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(prefix=f".{self.state_path.name}.", suffix=".tmp", dir=self.state_path.parent)
        try:
            with os.fdopen(fd, "wb") as f:
                pickle.dump((self.groups, self.counted_files), f)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_name, self.state_path)
        finally:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)

    def save(self):
        self._save()

    def finalize(self) -> pl.DataFrame:
        rows = []
        for (corpus, split, language), gs in self.groups.items():
            row = {"corpus": corpus, "split": split, "language": language,
                   "hours": gs.hours, "count": gs.count}
            for col in METRIC_COLS:
                n = gs.n[col]
                mean = gs.sum[col] / n if n else None
                var = (gs.sumsq[col] / n - mean ** 2) if n else None
                std = max(var, 0.0) ** 0.5 if var is not None and math.isfinite(var) else None
                row[f"{col}_mean"] = mean
                row[f"{col}_std"] = std
            rows.append(row)
        return pl.DataFrame(rows)

    def write_tsv(self, out_path: str):
        self.finalize().write_csv(out_path, separator="\t")
