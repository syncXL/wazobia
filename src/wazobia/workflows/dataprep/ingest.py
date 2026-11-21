import os
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import gc
import os
from pathlib import Path

import datasets
import fire
import polars as pl
import pyarrow as pa
import ray

from wazobia import config
from wazobia.file_manager import manager
from wazobia.workflows.dataprep import (
    metrics,
    utils,
)
from wazobia.workflows.dataprep.corpora import base
from wazobia.workflows.dataprep.corpora import config as corpus_config


class DataPrepCLI:

    def __init__(self) -> None:
        self.settings = config.settings
        self.storage = manager.HFDatasetRepo(self.settings.hf_dataset_repo, self.settings.hf_token)

    @staticmethod
    def _load_configs(config_name: str) -> corpus_config.Corpora:
        if not config_name.replace("_", "").replace("-", "").isalnum():
            raise ValueError(f"Invalid config name: {config_name!r}")
        config_path = Path(__file__).parent / "corpora/config" / f"{config_name}.yaml"
        if not config_path.is_file():
            raise FileNotFoundError(f"Corpus config not found: {config_path}")
        return utils.load_yaml(config_path, corpus_config.Corpora)

    @staticmethod
    def _resolve_accumulator(output_dir: str, accumulator: metrics.MetricsAccumulator | None) -> metrics.MetricsAccumulator:
        if accumulator is not None:
            return accumulator
        return metrics.MetricsAccumulator(str(Path(output_dir) / "metrics" / "metrics.pkl"))

    @staticmethod
    def check_versions():
        """Check and display versions of critical packages used in data preparation.

        This helps ensure compatibility and reproducibility of the data preparation pipeline.
        """
        print("📦 Package Versions:")
        print(f"  datasets: {datasets.__version__}")
        print(f"  pyarrow:  {pa.__version__}")
        print(f"  ray:      {ray.__version__}")
        print(f"  polars:   {pl.__version__}")

        # Check for known compatibility issues
        if hasattr(datasets, "__version__"):
            datasets_ver = tuple(map(int, datasets.__version__.split(".")))
            if datasets_ver >= (3, 6, 0):
                print(
                    "⚠️  Warning: datasets version >= 3.6.0 may have compatibility issues"
                )

        if hasattr(ray, "__version__"):
            ray_ver = tuple(
                map(int, ray.__version__.split(".")[:2])
            )  # Major.minor only
            if ray_ver < (2, 49):
                print("⚠️  Warning: ray version < 2.49 may have performance issues")

    def run_full(self, output_dir: str, load_from_hf: bool = False, config_name: str = "corpora"):
        self.configs = self._load_configs(config_name)
        if load_from_hf:
            self.storage.download(output_dir + "/ledger", "ledger")
            self.storage.download(output_dir + "/metrics/metrics.pkl", "metrics/metrics.pkl")
        accumulator = metrics.MetricsAccumulator(output_dir + "/metrics/metrics.pkl")
        corpiler = base.CorpusCompiler()
        for cname, cconfig in self.configs.corpora.items():
            print(f"Processing  {cname}")
            corpus_storage = (
                manager.HFDatasetRepo(cconfig.output_repo_id, self.settings.hf_token)
                if cconfig.output_repo_id
                else self.storage
            )
            if load_from_hf and cconfig.output_repo_id:
                corpus_storage.download(output_dir + "/ledger", "ledger")
            corpiler(output_dir, accumulator, cconfig, corpus_storage)
            gc.collect()
            print(f"{cname} completed")
        metrics_dir = Path(output_dir) / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        tsv_path = metrics_dir / "metrics.tsv"
        accumulator.write_tsv(str(tsv_path))
        accumulator.save()
        failed = self.storage.upload([accumulator.state_path, tsv_path], local_dir=output_dir)
        if failed:
            raise ValueError(f"Failed to upload final metrics {failed}")
        readme_path = Path(output_dir) / "README.md"
        readme_path.write_text(base.build_dataset_readme(self.configs), encoding="utf-8")
        failed = self.storage.upload([readme_path], local_dir=output_dir)
        if failed:
            raise ValueError(f"Failed to upload dataset README {failed}")

if __name__ == "__main__":
    if not ray.is_initialized():
        kaggle_working = Path("/kaggle/working")
        if os.environ.get("RAY_TMPDIR"):
            ray_temp_dir = Path(os.environ["RAY_TMPDIR"])
        elif kaggle_working.is_dir():
            ray_temp_dir = kaggle_working / "ray"
        else:
            ray_temp_dir = Path.home() / ".cache" / "wazobia" / "ray"
        Path(ray_temp_dir).mkdir(parents=True, exist_ok=True)
        ray.init(
            _temp_dir=str(ray_temp_dir),
            runtime_env={
                "working_dir": str(Path.cwd()),
                "excludes": [".venv", "data", "notebooks", ".env", "bucket", "ledger", "metrics", "staging"],
                "env_vars": {"HF_HUB_DISABLE_XET": "1"},
            },
        )

    ctx = ray.data.DataContext.get_current()
    ctx.enable_rich_progress_bars = True
    ctx.use_ray_tqdm = False

    try:
        fire.Fire(DataPrepCLI)
    finally:
        if ray.is_initialized():
            ray.shutdown()
