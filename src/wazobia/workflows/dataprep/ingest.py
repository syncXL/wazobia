import os
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import datasets
import fire
import gc
import hashlib
import polars as pl
import pyarrow as pa
import os
import shutil
import ray

from datasets import load_dataset, Audio
from huggingface_hub import HfApi, HfFileSystem
from functools import partial
from pathlib import Path
from typing import Sequence
from wazobia import config
from wazobia.workflows.dataprep import text_tools_corpus, audio_features, audio_tools, metrics
from wazobia.ledger import ledger
from wazobia.file_manager import manager


def remove_files(files: Sequence[str | Path]) -> None:
    """Remove the given files, ignoring files that do not exist."""
    for file in files:
        Path(file).unlink(missing_ok=True)

def _language_partition(lang: str) -> str:
    return lang.removesuffix("_ng")

def _shard_id(filename: str) -> str:
    return hashlib.sha256(filename.encode("utf-8")).hexdigest()[:20]

def _unique_parquet_names(files: list[Path], shard_id: str) -> list[Path]:
    renamed = []
    for index, file in enumerate(sorted(files)):
        target = file.with_name(f"{shard_id}-{index:05d}.parquet")
        file.rename(target)
        renamed.append(target)
    return renamed

def _staging_root(file: Path) -> Path | None:
    return next((candidate for candidate in [file.parent, *file.parents] if candidate.parent.name == "staging"), None)

def _ray_workers() -> int:
    return max(1, int(os.environ.get("WAZOBIA_RAY_WORKERS", "1")))

def _ray_batch_size() -> int:
    return max(1, int(os.environ.get("WAZOBIA_RAY_BATCH_SIZE", "16")))

def _shuffle_buffer_size() -> int:
    return max(1, int(os.environ.get("WAZOBIA_SHUFFLE_BUFFER_SIZE", "256")))

def assign_split(file_path: str, weights=(0.7, 0.1, 0.2)) -> str:
        # deterministic: same file always gets same split, across reruns
        h = int(hashlib.sha256(file_path.encode()).hexdigest(), 16)
        r = (h % 10_000) / 10_000  # -> [0, 1)
        cum = 0.0
        for split, w in zip(("train", "val", "test"), weights):
            cum += w
            if r < cum:
                return split
        return "test"  # float rounding fallback

class DataPrepCLI:
    """Command-line interface for ASR data preparation tasks"""
    NAIJAVOICES = ["yoruba", "igbo", "hausa"]
    FLEURS = ["ha_ng", "ig_ng", "yo_ng"]
    OBSA = ["Yoruba", "Igbo", "Hausa"] #open bible speech african
    YAS = ["Yoruba_yor", "Igbo_ibo", "Hausa_hau"] #youversion
    ASPV1 = ["yoruba_yor", "igbo_ibo", "hausa_hau", "pidgin_west_africa_wes"]
    TWB = ["hau"]
    OPEN_SLR = ["yor_ng", "eng_ng"]
    MONOLANG_REPO = {
        "pidgin" : ["asr-nigerian-pidgin", "timniel"],
        "english" : ["nigerian_accented_english_dataset"],
        "igbo" : ["igbo_sync"],
        "yoruba" : ["yfacc", "yecs"]
    }
    HFAPI = HfApi()

    def __init__(self) -> None:
        settings = config.settings
        self.storage = manager.HFDatasetRepo(settings.hf_dataset_repo, settings.hf_token)

    def _persist_metrics(self, accumulator: metrics.MetricsAccumulator, output_dir: str) -> None:
        failed = self.storage.upload([accumulator.state_path], local_dir=output_dir)
        if failed:
            raise ValueError(f"Failed to upload metrics checkpoint {failed}")

    def _resume_completed_shard(
        self,
        entity: dict,
        corpus_ledger: ledger.CorpusLedger,
        accumulator: metrics.MetricsAccumulator,
        output_dir: str,
    ) -> bool:
        parquet_files = [
            Path(p).resolve()
            for p in entity.get("local_dir", [])
            if Path(p).suffix == ".parquet"
        ]
        if not parquet_files or not all(p.is_file() for p in parquet_files):
            corpus_ledger.mark_pending(entity["filename"])
            return False

        stage_root = _staging_root(parquet_files[0])
        if stage_root is not None and all(_staging_root(p) == stage_root for p in parquet_files):
            upload_root = stage_root
        else:
            # Older ledger entries stored shard files under output_dir/data and
            # included the ledger JSON in local_dir. Resume those in place.
            upload_root = (Path(output_dir) / "data").resolve()
            if not all(p.is_relative_to(upload_root) for p in parquet_files):
                corpus_ledger.mark_pending(entity["filename"])
                return False

        partition_values = {
            key: value
            for part in parquet_files[0].parts
            if "=" in part
            for key, value in [part.split("=", 1)]
        }
        if not all(key in partition_values for key in ("corpus", "split", "language")):
            corpus_ledger.mark_pending(entity["filename"])
            return False
        accumulator.update_shard_from_local(
            parquet_files,
            partition_values["corpus"],
            partition_values["split"],
            partition_values["language"],
            identity=entity["filename"],
        )
        self._persist_metrics(accumulator, output_dir)

        failed = self.storage.upload(parquet_files, local_dir=str(upload_root))
        if failed:
            raise ValueError(f"Failed to resume upload {failed}")
        corpus_ledger.mark_uploaded(entity["filename"])
        failed = self.storage.upload([corpus_ledger.repo.path], local_dir=output_dir)
        if failed:
            raise ValueError(f"Failed to upload ledger {failed}")

        if stage_root is not None:
            shutil.rmtree(stage_root, ignore_errors=True)
        else:
            remove_files(parquet_files)
        return True

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

    def _process_batches(self, ds, lang, split, corpus, output_dir,audio_column: str = "audio", remove_numbers=False):
        output_path = Path(output_dir).resolve()
        ray_ds_stream_ = ray.data.from_huggingface(ds)
        
        ray_ds_stream_ = ray_ds_stream_.map_batches(
            text_tools_corpus.TextProcessor,
            fn_constructor_kwargs={"lang" : lang, "remove_numbers" : remove_numbers},
            batch_size=_ray_batch_size(),
            batch_format="pyarrow",
            compute=ray.data.ActorPoolStrategy(size=_ray_workers()),
        )

        ray_ds_stream_ = ray_ds_stream_.map_batches(
            audio_features.AudioFeatureProcessor,
            fn_constructor_kwargs={
                "audio_column" : audio_column,
            },
            batch_size=_ray_batch_size(),
            batch_format="pyarrow",
            compute=ray.data.ActorPoolStrategy(size=_ray_workers()),
        )

        ray_ds_stream_ = ray_ds_stream_.map_batches(
            partial(
                audio_tools.map_to_target_schema,
                split=split,
                corpus=corpus,
                audio_col=audio_column
            ),
            batch_size=_ray_batch_size(),
            batch_format="pyarrow",
            compute=ray.data.TaskPoolStrategy(size=1),
        )

        try:
            ray_ds_stream_.write_parquet(
                str(output_path),
                partition_cols=["corpus", "split", "language"],
                min_rows_per_file=10_000,
                row_group_size=100
            )
            return output_path
        finally:
            # Drop the lazy Ray plan promptly so completed shard references can
            # be reclaimed before the next shard starts.
            del ray_ds_stream_
            gc.collect()

    def _ingest_corpus_internal(
        self,
        output_dir: str,
        corpus_name : str,
        repo_id: str,
        add_lang_config : bool,
        corpus_ledger : ledger.CorpusLedger,
        text_col : str,
        accx: metrics.MetricsAccumulator,
        lang: str | None = None,
        split_map : dict[str, str] | None = None,
        lang_map : dict[str,str] | None = None,
        remove_numbers : bool = False,
        rem_cols: list[str] | None = None,
    ):
        split_map = split_map or {}
        lang_map = lang_map or {}
        entities = corpus_ledger.claim()
        if add_lang_config:
            entities = [e for e in entities if e["filename"].split("/")[-2] == lang]
        for entity in entities:
            if entity["status"] == "completed":
                if self._resume_completed_shard(entity, corpus_ledger, accx, output_dir):
                    continue
            split = entity["filename"].split("/")[-1]
            if not add_lang_config:
                corpus_hf = load_dataset(repo_id, split=split, streaming=True)
            else:
                corpus_hf = load_dataset(repo_id, lang,split=split, streaming=True)
            try:
                columns_to_remove = [column for column in (rem_cols or []) if column in corpus_hf.column_names]
                if columns_to_remove:
                    corpus_hf = corpus_hf.remove_columns(columns_to_remove)

                corpus_hf = corpus_hf.shuffle(seed=42, buffer_size=_shuffle_buffer_size())
                corpus_hf = corpus_hf.cast_column("audio", Audio(decode=False, sampling_rate=16000))
                if text_col != "transcript":
                    corpus_hf = corpus_hf.rename_column(text_col, "transcript")

                corpus_lang = lang_map.get(lang or "", lang or "")
                output_split = split_map.get(split, split)
                shard_dir = Path(output_dir) / "staging" / _shard_id(entity["filename"])
                if shard_dir.exists():
                    shutil.rmtree(shard_dir)
                shard_dir = shard_dir.resolve()
                self._process_batches(corpus_hf, corpus_lang, output_split, corpus_name, str(shard_dir), remove_numbers=remove_numbers)
                files = _unique_parquet_names(list(shard_dir.rglob("*.parquet")), _shard_id(entity["filename"]))
                if not files:
                    raise RuntimeError(f"No Parquet files produced for shard {entity['filename']}")

                accx.update_shard_from_local(
                    files, corpus_name, output_split, _language_partition(corpus_lang),
                    identity=entity["filename"],
                )
                self._persist_metrics(accx, output_dir)

                corpus_ledger.mark_completed(entity["filename"], files)
                failed = self.storage.upload(files, local_dir=str(shard_dir))

                if len(failed) != 0:
                    raise ValueError(f"Failed to upload {failed}")

                corpus_ledger.mark_uploaded(entity["filename"])
            finally:
                # Hugging Face streaming datasets retain iterator/shuffle state;
                # release it before moving on even when this shard fails.
                del corpus_hf
                gc.collect()
            failed = self.storage.upload([corpus_ledger.repo.path], local_dir=output_dir)
            if failed:
                raise ValueError(f"Failed to upload ledger {failed}")
            shutil.rmtree(shard_dir, ignore_errors=True)

    def _ingest_naijavoices_internal(self, output_dir: str, accx: metrics.MetricsAccumulator , lang_subset: list[str] |None = None):
        repo_id = "naijavoices/naijavoices-dataset"
        languages = self.NAIJAVOICES if not lang_subset else lang_subset
        lang_map = {
            "hausa" : "hau_ng",
            "igbo" : "ibo_ng",
            "yoruba" : "yor_ng"
        }
        batches = ["-batch-0", "-batch-1", "-batch-2"]
        folders = [l + batch for l in languages for batch in batches]
        corpus = {
            "train" : [],
            "val" : [],
            "test" : []
        }
        corpus_ledger = ledger.CorpusLedger(output_dir + "/ledger", repo_id=repo_id, repo_type="dataset")
        hf_fs = HfFileSystem(token=config.settings.hf_token)
        for folder in folders:
            for file in self.HFAPI.list_repo_tree(repo_id, path_in_repo=folder, repo_type="dataset"):
                fpath = file.path
                sp = assign_split(fpath)
                corpus[sp].append(fpath)
                corpus_ledger.register_file(f"{sp}^{fpath}" )

        for entity in corpus_ledger.claim():
            if entity["status"] == "completed":
                if self._resume_completed_shard(entity, corpus_ledger, accx, output_dir):
                    continue
            split, fp = entity["filename"].split("^")
            lang = fp.split("-")[0]
            lang = lang_map.get(lang) or lang
            parquet_path = f"hf://datasets/{repo_id}/{fp}"
            # Read Parquet as Arrow so Hugging Face's Audio feature decoder does
            # not invoke TorchCodec while Ray is reading the source file.
            ray_ds = ray.data.read_parquet(parquet_path, filesystem=hf_fs)
            ray_ds = ray_ds.rename_columns({"text": "transcript"})
            ray_ds = ray_ds.map_batches(
                text_tools_corpus.TextProcessor,
                fn_constructor_kwargs={"lang" : lang, "remove_numbers" : False},
                batch_size=_ray_batch_size(),
                batch_format="pyarrow",
                compute=ray.data.ActorPoolStrategy(size=_ray_workers()),
            )
            
            ray_ds = ray_ds.map_batches(
                audio_features.AudioFeatureProcessor,
                fn_constructor_kwargs={
                    "audio_column" : "audio",
                },
                batch_size=_ray_batch_size(),
                batch_format="pyarrow",
                compute=ray.data.ActorPoolStrategy(size=_ray_workers()),
            )
            
            ray_ds = ray_ds.map_batches(
                partial(
                    audio_tools.map_to_target_schema,
                    split=split,
                    corpus="naijavoices",
                    audio_col="audio"
                ),
                batch_size=_ray_batch_size(),
                batch_format="pyarrow",
                compute=ray.data.TaskPoolStrategy(size=1),
            )
    
            shard_dir = (Path(output_dir) / "staging" / _shard_id(entity["filename"])).resolve()
            if shard_dir.exists():
                shutil.rmtree(shard_dir)
            ray_ds.write_parquet(
                str(shard_dir),
                partition_cols=["corpus", "split", "language"],
                min_rows_per_file=10_000,
                row_group_size=100
            )

            files = _unique_parquet_names(list(shard_dir.rglob("*.parquet")), _shard_id(entity["filename"]))
            if not files:
                raise RuntimeError(f"No Parquet files produced for shard {entity['filename']}")
            accx.update_shard_from_local(
                files, "naijavoices", split, _language_partition(lang),
                identity=entity["filename"],
            )
            self._persist_metrics(accx, output_dir)
            corpus_ledger.mark_completed(entity["filename"], files)
            failed = self.storage.upload(files, local_dir=str(shard_dir))
            if len(failed) != 0:
                raise ValueError(f"Failed to upload {failed}")

            corpus_ledger.mark_uploaded(entity["filename"])
            failed = self.storage.upload([corpus_ledger.repo.path], local_dir=output_dir)
            if failed:
                raise ValueError(f"Failed to upload ledger {failed}")
            shutil.rmtree(shard_dir, ignore_errors=True)

    def _ingest_yfacc_internal(
        self, output_dir: str, accx: metrics.MetricsAccumulator, lang_subset: list[str] | None = None
    ):
        # see https://huggingface.co/datasets/nolimitsxl/yfacc_yoruba
        repo_id = "nolimitsxl/yfacc_yoruba"
        splits = ["test", "val", "train"]
        corpus_ledger = ledger.CorpusLedger(output_dir + "/ledger", repo_id=repo_id, repo_type="dataset")
        corpus_ledger.register_files([f"{repo_id}/{split}" for split in splits])
        self._ingest_corpus_internal(output_dir, "YFACC", repo_id, False, corpus_ledger,"transcript", accx, "yor_ng",rem_cols=["language_id_per_token"])
        
    def _ingest_yecs_internal(
            self, output_dir: str, accx: metrics.MetricsAccumulator, lang_subset: list[str] | None = None
        ):
            # see https://huggingface.co/datasets/nolimitsxl/yecs_lyngual_labs
            repo_id = "nolimitsxl/yecs_lyngual_labs"
            splits = ["val", "train"]
            corpus_ledger = ledger.CorpusLedger(output_dir + "/ledger", repo_id=repo_id, repo_type="dataset")
            corpus_ledger.register_files([f"{repo_id}/{split}" for split in splits])
            self._ingest_corpus_internal(output_dir, "YECS_LYNGUAL_LABS", repo_id, False, corpus_ledger, "transcript", accx, "yor_ng", rem_cols=["language_id_per_token"])

    def _ingest_igbo_sync_internal(
            self, output_dir: str, accx: metrics.MetricsAccumulator, lang_subset: list[str] | None = None
        ):
        # see https://huggingface.co/datasets/nolimitsxl/igbo_sync_processed
        repo_id = "nolimitsxl/igbo_sync_processed"
        splits = ["val", "train"]
        corpus_ledger = ledger.CorpusLedger(output_dir + "/ledger", repo_id=repo_id, repo_type="dataset")
        corpus_ledger.register_files([f"{repo_id}/{split}" for split in splits])
        self._ingest_corpus_internal(output_dir, "Igbo_sync", repo_id, False, corpus_ledger, "transcript", accx,"ibo_ng", rem_cols=["language_id_per_token"])

    def _ingest_naed_internal(
            self, output_dir: str, accx: metrics.MetricsAccumulator, lang_subset: list[str] | None = None
        ):
        # see https://huggingface.co/datasets/benjaminogbonna/nigerian_accented_english_dataset
        repo_id = "benjaminogbonna/nigerian_accented_english_dataset"
        splits = ["test","validation", "train"]
        corpus_ledger = ledger.CorpusLedger(output_dir + "/ledger", repo_id=repo_id, repo_type="dataset")
        corpus_ledger.register_files([f"{repo_id}/{split}" for split in splits])
        split_remap = {"validation" : "val"}
        self._ingest_corpus_internal(output_dir, "nigerian_accented_english_dataset", repo_id, False, corpus_ledger, "sentence", accx, "eng_ng", split_map=split_remap)

    def _ingest_ud_naija_nsc_internal(self, output_dir: str, accx: metrics.MetricsAccumulator, lang_subset: list[str] | None = None):
        # see https://huggingface.co/datasets/timniel/Pidgin_ASR_Dataset_Combined
        repo_id = "timniel/Pidgin_ASR_Dataset_Combined"
        splits = ["train"]
        corpus_ledger = ledger.CorpusLedger(output_dir + "/ledger", repo_id=repo_id, repo_type="dataset")
        corpus_ledger.register_files([f"{repo_id}/{split}" for split in splits])
        self._ingest_corpus_internal(output_dir, "UD NAIJA NSC", repo_id, False, corpus_ledger, "text", accx,"pcm_ng")

    def _ingest_asr_nigerian_pidgin_internal(
            self, output_dir: str, accx: metrics.MetricsAccumulator, lang_subset: list[str] | None = None
            ):
        # see https://huggingface.co/datasets/asr-nigerian-pidgin/nigerian-pidgin-1.0
        repo_id = "asr-nigerian-pidgin/nigerian-pidgin-1.0"
        splits = ["train", "validation", "test"]
        corpus_ledger = ledger.CorpusLedger(output_dir + "/ledger", repo_id=repo_id, repo_type="dataset")
        corpus_ledger.register_files([f"{repo_id}/{split}" for split in splits])
        split_remap = {"validation" : "val"}
        self._ingest_corpus_internal(output_dir, "nigerian-pidgin-1.0", repo_id, False, corpus_ledger, "sentence", accx,"pcm_ng",split_map=split_remap)

    def _ingest_open_slr_internal(self, output_dir: str, accx: metrics.MetricsAccumulator, lang_subset: list[str] | None = None):
        repo_id = "nolimitsxl/open_slr_lang_resource"
        corpus_ledger = ledger.CorpusLedger(output_dir + "/ledger", repo_id=repo_id, repo_type="dataset")
        splits = ["train", "test"]
        if not lang_subset:
            lang_subset = self.OPEN_SLR
        for lang in lang_subset:
            if lang not in self.OPEN_SLR:
                print(f"{lang} does not exist. Skipping...")
                continue
            
            corpus_ledger.register_files([f"{repo_id}/{lang}/{split}" for split in splits])
            self._ingest_corpus_internal(output_dir, "Open SLR", repo_id, True, corpus_ledger, "transcript", accx,lang, rem_cols=["language_id_per_token"])

    def _ingest_twb_internal(self, output_dir: str, accx: metrics.MetricsAccumulator, lang_subset: list[str] | None = None):
        repo_id = "CLEAR-Global/TWB-Voice-1.0"
        splits = ["train", "dev", "test"]
        split_remap = {"dev" : "val"}
        lang_remap = {"hau": "hau_ng"}
        corpus_ledger = ledger.CorpusLedger(output_dir + "/ledger", repo_id=repo_id, repo_type="dataset")
        if not lang_subset:
            lang_subset = self.TWB
        for lang in lang_subset:
            if lang not in self.TWB:
                print(f"{lang} does not exist. Skipping...")
                continue
            corpus_ledger.register_files([f"{repo_id}/{lang}/{split}" for split in splits])
            self._ingest_corpus_internal(output_dir, "clearVoiceTWB", repo_id, True, corpus_ledger, "sentence", accx,lang, split_map=split_remap, lang_map=lang_remap)
        
    def _ingest_aspv1_internal(self, output_dir: str, accx: metrics.MetricsAccumulator, lang_subset: list[str] | None = None):
        repo_id = "AfriSpeech/african-speech-public_v1"
        splits = ["train", "validation", "test"]
        split_remap = {"validation" : "val"}
        lang_remap = {
            "yoruba_yor" : "yor_ng",
            "igbo_ibo" : "ibo_ng",
            "hausa_hau" : "hau_ng",
            "pidgin_west_africa_wes" : "pcm_ng"
        }
        corpus_ledger = ledger.CorpusLedger(output_dir + "/ledger", repo_id=repo_id, repo_type="dataset")
        if not lang_subset:
            lang_subset = self.ASPV1
        for lang in lang_subset:
            if lang not in self.ASPV1:
                print(f"{lang} does not exist. Skipping...")
                continue
            corpus_ledger.register_files([f"{repo_id}/{lang}/{split}" for split in splits])
            self._ingest_corpus_internal(output_dir, "afrispeechASP", repo_id, True, corpus_ledger, "text", accx,lang, split_map=split_remap, lang_map=lang_remap, remove_numbers=True)
    
    def _ingest_obsa_internal(self, output_dir: str, accx: metrics.MetricsAccumulator, lang_subset: list[str] | None = None):
        repo_id = "AfriSpeech/open-bible-speech-african"
        lang_remap = {
            "Hausa" : "hau_ng",
            "Yoruba" : "yor_ng",
            "Igbo" : "ibo_ng"
        }
        corpus_ledger = ledger.CorpusLedger(output_dir + "/ledger", repo_id=repo_id, repo_type="dataset")
        if not lang_subset:
            lang_subset = self.OBSA
        for lang in lang_subset:
            if lang not in self.OBSA:
                print(f"{lang} does not exist. Skipping...")
                continue

            # Process the repository's Parquet shards individually. Loading the
            # language as one Hugging Face dataset creates a large ReadHuggingFace
            # task; per-file streaming keeps each read bounded to one shard.
            files = [
                item.path
                for item in self.HFAPI.list_repo_tree(
                    repo_id,
                    path_in_repo=lang,
                    recursive=True,
                    repo_type="dataset",
                )
                if getattr(item, "path", "").lower().endswith(".parquet")
            ]
            corpus_ledger.register_files(
                [
                    f"{Path(file_path).name.split('-', 1)[0].split('.', 1)[0]}^{file_path}"
                    for file_path in files
                ]
            )

        for entity in corpus_ledger.claim():
            # Ignore legacy whole-split ledger keys created by the previous
            # OBSA loader; current keys are `<split>^<parquet path>`.
            if "^" not in entity["filename"]:
                continue
            if entity["status"] == "completed":
                if self._resume_completed_shard(entity, corpus_ledger, accx, output_dir):
                    continue

            split, file_path = entity["filename"].split("^", 1)
            language = file_path.split("/", 1)[0]
            if language not in lang_subset:
                continue
            split = {"validation": "val", "dev": "val"}.get(split, split)
            corpus_lang = lang_remap[language]
            parquet_url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/{file_path}"
            corpus_hf = load_dataset(
                "parquet",
                data_files=parquet_url,
                split="train",
                streaming=True,
            )
            ray_ds = None
            try:
                corpus_hf = corpus_hf.rename_column("text", "transcript")
                corpus_hf = corpus_hf.cast_column("audio", Audio(decode=False, sampling_rate=16_000))
                ray_ds = ray.data.from_huggingface(corpus_hf, concurrency=1)
                ray_ds = ray_ds.map_batches(
                    text_tools_corpus.TextProcessor,
                    fn_constructor_kwargs={"lang": corpus_lang, "remove_numbers": True},
                    batch_size=_ray_batch_size(),
                    batch_format="pyarrow",
                    compute=ray.data.ActorPoolStrategy(size=_ray_workers()),
                )
                ray_ds = ray_ds.map_batches(
                    audio_features.AudioFeatureProcessor,
                    fn_constructor_kwargs={"audio_column": "audio"},
                    batch_size=_ray_batch_size(),
                    batch_format="pyarrow",
                    compute=ray.data.ActorPoolStrategy(size=_ray_workers()),
                )
                ray_ds = ray_ds.map_batches(
                    partial(
                        audio_tools.map_to_target_schema,
                        split=split,
                        corpus="afriSpeechOpenBibleSpeech",
                        audio_col="audio",
                    ),
                    batch_size=_ray_batch_size(),
                    batch_format="pyarrow",
                    compute=ray.data.TaskPoolStrategy(size=1),
                )

                shard_dir = (Path(output_dir) / "staging" / _shard_id(entity["filename"])).resolve()
                if shard_dir.exists():
                    shutil.rmtree(shard_dir)
                ray_ds.write_parquet(
                    str(shard_dir),
                    partition_cols=["corpus", "split", "language"],
                    min_rows_per_file=10_000,
                    row_group_size=100,
                )

                output_files = _unique_parquet_names(
                    list(shard_dir.rglob("*.parquet")), _shard_id(entity["filename"])
                )
                if not output_files:
                    raise RuntimeError(f"No Parquet files produced for shard {entity['filename']}")
                accx.update_shard_from_local(
                    output_files,
                    "afriSpeechOpenBibleSpeech",
                    split,
                    _language_partition(corpus_lang),
                    identity=entity["filename"],
                )
                self._persist_metrics(accx, output_dir)
                corpus_ledger.mark_completed(entity["filename"], output_files)
                failed = self.storage.upload(output_files, local_dir=str(shard_dir))
                if failed:
                    raise ValueError(f"Failed to upload {failed}")

                corpus_ledger.mark_uploaded(entity["filename"])
                failed = self.storage.upload([corpus_ledger.repo.path], local_dir=output_dir)
                if failed:
                    raise ValueError(f"Failed to upload ledger {failed}")
                shutil.rmtree(shard_dir, ignore_errors=True)
            finally:
                del ray_ds
                del corpus_hf
                gc.collect()
    
    def _ingest_yas_internal(self, output_dir: str, accx: metrics.MetricsAccumulator, lang_subset: list[str] | None = None):
        repo_id = "AfriSpeech/youversion-african-speech"
        lang_remap = {
            "Hausa_hau" : "hau_ng",
            "Yoruba_yor" : "yor_ng",
            "Igbo_ibo" : "ibo_ng"
        }
        splits = ["train"]
        corpus_ledger = ledger.CorpusLedger(output_dir + "/ledger", repo_id=repo_id, repo_type="dataset")
        if not lang_subset:
            lang_subset = self.YAS
        for lang in lang_subset:
            if lang not in self.YAS:
                print(f"{lang} does not exist. Skipping...")
                continue
            corpus_ledger.register_files([f"{repo_id}/{lang}/{split}" for split in splits])
            self._ingest_corpus_internal(output_dir, "YouVersionAfricanSpeech", repo_id, True, corpus_ledger, "text", accx,lang, lang_map=lang_remap, remove_numbers=True)
    
    def _ingest_fleurs_internal(self, output_dir: str, accx: metrics.MetricsAccumulator, lang_subset: list[str] | None = None):
        repo_id = "google/fleurs"
        splits = ["train","validation","test"]
        lang_remap = {
            "ha_ng" : "hau_ng",
            "yo_ng" : "yor_ng",
            "ig_ng" : "ibo_ng"
        }
        split_remap = {
            "validation": "val"
        }
        corpus_ledger = ledger.CorpusLedger(output_dir + "/ledger", repo_id=repo_id, repo_type="dataset")
        if not lang_subset:
            lang_subset = self.FLEURS
        for lang in lang_subset:
            if lang not in self.FLEURS:
                print(f"{lang} does not exist. Skipping...")
                continue

            corpus_ledger.register_files([f"{repo_id}/{lang}/{split}" for split in splits])
            self._ingest_corpus_internal(output_dir, "fleurs", repo_id, True, corpus_ledger, "transcription", accx,lang, lang_map=lang_remap,split_map=split_remap)
    
    def ingest_yfacc(self, output_dir: str, accx: metrics.MetricsAccumulator | None = None):
        """Ingest YFACC datasets.

        Args:
            output_dir: Output directory path for processed Parquet files

        """
        accx = self._resolve_accumulator(output_dir, accx)
        print(f"Starting YFACC ingestion to: {output_dir}")
        self._ingest_yfacc_internal(output_dir, accx=accx)
        print("YFACC ingestion completed")
    
    def ingest_yecs(self, output_dir: str, accx: metrics.MetricsAccumulator | None = None):
        """Ingest YECS datasets.
        
        Args:
            output_dir: Output directory path for processed Parquet files

        """
        accx = self._resolve_accumulator(output_dir, accx)
        print(f"Starting YECS ingestion to: {output_dir}")
        self._ingest_yecs_internal(output_dir, accx=accx)
        print("YECS ingestion completed")

    def ingest_igbo_sync(self, output_dir: str, accx: metrics.MetricsAccumulator | None = None):
        """Ingest Igbo Sync datasets.
        
        Args:
            output_dir: Output directory path for processed Parquet files

        """
        accx = self._resolve_accumulator(output_dir, accx)
        print(f"Starting Igbo Sync ingestion to: {output_dir}")
        self._ingest_igbo_sync_internal(output_dir, accx=accx)
        print("Igbo Sync ingestion completed")

    def ingest_naed_sync(self, output_dir: str, accx: metrics.MetricsAccumulator | None = None):
        """Ingest NAED datasets.
        
        Args:
            output_dir: Output directory path for processed Parquet files

        """
        accx = self._resolve_accumulator(output_dir, accx)
        print(f"Starting NAED ingestion to: {output_dir}")
        self._ingest_naed_internal(output_dir, accx=accx)
        print("NAED ingestion completed")

    def ingest_asr_nigerian_pidgin(self, output_dir: str, accx: metrics.MetricsAccumulator | None = None):
        """Ingest ASR Nigerian Pidgin  datasets.
                
        Args:
            output_dir: Output directory path for processed Parquet files

        """
        accx = self._resolve_accumulator(output_dir, accx)
        print(f"Starting ASR Nigerian Pidgin ingestion to: {output_dir}")
        self._ingest_asr_nigerian_pidgin_internal(output_dir, accx=accx)
        print("ASR Nigerian Pidgin ingestion completed")

    def ingest_ud_naija_nsc(self, output_dir: str, accx: metrics.MetricsAccumulator | None = None):
        """Ingest ASR Nigerian Pidgin  datasets.
                
        Args:
            output_dir: Output directory path for processed Parquet files

        """
        accx = self._resolve_accumulator(output_dir, accx)
        print(f"Starting UD Naija NSC ingestion to: {output_dir}")
        self._ingest_ud_naija_nsc_internal(output_dir, accx=accx)
        print("UD Naija NSC ingestion completed")

    def ingest_open_slr(self, output_dir: str, accx: metrics.MetricsAccumulator | None = None):
        """Ingest Open SLR dataset.
                            
        Args:
            output_dir: Output directory path for processed Parquet files

        """
        accx = self._resolve_accumulator(output_dir, accx)
        print(f"Starting Open SLR ingestion to: {output_dir}")
        self._ingest_open_slr_internal(output_dir, accx=accx)
        print("OpenSLR ingestion completed")

    def ingest_twb(self, output_dir: str, accx: metrics.MetricsAccumulator | None = None):
        """Ingest TWB dataset.
                            
        Args:
            output_dir: Output directory path for processed Parquet files

        """
        accx = self._resolve_accumulator(output_dir, accx)
        print(f"Starting TWB ingestion to: {output_dir}")
        self._ingest_twb_internal(output_dir, accx=accx)
        print("TWB ingestion completed") 

    def ingest_aspv1(self, output_dir: str, accx: metrics.MetricsAccumulator | None = None):
        """Ingest AfriSpeech/ASPV1 dataset.
                                    
        Args:
            output_dir: Output directory path for processed Parquet files

        """
        accx = self._resolve_accumulator(output_dir, accx)
        print(f"Starting AfriSpeech ingestion to: {output_dir}")
        self._ingest_aspv1_internal(output_dir, accx=accx)
        print("ASPV1 ingestion completed") 

    def ingest_yas(self, output_dir: str, accx: metrics.MetricsAccumulator | None = None):
        """Ingest AfriSpeech/Youversion-African-Speech dataset.
                                   , accx=accx 
        Args:
            output_dir: Output directory path for processed Parquet files

        """
        accx = self._resolve_accumulator(output_dir, accx)
        print(f"Starting AfriSpeech/Youversion-African-Speech ingestion to: {output_dir}")
        self._ingest_yas_internal(output_dir, accx=accx)
        print("AfriSpeech/Youversion-African-Speech ingestion completed") 

    def ingest_obsa(self, output_dir: str, accx: metrics.MetricsAccumulator | None = None):
        """Ingest AfriSpeech/Open Bible Speech dataset.
                                    
        Args:
            output_dir: Output directory path for processed Parquet files

        """
        accx = self._resolve_accumulator(output_dir, accx)
        print(f"Starting AfriSpeech/Open Bible Speech ingestion to: {output_dir}")
        self._ingest_obsa_internal(output_dir, accx=accx)
        print("AfriSpeech/Open Bible Speech ingestion completed") 

    def ingest_fleurs(self, output_dir: str, accx: metrics.MetricsAccumulator | None = None):
        """Ingest FLEURS dataset.
                                    
        Args:
            output_dir: Output directory path for processed Parquet files

        """
        accx = self._resolve_accumulator(output_dir, accx)
        print(f"Starting FLEURS ingestion to: {output_dir}")
        self._ingest_fleurs_internal(output_dir, accx=accx)
        print("FLEURS completed")

    def ingest_naijavoices(self, output_dir: str, accx: metrics.MetricsAccumulator | None = None):
        """Ingest NaijaVoices dataset.
                                            
        Args:
            output_dir: Output directory path for processed Parquet files

        """
        accx = self._resolve_accumulator(output_dir, accx)
        print(f"Starting NaijaVoices ingestion to: {output_dir}")
        self._ingest_naijavoices_internal(output_dir,accx=accx)
        print("NaijaVoices completed")

    def run_set(self, output_dir: str, load_from_hf: bool = False):
        corpii = [self.ingest_yfacc, self.ingest_yecs, self.ingest_igbo_sync, self.ingest_naed_sync, self.ingest_asr_nigerian_pidgin, self.ingest_ud_naija_nsc, self.ingest_open_slr, self.ingest_twb, self.ingest_aspv1, self.ingest_yas, self.ingest_fleurs, self.ingest_naijavoices, self.ingest_obsa]
        if load_from_hf:
            self.storage.download(output_dir + "/ledger", "ledger")
            self.storage.download(output_dir + "/metrics/metrics.pkl", "metrics/metrics.pkl")
        accumulator = metrics.MetricsAccumulator(output_dir + "/metrics/metrics.pkl")

        for corpus in corpii:
            print(f"Processing  {corpus.__name__}")
            corpus(output_dir, accx=accumulator)
            gc.collect()
            print(f"{corpus.__name__} completed")
        metrics_dir = Path(output_dir) / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        tsv_path = metrics_dir / "metrics.tsv"
        accumulator.write_tsv(str(tsv_path))
        accumulator.save()
        failed = self.storage.upload([accumulator.state_path, tsv_path], local_dir=output_dir)
        if failed:
            raise ValueError(f"Failed to upload final metrics {failed}")
    

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
