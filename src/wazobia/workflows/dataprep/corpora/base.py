import gc
import os
import shutil
from collections import defaultdict
from collections.abc import Mapping, Sequence
from functools import partial
from itertools import product
from pathlib import Path

import ray
import pyarrow.parquet as pq
from datasets import Audio, Features, IterableDataset, load_dataset
from huggingface_hub import HfFileSystem

from wazobia import config as env_config
from wazobia.file_manager import manager
from wazobia.ledger import ledger
from wazobia.workflows.dataprep import (
    audio_features,
    audio_tools,
    metrics,
    text_tools_corpus,
    utils,
)

from . import config


def _ray_batch_size() -> int:
    return max(1, int(os.environ.get("WAZOBIA_RAY_BATCH_SIZE", "16")))

def _ray_workers() -> int:
    return max(1, int(os.environ.get("WAZOBIA_RAY_WORKERS", "1")))

def _language_partition(lang: str) -> str:
    return lang.removesuffix("_ng")

def _shuffle_buffer_size() -> int:
    return max(1, int(os.environ.get("WAZOBIA_SHUFFLE_BUFFER_SIZE", "256")))

def _unique_parquet_names(files: list[Path], shard_id: str) -> list[Path]:
    renamed = []
    for index, file in enumerate(sorted(files)):
        target = file.with_name(f"{shard_id}-{index:05d}.parquet")
        file.rename(target)
        renamed.append(target)
    return renamed


def _compile_manifest(files: Sequence[Path], output_dir: str, corpus: str, split: str, language: str) -> Path:
    """Append normalized transcription rows from a processed shard to its split manifest."""
    manifest = Path(output_dir) / "data" / f"corpus={corpus}" / f"split={split}" / f"language={language}" / "transcript_manifest.txt"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("a", encoding="utf-8", newline="\n") as target:
        for parquet_file in sorted(files):
            table = pq.read_table(parquet_file, columns=["text"])
            for text in table.column("text").to_pylist():
                if text:
                    target.write(text.replace("\r", " ").replace("\n", " ") + "\n")
    return manifest


class CorpusCompiler:
    def __init__(self):
        self.register_strategy = {
           "parquet" : self._register_files_by_parquet,
            "base" : self._register_files_by_streaming
        }

        self.ingest_strategy = {
            "parquet" : self._ingest_corpus_parquet,
            "base" : self._ingest_corpus_stream
        }


    def __call__(self,
        output_dir: str,
        accx: metrics.MetricsAccumulator,
        ds_config: config.Corpus,
        storage: manager.HFDatasetRepo):
        ledger_pth = output_dir + "/ledger"
        strategy = ds_config.ingest_strategy
        corpus_ledger = ledger.CorpusLedger(ledger_pth, repo_id=ds_config.repo_id, repo_type="dataset")
        corpus_content = self.register_strategy[strategy](corpus_ledger, ds_config)
        self.ingest_strategy[strategy](output_dir, ds_config, corpus_ledger, accx, storage, corpus_content)
        print("Done!!!")


    def get_files_from_path(self, fp: str) -> list[str]:
        fs = HfFileSystem(token=env_config.settings.hf_token)

        files = fs.glob(fp)
        return files

    def _process_batches(self, ray_ds_stream_: ray.data.Dataset, lang, split, corpus, output_path, unuttered_words: Sequence[str] | None = None,audio_column: str = "audio", remove_numbers=False, normalizer: str | None = None):
        
        ray_ds_stream_ = ray_ds_stream_.map_batches(
            text_tools_corpus.TextProcessor,
            fn_constructor_kwargs={"lang" : lang, "remove_numbers" : remove_numbers, "unuttered_words": unuttered_words, "normalizer": normalizer},
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

    def _register_files_by_parquet(self,
        corpus_ledger: ledger.CorpusLedger,
        ds_config: config.Corpus) -> dict[tuple, list[str]]:
        corpus_content = defaultdict(list)
        for old_lang in ds_config.languages:
            dir_formatting = ds_config.dir_format
            split_weight = ds_config.split_weight or []
            if len(split_weight) != 0:
                fpath = f"hf://datasets/{ds_config.repo_id}/" + dir_formatting.format(lang=old_lang, split="*")
                fpath = fpath.replace("**", "*")
                files = self.get_files_from_path(fpath)
                for file in files:
                    new_split = utils.assign_split(file, split_weight)
                    corpus_content[(old_lang, new_split)].append(file)
                    corpus_ledger.register_files([file])
                continue
            for old_split in ds_config.splits:
                fpath = f"hf://datasets/{ds_config.repo_id}/" + dir_formatting.format(lang=old_lang, split=old_split)
                # A split pattern and a wildcard in dir_format can otherwise
                # combine into `**`, which fsspec rejects inside a path segment.
                fpath = fpath.replace("**", "*")
                files = self.get_files_from_path(fpath)
                if not files:
                    raise FileNotFoundError(
                        f"No Parquet files found for {ds_config.repo_id}, "
                        f"language {old_lang!r}, split {old_split!r}; glob was {fpath!r}"
                    )
                n_key = (old_lang, old_split)
                cur_files = corpus_content.get(n_key,[])
                cur_files.extend(files)
                corpus_content[n_key] = cur_files
                corpus_ledger.register_files(files)
        return corpus_content

    def _register_files_by_streaming(self,
        corpus_ledger: ledger.CorpusLedger,
        ds_config: config.Corpus) -> None:
        corpus_content = {}
        splits = list(ds_config.splits.keys())
        fpaths = []
        if ds_config.include_lang_as_arg:
            langs = list(ds_config.languages.keys())
            combinations = list(product(langs, splits))
            for lang, split in combinations:
                fp = f"{ds_config.repo_id}/{lang}/{split}"
                fpaths.append(fp)
                corpus_content[(lang,split)] = [fp]
        else:
            lang = next(iter(ds_config.languages.keys()))
            for split in splits:
                fp = f"{ds_config.repo_id}/{split}"
                fpaths.append(fp)
                corpus_content[(lang,split)] = [fp]
        corpus_ledger.register_files(fpaths)
        return corpus_content

    def _resume_completed_shard(
            self,
            entity: dict,
            corpus_ledger: ledger.CorpusLedger,
            accumulator: metrics.MetricsAccumulator,
            output_dir: str,
            storage: manager.HFDatasetRepo,
        ) -> bool:
            parquet_files = [
                Path(p).resolve()
                for p in entity.get("local_dir", [])
                if Path(p).suffix == ".parquet"
            ]
            if not parquet_files or not all(p.is_file() for p in parquet_files):
                corpus_ledger.mark_pending(entity["filename"])
                return False
    
            stage_root = utils.staging_root(parquet_files[0])
            if stage_root is not None and all(utils.staging_root(p) == stage_root for p in parquet_files):
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
            self._persist_metrics(accumulator, output_dir, storage)
    
            failed = storage.upload(parquet_files, local_dir=str(upload_root))
            if failed:
                raise ValueError(f"Failed to resume upload {failed}")
            corpus_ledger.mark_uploaded(entity["filename"])
            failed = storage.upload([corpus_ledger.repo.path], local_dir=output_dir)
            if failed:
                raise ValueError(f"Failed to upload ledger {failed}")
    
            if stage_root is not None:
                shutil.rmtree(stage_root, ignore_errors=True)
            else:
                utils.remove_files(parquet_files)
            return True

    def _prepare_data(
        self,
        ds: IterableDataset,
        remove_cols: Sequence[str] | None = None,
        rename_cols: Mapping[str, str] | None = None,
    ):
        # 1. Disable Audio decoding FIRST on the raw dataset so it reaches the root
        # ArrowExamplesIterable before shuffle/remove/rename wrap the ex_iterable.
        if ds.features:
            for col_name, feat in list(ds.features.items()):
                if isinstance(feat, Audio):
                    ds = ds.cast_column(col_name, Audio(decode=False))

        # 2. Now apply column removals, shuffle, and renames
        remove_cols = [
            column for column in (remove_cols or []) if column in ds.column_names
        ]
        if remove_cols:
            ds = ds.remove_columns(remove_cols)

        ds = ds.shuffle(seed=42, buffer_size=_shuffle_buffer_size())

        rename_cols = {
            source: target
            for source, target in (rename_cols or {}).items()
            if source != target
            and source in ds.column_names
            and target not in ds.column_names
        }
        if rename_cols:
            ds = ds.rename_columns(rename_cols)

        return ds

    def _persist_metrics(self, accumulator: metrics.MetricsAccumulator, output_dir: str, storage: manager.HFDatasetRepo) -> None:
        failed = storage.upload([accumulator.state_path], local_dir=output_dir)
        if failed:
            raise ValueError(f"Failed to upload metrics checkpoint {failed}")

    def _ingest_corpus_stream(self,
        output_dir: str,
        ds_config: config.Corpus,
        corpus_ledger: ledger.CorpusLedger,
        accx: metrics.MetricsAccumulator,
        storage: manager.HFDatasetRepo,
        corpus_content: Mapping[tuple, str]
    ):
        entities = corpus_ledger.claim()
        for (lang, split), source_files in corpus_content.items():
            source_file_set = set(source_files)
            selected_entities = [x for x in entities if x["filename"] in source_file_set]
            for entity in selected_entities:
                if entity["status"] == "completed" and \
                    self._resume_completed_shard(entity, corpus_ledger, accx, output_dir, storage):
                        continue
                if ds_config.include_lang_as_arg:
                    corpus_hf = load_dataset(ds_config.repo_id, lang, split=split, streaming=True)
                else:
                    corpus_hf = load_dataset(ds_config.repo_id, split=split, streaming=True)
                try:
                    corpus_hf = self._prepare_data(corpus_hf, ds_config.exclude_cols, ds_config.columns)
                    new_lang = ds_config.languages.get(lang) or lang
                    new_split = ds_config.splits.get(split) or split
                    shard_dir = Path(output_dir) / "staging" / utils.shard_id(entity["filename"])
                    name = (ds_config.corpus_name or ds_config.repo_id.split("/")[-1]).lower()
                    ray_ds_stream_ = ray.data.from_huggingface(corpus_hf)
                    u_words = (ds_config.exclude_annotations or {}).get(new_lang)
                    self._process_batches(ray_ds_stream_, new_lang, new_split, name, shard_dir.resolve(), unuttered_words=u_words, normalizer=ds_config.normalize)
                    out_files = _unique_parquet_names(list(shard_dir.rglob("*.parquet")), utils.shard_id(entity["filename"]))
                    if not out_files:
                        raise RuntimeError(f"No Parquet files produced for shard {entity['filename']}")

                    if ds_config.compile_text is True or split in (ds_config.compile_text or []):
                        manifest = _compile_manifest(out_files, output_dir, name, new_split, _language_partition(new_lang))
                        failed = storage.upload([manifest], local_dir=output_dir)
                        if failed:
                            raise ValueError(f"Failed to upload transcript manifest {failed}")

                    accx.update_shard_from_local(out_files, name, new_split, _language_partition(new_lang), identity=entity["filename"])

                    self._persist_metrics(accx, output_dir, storage)

                    corpus_ledger.mark_completed(entity["filename"], out_files)
                    failed = storage.upload(out_files, local_dir=str(shard_dir))

                    if len(failed) != 0:
                        raise ValueError(f"Failed to upload {failed}")

                    corpus_ledger.mark_uploaded(entity["filename"])
                finally:
                    del corpus_hf
                    gc.collect()
                failed = storage.upload([corpus_ledger.repo.path], local_dir=output_dir)
                if failed:
                    raise ValueError(f"Failed to upload ledger {failed}")
                shutil.rmtree(shard_dir, ignore_errors=True)

    def _ingest_corpus_parquet(
        self,
        output_dir: str,
        ds_config: config.Corpus,
        corpus_ledger: ledger.CorpusLedger,
        accx: metrics.MetricsAccumulator,
        storage: manager.HFDatasetRepo,
        corpus_content: Mapping[tuple, str]
    ):
        pending_uploads = []
        entities = corpus_ledger.claim()
        hf_fs = HfFileSystem(token=env_config.settings.hf_token)
        for (lang, split), source_files in corpus_content.items():
            source_file_set = set(source_files)
            selected_entities = [x for x in entities if x["filename"] in source_file_set]
            for entity in selected_entities:
                if entity["status"] == "completed" and self._resume_completed_shard(
                    entity, corpus_ledger, accx, output_dir, storage
                ):
                    continue
                ray_ds = ray.data.read_parquet(entity["filename"], filesystem=hf_fs)
                rename_cols = {
                    source: target
                    for source, target in (ds_config.columns or {}).items()
                    if source != target and source in ray_ds.schema().names and target not in ray_ds.schema().names
                }
                if rename_cols:
                    ray_ds = ray_ds.rename_columns(rename_cols)
                if ds_config.exclude_cols:
                    drop_cols = [c for c in ds_config.exclude_cols if c in ray_ds.schema().names]
                    if drop_cols:
                        ray_ds = ray_ds.drop_columns(drop_cols)
                new_lang = ds_config.languages.get(lang) or lang
                new_split = ds_config.splits.get(split) or split
                name = (ds_config.corpus_name or ds_config.repo_id.split("/")[-1]).lower()
                shard_dir = (Path(output_dir) / "staging" / utils.shard_id(entity["filename"])).resolve()
                if shard_dir.exists():
                    shutil.rmtree(shard_dir)
                u_words = (ds_config.exclude_annotations or {}).get(new_lang)
                self._process_batches(ray_ds, new_lang, new_split, name, shard_dir, unuttered_words=u_words, normalizer=ds_config.normalize)
                shard_files = sorted(shard_dir.rglob("*.parquet")) if shard_dir.is_dir() else []
                print(
                    "[shard-write-check] "
                    f"cwd={Path.cwd()} shard_dir={shard_dir} "
                    f"directory_exists={shard_dir.is_dir()} parquet_count={len(shard_files)}"
                )
                for shard_file in shard_files:
                    try:
                        print(
                            f"[shard-write-check] file={shard_file.resolve()} "
                            f"is_file={shard_file.is_file()} bytes={shard_file.stat().st_size}"
                        )
                    except OSError as exc:
                        print(f"[shard-write-check] file={shard_file} stat_error={exc!r}")

                out_files = _unique_parquet_names(shard_files, utils.shard_id(entity["filename"]))
                if not out_files:
                    raise RuntimeError(f"No Parquet files produced for shard {entity['filename']}")
                if ds_config.compile_text is True or split in (ds_config.compile_text or []):
                    manifest = _compile_manifest(out_files, output_dir, name, new_split, _language_partition(new_lang))
                    failed = storage.upload([manifest], local_dir=output_dir)
                    if failed:
                        raise ValueError(f"Failed to upload transcript manifest {failed}")
                accx.update_shard_from_local(
                    out_files, name, new_split, _language_partition(new_lang),
                    identity=entity["filename"],
                )
                corpus_ledger.mark_completed(entity["filename"], out_files)
                pending_uploads.append((entity["filename"], out_files, shard_dir))
                if sum(len(pending_files) for _, pending_files, _ in pending_uploads) >= 25:
                    self._upload_completed_shard_batch(pending_uploads, corpus_ledger, accx, output_dir, storage)
                    pending_uploads.clear()
        
        if pending_uploads:
            self._upload_completed_shard_batch(pending_uploads, corpus_ledger, accx, output_dir, storage)

    def _upload_completed_shard_batch(self, pending, corpus_ledger, accx, output_dir: str, storage: manager.HFDatasetRepo) -> None:
        """Commit queued shard Parquets in groups of five, with ledger in the final commit."""
        parquet_entries = []
        for _, files, shard_dir in pending:
            for file in files:
                relative_path = Path(file).resolve().relative_to(Path(shard_dir).resolve())
                parquet_entries.append((Path(file), (Path("data") / relative_path).as_posix()))

        batches = [parquet_entries[index : index + 5] for index in range(0, len(parquet_entries), 5)]
        ledger_path = Path(corpus_ledger.repo.path).resolve()
        ledger_repo_path = ledger_path.relative_to(Path(output_dir).resolve()).as_posix()
        metrics_path = Path(accx.state_path).resolve()
        metrics_repo_path = metrics_path.relative_to(Path(output_dir).resolve()).as_posix()
        if batches:
            batches[-1].extend(
                [(ledger_path, ledger_repo_path), (metrics_path, metrics_repo_path)]
            )
        else:
            batches = [[(ledger_path, ledger_repo_path), (metrics_path, metrics_repo_path)]]

        missing_files = [
            str(Path(file).resolve())
            for batch in batches
            for file, _ in batch
            if not Path(file).is_file()
        ]
        if missing_files:
            staging_dir = Path(output_dir) / "staging"
            staging_entries = list(staging_dir.glob("*")) if staging_dir.is_dir() else []
            print(
                "[pre-upload-check] "
                f"cwd={Path.cwd()} output_dir={Path(output_dir).resolve()} "
                f"staging_exists={staging_dir.is_dir()} "
                f"staging_entry_count={len(staging_entries)}"
            )
            for file in missing_files:
                missing_path = Path(file)
                parent = missing_path.parent
                try:
                    entries = sorted(p.name for p in parent.iterdir()) if parent.is_dir() else []
                    print(
                        f"[pre-upload-check] missing={missing_path} "
                        f"parent_exists={parent.is_dir()} parent_entries={entries[:20]}"
                    )
                except OSError as exc:
                    print(f"[pre-upload-check] missing={missing_path} parent_stat_error={exc!r}")
            raise FileNotFoundError(
                "Cannot upload shard batch because these local files are missing "
                "or are not regular files:\n- "
                + "\n- ".join(missing_files)
            )

        for filename, files, _ in pending:
            corpus_ledger.mark_uploaded(filename)

        try:
            for batch in batches:
                failed = storage.upload_entries(batch, group=len(batch))
                if failed:
                    raise ValueError(f"Failed to upload {failed}")
        except Exception:
            for filename, files, _ in pending:
                corpus_ledger.mark_completed(filename, files)
            raise

        for _, _, shard_dir in pending:
            shutil.rmtree(shard_dir, ignore_errors=True)


def build_dataset_readme(ds_configs: config.Corpora) -> str:
    """Create a datasets-compatible README config map from the configured partitions."""
    from collections import defaultdict
    configs = defaultdict(lambda: defaultdict(set))
    for _, corpus in ds_configs.corpora.items():
        corpus_name = (corpus.corpus_name or corpus.repo_id.split("/")[-1]).lower()
        for source_language, target_language in corpus.languages.items():
            language = target_language or source_language
            short_language = language.removesuffix("_ng")
            config_name = f"{corpus_name.lower()}_{short_language}"
            for source_split, target_split in corpus.splits.items():
                split = target_split or source_split
                configs[config_name][split].add((corpus_name, short_language))
    lines = ["---", "license: cc-by-nc-4.0", "task_categories:", "- automatic-speech-recognition", "language:", "- yo", "- ig", "- ha", "configs:"]
    for name, split_map in sorted(configs.items()):
        lines += [f"- config_name: {name}", "  data_files:"]
        for split in sorted(split_map):
            lines += [f"  - split: {split}", f"    path: data/corpus={next(iter(split_map[split]))[0]}/split={split}/language={next(iter(split_map[split]))[1]}/*.parquet"]
    lines += ["---", "", "# Wazobia ASR Dataset", "", "Speech datasets processed into a common schema. Configurations are generated from the ingestion registry.", ""]
    return "\n".join(lines)
    

# annotations


## note a corpus was edited because the annotation wasn't removed
# _ingest_corpus_internal, _self._persist_metrics would be a method
# batch_processor: type: base | naijavoices
# 
