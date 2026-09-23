import os
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import datasets
import fire
import hashlib
import polars as pl
import pyarrow as pa
import os
import ray
import sys

from datasets import load_dataset, Audio, Dataset
from huggingface_hub import HfApi, hf_hub_download
from functools import partial
from math import floor
from pathlib import Path
from typing import Sequence
from wazobia import config
from wazobia.workflows.dataprep import text_tools_corpus, audio_features, audio_tools, metrics
from wazobia.ledger import ledger
from wazobia.file_manager import manager


def remove_files(files: Sequence[str | Path]) -> None:
    """Remove the given files, ignoring files that do not exist."""
    for file in files:
        Path(file).unlink()

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
        self.bucket = manager.HFBucket(settings.hf_bucket, settings.hf_token)

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
        ray_ds_stream_ = ray.data.from_huggingface(ds)
        
        num_cpus = max(floor((os.cpu_count() or 1) / 4), 1)
        ray_ds_stream_ = ray_ds_stream_.map_batches(
            text_tools_corpus.TextProcessor,
            fn_constructor_kwargs={"lang" : lang, "remove_numbers" : remove_numbers},
            batch_size=100,
            batch_format="pyarrow",
            concurrency=num_cpus
        )

        ray_ds_stream_ = ray_ds_stream_.map_batches(
            audio_features.AudioFeatureProcessor,
            fn_constructor_kwargs={
                "audio_column" : audio_column,
            },
            batch_size=100,
            batch_format="pyarrow",
            concurrency=num_cpus
        )

        ray_ds_stream_ = ray_ds_stream_.map_batches(
            partial(
                audio_tools.map_to_target_schema,
                split=split,
                corpus=corpus,
                audio_col=audio_column
            ),
            batch_size=100,
            batch_format="pyarrow"
        )

        ray_ds_stream_.write_parquet(
            output_dir,
            partition_cols=["corpus", "split", "language"],
            min_rows_per_file=10_000,
            row_group_size=100
        )

        return f"{output_dir}/corpus={corpus}/split={split}/language={lang.rstrip("_ng")}"

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
        split_map : dict[str, str] = dict(),
        lang_map : dict[str,str] = dict(),
        remove_numbers : bool = False
    ):
        for entity in corpus_ledger.claim():
            split = entity["filename"].split("/")[-1]
            if not add_lang_config:
                corpus_hf = load_dataset(repo_id, split=split, streaming=True)
            else:
                corpus_hf = load_dataset(repo_id, lang,split=split, streaming=True)

            corpus_hf = corpus_hf.shuffle(seed=42, buffer_size=10_000)
            corpus_hf = corpus_hf.cast_column("audio", Audio(decode=False, sampling_rate=16000))
            corpus_hf = corpus_hf.rename_column({text_col : "transcript"})
            
            
            corpus_lang = lang or ""
            hive_dir = self._process_batches(corpus_hf, lang_map.get(corpus_lang, corpus_lang), split_map.get(split, split), corpus_name, output_dir + "/data", remove_numbers=remove_numbers)
            files = list(Path(hive_dir).glob("*.parquet"))

            for file in files:
                accx.update_from_local(str(file), corpus_name, split_map.get(split,split), lang_map.get(lang,lang))

            files.append(corpus_ledger.repo.path)
            corpus_ledger.mark_completed(entity["filename"], files)
            failed = self.bucket.upload(files,local_dir=output_dir)

            if len(failed) != 0:
                raise ValueError(f"Failed to upload {failed}")

            corpus_ledger.mark_uploaded(entity["filename"])
            remove_files(files)

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
        for folder in folders:
            for file in self.HFAPI.list_repo_tree(repo_id, path_in_repo=folder, repo_type="dataset"):
                fpath = file.path
                sp = assign_split(fpath)
                corpus[sp].append(fpath)
                corpus_ledger.register_file(f"{sp}^{fpath}" )

        for entity in corpus_ledger.claim():
            split, fp = entity["filename"].split("^")
            lang = fp.split("-")[0]
            lang = lang_map.get(lang) or lang
            download_path = hf_hub_download(
                repo_id=repo_id,
                filename=fp,
                repo_type="dataset"
            )
            fp_hf = Dataset.from_parquet(download_path)
            fp_hf = fp_hf.rename_columns({"text" : "transcript"})
            fp_hf = fp_hf.cast_column("audio", Audio(decode=False, sampling_rate=16_000))
            num_cpus = max(floor((os.cpu_count() or 1) / 4), 1)

            ray_ds = ray.data.from_huggingface(fp_hf)            
            ray_ds = ray_ds.map_batches(
                text_tools_corpus.TextProcessor,
                fn_constructor_kwargs={"lang" : lang, "remove_numbers" : False},
                batch_size=100,
                batch_format="pyarrow",
                concurrency=num_cpus
            )
            
            ray_ds = ray_ds.map_batches(
                audio_features.AudioFeatureProcessor,
                fn_constructor_kwargs={
                    "audio_column" : "audio",
                },
                batch_size=100,
                batch_format="pyarrow",
                concurrency=num_cpus
            )
            
            ray_ds = ray_ds.map_batches(
                partial(
                    audio_tools.map_to_target_schema,
                    split=split,
                    corpus="naijavoices",
                    audio_col="audio"
                ),
                batch_size=100,
                batch_format="pyarrow"
            )
    
            ray_ds.write_parquet(
                output_dir + "/data",
                partition_cols=["corpus", "split", "language"],
                min_rows_per_file=10_000,
                row_group_size=100
            )

            output_path = f"{output_dir}/data/corpus=naijavoices/split={split}/language={lang.rstrip("_ng")}"
            files = list(Path(output_path).glob("*.parquet"))
            for file in files:
                accx.update_from_local(str(file),"naijavoices",split, lang_map.get(lang,lang))
            files.append(corpus_ledger.repo.path)
            corpus_ledger.mark_completed(entity["filename"], files)
            failed = self.bucket.upload(files,local_dir=output_dir)
            if len(failed) != 0:
                raise ValueError(f"Failed to upload {failed}")

            corpus_ledger.mark_uploaded(entity["filename"])
            remove_files(files)

    def _ingest_yfacc_internal(
        self, output_dir: str,accx=metrics.MetricsAccumulator, lang_subset: list[str] | None = None
    ):
        # see https://huggingface.co/datasets/nolimitsxl/yfacc_yoruba
        repo_id = "nolimitsxl/yfacc_yoruba"
        splits = ["test", "val", "train"]
        corpus_ledger = ledger.CorpusLedger(output_dir + "/ledger", repo_id=repo_id, repo_type="dataset")
        corpus_ledger.register_files([f"{repo_id}/{split}" for split in splits])
        self._ingest_corpus_internal(output_dir, "YFACC", repo_id, False, corpus_ledger,"transcript", accx, "yor_ng")
        
    def _ingest_yecs_internal(
            self, output_dir: str,accx=metrics.MetricsAccumulator, lang_subset: list[str] | None = None
        ):
            # see https://huggingface.co/datasets/nolimitsxl/yecs_lyngual_labs
            repo_id = "nolimitsxl/yecs_lyngual_labs"
            splits = ["val", "train"]
            corpus_ledger = ledger.CorpusLedger(output_dir + "/ledger", repo_id=repo_id, repo_type="dataset")
            corpus_ledger.register_files([f"{repo_id}/{split}" for split in splits])
            self._ingest_corpus_internal(output_dir, "YECS_LYNGUAL_LABS", repo_id, False, corpus_ledger, "transcript", accx, "yor_ng")

    def _ingest_igbo_sync(
            self, output_dir: str,accx=metrics.MetricsAccumulator, lang_subset: list[str] | None = None
        ):
        # see https://huggingface.co/datasets/nolimitsxl/igbo_sync_processed
        repo_id = "nolimitsxl/igbo_sync_processed"
        splits = ["val", "train"]
        corpus_ledger = ledger.CorpusLedger(output_dir + "/ledger", repo_id=repo_id, repo_type="dataset")
        corpus_ledger.register_files([f"{repo_id}/{split}" for split in splits])
        self._ingest_corpus_internal(output_dir, "Igbo_sync", repo_id, False, corpus_ledger, "transcript", accx,"ibo_ng")

    def _ingest_naed_internal(
            self, output_dir: str,accx=metrics.MetricsAccumulator, lang_subset: list[str] | None = None
        ):
        # see https://huggingface.co/datasets/benjaminogbonna/nigerian_accented_english_dataset
        repo_id = "benjaminogbonna/nigerian_accented_english_dataset"
        splits = ["test","validation", "train"]
        corpus_ledger = ledger.CorpusLedger(output_dir + "/ledger", repo_id=repo_id, repo_type="dataset")
        corpus_ledger.register_files([f"{repo_id}/{split}" for split in splits])
        split_remap = {"validation" : "val"}
        self._ingest_corpus_internal(output_dir, "nigerian_accented_english_dataset", repo_id, False, corpus_ledger, "sentence", accx, "eng_ng", split_map=split_remap)

    def _ingest_ud_naija_nsc_internal(self, output_dir,accx=metrics.MetricsAccumulator, lang_subset:list[str] | None = None):
        # see https://huggingface.co/datasets/timniel/Pidgin_ASR_Dataset_Combined
        repo_id = "timniel/Pidgin_ASR_Dataset_Combined"
        splits = ["train"]
        corpus_ledger = ledger.CorpusLedger(output_dir + "/ledger", repo_id=repo_id, repo_type="dataset")
        corpus_ledger.register_files([f"{repo_id}/{split}" for split in splits])
        self._ingest_corpus_internal(output_dir, "UD NAIJA NSC", repo_id, False, corpus_ledger, "text", accx,"pcm_ng")

    def _ingest_asr_nigerian_pidgin_internal(
            self, output_dir: str,accx=metrics.MetricsAccumulator,, lang_subset: list[str] | None = None
            ):
        # see https://huggingface.co/datasets/asr-nigerian-pidgin/nigerian-pidgin-1.0
        repo_id = "asr-nigerian-pidgin/nigerian-pidgin-1.0"
        splits = ["train", "validation", "test"]
        corpus_ledger = ledger.CorpusLedger(output_dir + "/ledger", repo_id=repo_id, repo_type="dataset")
        corpus_ledger.register_files([f"{repo_id}/{split}" for split in splits])
        split_remap = {"validation" : "val"}
        self._ingest_corpus_internal(output_dir, "nigerian-pidgin-1.0", repo_id, False, corpus_ledger, "sentence", accx,"pcm_ng",split_map=split_remap)

    def _ingest_open_slr_internal(self, output_dir: str, accx=metrics.MetricsAccumulator,lang_subset: list[str] | None = None):
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
            self._ingest_corpus_internal(output_dir, "Open SLR", repo_id, True, corpus_ledger, "transcript", accx,lang)

    def _ingest_twb_internal(self, output_dir: str, accx=metrics.MetricsAccumulator,lang_subset: list[str] | None = None):
        repo_id = "CLEAR-Global/TWB-Voice-1.0"
        splits = ["train", "dev", "test"]
        split_remap = {"dev" : "val"}
        corpus_ledger = ledger.CorpusLedger(output_dir + "/ledger", repo_id=repo_id, repo_type="dataset")
        if not lang_subset:
            lang_subset = self.TWB
        for lang in lang_subset:
            if lang not in self.TWB:
                print(f"{lang} does not exist. Skipping...")
                continue
            corpus_ledger.register_files([f"{repo_id}/{lang}/{split}" for split in splits])
            self._ingest_corpus_internal(output_dir, "ClearVoice/TWB", repo_id, True, corpus_ledger, "sentence", accx,lang, split_map=split_remap)
        
    def _ingest_aspv1_internal(self, output_dir: str, accx=metrics.MetricsAccumulator,lang_subset: list[str] | None = None):
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
            self._ingest_corpus_internal(output_dir, "Afrispeech/ASP", repo_id, True, corpus_ledger, "text", accx,lang, split_map=split_remap, lang_map=lang_remap, remove_numbers=True)
    
    def _ingest_obsa_internal(self, output_dir: str, accx=metrics.MetricsAccumulator,lang_subset: list[str] | None = None):
        repo_id = "AfriSpeech/open-bible-speech-african"
        lang_remap = {
            "Hausa" : "hau_ng",
            "Yoruba" : "yor_ng",
            "Igbo" : "ibo_ng"
        }
        splits = ["train","test"]
        corpus_ledger = ledger.CorpusLedger(output_dir + "/ledger", repo_id=repo_id, repo_type="dataset")
        if not lang_subset:
            lang_subset = self.OBSA
        for lang in lang_subset:
            if lang not in self.OBSA:
                print(f"{lang} does not exist. Skipping...")
                continue
            corpus_ledger.register_files([f"{repo_id}/{lang}/{split}" for split in splits])
            self._ingest_corpus_internal(output_dir, "AfriSpeech/open-bible-speech-african", repo_id, True, corpus_ledger, "text", accx,lang, lang_map=lang_remap, remove_numbers=True)
    
    def _ingest_yas_internal(self, output_dir: str, accx=metrics.MetricsAccumulator,lang_subset: list[str] | None = None):
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
            self._ingest_corpus_internal(output_dir, "YouVersion African Speech", repo_id, True, corpus_ledger, "text", accx,lang, lang_map=lang_remap, remove_numbers=True)
    
    def _ingest_fleurs_internal(self, output_dir: str, accx=metrics.MetricsAccumulator,lang_subset: list[str] | None = None):
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
    
    def ingest_yfacc(self, output_dir: str, accx: metrics.MetricsAccumulator):
        """Ingest YFACC datasets.

        Args:
            output_dir: Output directory path for processed Parquet files

        """
        print(f"Starting YFACC ingestion to: {output_dir}")
        self._ingest_yfacc_internal(output_dir, accx=accx)
        print("YFACC ingestion completed")
    
    def ingest_yecs(self, output_dir: str, accx: metrics.MetricsAccumulator):
        """Ingest YECS datasets.
        
        Args:
            output_dir: Output directory path for processed Parquet files

        """
        print(f"Starting YECS ingestion to: {output_dir}")
        self._ingest_yecs_internal(output_dir, accx=accx)
        print("YECS ingestion completed")

    def ingest_igbo_sync(self, output_dir: str, accx: metrics.MetricsAccumulator):
        """Ingest Igbo Sync datasets.
        
        Args:
            output_dir: Output directory path for processed Parquet files

        """
        print(f"Starting Igbo Sync ingestion to: {output_dir}")
        self._ingest_igbo_sync(output_dir, accx=accx)
        print("Igbo Sync ingestion completed")

    def ingest_naed_sync(self, output_dir: str, accx: metrics.MetricsAccumulator):
        """Ingest NAED datasets.
        
        Args:
            output_dir: Output directory path for processed Parquet files

        """
        print(f"Starting NAED ingestion to: {output_dir}")
        self._ingest_naed_internal(output_dir, accx=accx)
        print("NAED ingestion completed")

    def ingest_asr_nigerian_pidgin(self, output_dir: str, accx: metrics.MetricsAccumulator):
        """Ingest ASR Nigerian Pidgin  datasets.
                
        Args:
            output_dir: Output directory path for processed Parquet files

        """
        print(f"Starting ASR Nigerian Pidgin ingestion to: {output_dir}")
        self._ingest_asr_nigerian_pidgin_internal(output_dir, accx=accx)
        print("ASR Nigerian Pidgin ingestion completed")

    def ingest_ud_naija_nsc(self, output_dir: str, accx: metrics.MetricsAccumulator):
        """Ingest ASR Nigerian Pidgin  datasets.
                
        Args:
            output_dir: Output directory path for processed Parquet files

        """
        print(f"Starting UD Naija NSC ingestion to: {output_dir}")
        self._ingest_ud_naija_nsc_internal(output_dir, accx=accx)
        print("UD Naija NSC ingestion completed")

    def ingest_open_slr(self, output_dir: str, accx: metrics.MetricsAccumulator):
        """Ingest Open SLR dataset.
                            
        Args:
            output_dir: Output directory path for processed Parquet files

        """
        print(f"Starting Open SLR ingestion to: {output_dir}")
        self._ingest_open_slr_internal(output_dir, accx=accx)
        print("OpenSLR ingestion completed")

    def ingest_twb(self, output_dir: str, accx: metrics.MetricsAccumulator):
        """Ingest TWB dataset.
                            
        Args:
            output_dir: Output directory path for processed Parquet files

        """
        print(f"Starting TWB ingestion to: {output_dir}")
        self._ingest_twb_internal(output_dir, accx=accx)
        print("TWB ingestion completed") 

    def ingest_aspv1(self, output_dir: str, accx: metrics.MetricsAccumulator):
        """Ingest AfriSpeech/ASPV1 dataset.
                                    
        Args:
            output_dir: Output directory path for processed Parquet files

        """
        print(f"Starting AfriSpeech ingestion to: {output_dir}")
        self._ingest_aspv1_internal(output_dir, accx=accx)
        print("ASPV1 ingestion completed") 

    def ingest_yas(self, output_dir: str, accx: metrics.MetricsAccumulator):
        """Ingest AfriSpeech/Youversion-African-Speech dataset.
                                   , accx=accx 
        Args:
            output_dir: Output directory path for processed Parquet files

        """
        print(f"Starting AfriSpeech/Youversion-African-Speech ingestion to: {output_dir}")
        self._ingest_yas_internal(output_dir, accx=accx)
        print("AfriSpeech/Youversion-African-Speech ingestion completed") 

    def ingest_obsa(self, output_dir: str, accx: metrics.MetricsAccumulator):
        """Ingest AfriSpeech/Open Bible Speech dataset.
                                    
        Args:
            output_dir: Output directory path for processed Parquet files

        """
        print(f"Starting AfriSpeech/Open Bible Speech ingestion to: {output_dir}")
        self._ingest_obsa_internal(output_dir, accx=accx)
        print("AfriSpeech/Open Bible Speech ingestion completed") 

    def ingest_fleurs(self, output_dir: str, accx: metrics.MetricsAccumulator):
        """Ingest FLEURS dataset.
                                    
        Args:
            output_dir: Output directory path for processed Parquet files

        """
        print(f"Starting FLEURS ingestion to: {output_dir}")
        self._ingest_fleurs_internal(output_dir, accx=accx)
        print("FLEURS completed")

    def ingest_naijavoices(self, output_dir: str, accx: metrics.MetricsAccumulator):
        """Ingest NaijaVoices dataset.
                                            
        Args:
            output_dir: Output directory path for processed Parquet files

        """
        print(f"Starting NaijaVoices ingestion to: {output_dir}")
        self._ingest_naijavoices_internal(output_dir,accx=accx)
        print("NaijaVoices completed")

    def run_set(self, output_dir: str, load_from_hf: bool = False):
        corpii = [self.ingest_yfacc, self.ingest_yecs, self.ingest_igbo_sync, self.ingest_naed_sync, self.ingest_asr_nigerian_pidgin, self.ingest_ud_naija_nsc, self.ingest_open_slr, self.ingest_twb, self.ingest_aspv1, self.ingest_yas, self.ingest_obsa, self.ingest_fleurs, self.ingest_naijavoices]
        if load_from_hf:
            self.bucket.download(output_dir,"ledger")
        accumulator = metrics.MetricsAccumulator(output_dir + "/ledger")

        for corpus in corpii:
            print(f"Processing  {corpus.__name__}")
            corpus(output_dir, accx=accumulator)
            print(f"{corpus.__name__} completed")
    

if __name__ == "__main__":
    if not ray.is_initialized():
        ray.init()

    ctx = ray.data.DataContext.get_current()
    ctx.enable_rich_progress_bars = True
    ctx.use_ray_tqdm = False

    try:
        fire.Fire(DataPrepCLI)
    finally:
        if ray.is_initialized():
            ray.shutdown()