import os
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import shutil
import time
from pathlib import Path, PurePosixPath

from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download


class HFDatasetRepo:
    """Upload and restore ingestion artifacts from a Hugging Face dataset repo."""

    def __init__(self, repo_id: str, token: str | None = None) -> None:
        # Accept a plain repo id or common Hugging Face repository URLs.
        self.repo_id = (
            repo_id.removeprefix("https://huggingface.co/datasets/")
            .removeprefix("hf://datasets/")
            .removeprefix("hf://buckets/")
            .rstrip("/")
        )
        self.token = token
        self.api = HfApi(token=token)

    def upload(
        self,
        files: list[Path],
        local_dir: str,
        remote_prefix: str = "",
        group: int = 1,
        tries: int = 5,
    ) -> list[Path]:
        """Upload files to a dataset repo, placing Parquet shards under ``data/``.

        Other files keep their path relative to ``local_dir``. This keeps ledger
        and metrics at the repository root while consolidating all corpus data.
        """
        if group < 1:
            raise ValueError("group must be at least 1")
        if tries < 1:
            raise ValueError("tries must be at least 1")
        root = Path(local_dir).resolve()
        upload_entries: list[tuple[Path, str]] = []

        for file in files:
            file = Path(file).resolve()
            try:
                relative_path = PurePosixPath(file.relative_to(root).as_posix())
            except ValueError as exc:
                raise ValueError(f"{file} is outside upload root {root}") from exc

            if file.suffix == ".parquet" and relative_path.parts[0] != "data":
                relative_path = PurePosixPath("data") / relative_path
            if remote_prefix:
                relative_path = PurePosixPath(remote_prefix) / relative_path

            path_in_repo = relative_path.as_posix()
            upload_entries.append((file, path_in_repo))

        return self.upload_entries(upload_entries, group=group, tries=tries)

    def upload_entries(
        self,
        entries: list[tuple[Path, str]],
        group: int = 1,
        tries: int = 5,
    ) -> list[Path]:
        """Upload local files to explicit repository paths in grouped commits."""
        if group < 1:
            raise ValueError("group must be at least 1")
        if tries < 1:
            raise ValueError("tries must be at least 1")
        failed: list[Path] = []

        for start in range(0, len(entries), group):
            batch = entries[start : start + group]
            for attempt in range(tries):
                try:
                    self.api.create_commit(
                        repo_id=self.repo_id,
                        repo_type="dataset",
                        operations=[
                            CommitOperationAdd(path_in_repo, str(file))
                            for file, path_in_repo in batch
                        ],
                        commit_message=f"Upload {len(batch)} file(s)",
                        token=self.token,
                    )
                    break
                except Exception as exc:
                    if attempt == tries - 1:
                        print(
                            "[HFDatasetRepo.upload FAILED] "
                            f"{[str(file) for file, _ in batch]}: {exc}"
                        )
                        failed.extend(file for file, _ in batch)
                    else:
                        time.sleep(2 ** attempt * 5)
        return failed

    def download(self, local_dir: str, file_name: str, tries: int = 5):
        """Download a repo file or directory into the requested local path."""
        local_path = Path(local_dir)
        remote_name = PurePosixPath(file_name.strip("/")).as_posix()

        for attempt in range(tries):
            try:
                repo_files = self.api.list_repo_files(
                    repo_id=self.repo_id,
                    repo_type="dataset",
                    token=self.token,
                )
                if remote_name in repo_files:
                    cached_file = Path(
                        hf_hub_download(
                            repo_id=self.repo_id,
                            filename=remote_name,
                            repo_type="dataset",
                            token=self.token,
                        )
                    )
                    local_path.parent.mkdir(parents=True, exist_ok=True)
                    if cached_file.resolve() != local_path.resolve():
                        shutil.copy2(cached_file, local_path)
                    return local_path

                matching_files = [
                    name for name in repo_files if name.startswith(remote_name + "/")
                ]
                if not matching_files:
                    return None

                for name in matching_files:
                    cached_file = Path(
                        hf_hub_download(
                            repo_id=self.repo_id,
                            filename=name,
                            repo_type="dataset",
                            token=self.token,
                        )
                    )
                    relative_name = PurePosixPath(name).relative_to(remote_name)
                    destination = local_path.joinpath(*relative_name.parts)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    if cached_file.resolve() != destination.resolve():
                        shutil.copy2(cached_file, destination)
                return local_path
            except Exception as exc:
                print(f"[download attempt {attempt}] {exc}")
                if attempt == tries - 1:
                    raise
                time.sleep(2 ** attempt * 5)
