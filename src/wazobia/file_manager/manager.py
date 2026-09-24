import os
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import time
from pathlib import Path
from huggingface_hub import HfFileSystem, batch_bucket_files


class HFBucket:
    def __init__(self, bucket_link: str, token: str | None = None) -> None:
        # accepts either "nolimitsxl/wazobia" or "hf://buckets/nolimitsxl/wazobia"
        self.bucket = bucket_link.removeprefix("hf://buckets/").rstrip("/")
        self.fs = HfFileSystem(token=token)

    def upload(
        self,files: list[Path],
        local_dir: str,
        remote_prefix: str = "",
        group: int = 8,
        tries: int = 5,
    ) -> list[Path]:
        root = Path(local_dir)
        if not files:
            return []

        failed: list[Path] = []
        for i in range(0, len(files), group):
            chunk = files[i:i + group]
            add: list[tuple[str | Path | bytes, str]] = [
                (str(f), f"{remote_prefix}/{f.relative_to(root).as_posix()}".lstrip("/"))
                for f in chunk
            ]
            print(add)
            for attempt in range(tries):
                try:
                    batch_bucket_files(self.bucket, add=add)
                    break
                except Exception as e:
                    if attempt == tries - 1:
                        print(f"[HFBucket.upload FAILED] {[str(f) for f in chunk]}: {e}")
                        failed.extend(chunk)
                    else:
                        time.sleep(2 ** attempt * 5)
        return failed

    def download(self, local_dir: str, file_name: str, tries: int = 5):
        local_path = Path(local_dir)
        remote_path = f"hf://buckets/{self.bucket}/{file_name.strip('/')}"

        self.fs.invalidate_cache()
        if not self.fs.exists(remote_path):
            return None

        for attempt in range(tries):
            try:
                self.fs.invalidate_cache()  # stale listings after recent uploads
                is_dir = self.fs.isdir(remote_path)
                if is_dir:
                    local_path.mkdir(parents=True, exist_ok=True)
                    self.fs.get(remote_path, str(local_path) + "/", recursive=True)
                else:
                    local_path.parent.mkdir(parents=True, exist_ok=True)
                    self.fs.get_file(remote_path, str(local_path))  # bypasses the dir-skip logic

                ok = any(local_path.rglob("*")) if is_dir else local_path.is_file()
                if not ok:
                    raise FileNotFoundError(f"Nothing downloaded from {remote_path}")
                return local_path
            except Exception as e:
                print(f"[download attempt {attempt}] {e}")
                if attempt == tries - 1:
                    raise
                time.sleep(2 ** attempt * 5)

# def list_f
