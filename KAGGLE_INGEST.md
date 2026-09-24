# Kaggle ingestion runbook

The ingest worker is CPU bound. Run it on Kaggle's free CPU session first; enabling a GPU does not accelerate the current audio feature pipeline. Keep the repo on a Python 3.12 environment with the locked project dependencies available.

## Notebook setup

Enable Internet for the notebook and add Kaggle Secrets named `HF_TOKEN` and `HF_BUCKET`. Set environment variables before importing Hugging Face or Ray packages:

```python
import os
from kaggle_secrets import UserSecretsClient

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
secrets = UserSecretsClient()
os.environ["HF_TOKEN"] = secrets.get_secret("HF_TOKEN")
os.environ["HF_BUCKET"] = secrets.get_secret("HF_BUCKET")
```

From the repository root, install only the ingestion dependencies into Kaggle's active Python environment. This avoids creating a second environment and skips Torch, Transformers, and TorchCodec, which the ingestion code does not import:

```python
%pip install --no-cache-dir \
  "ray[data]>=2.58.0" \
  "datasets>=5.0.1" \
  "librosa>=1.0.0" \
  "polars>=1.44.2" \
  "fire>=0.7.1" \
  "pydantic-settings>=2.15.0" \
  "soundfile>=0.14.0" \
  "PyYAML" \
  "unidecode>=1.4.0" \
  "huggingface_hub"
```

Restart the notebook session if Kaggle asks for it. Then run from the repository root, using the notebook's active Python executable and the source tree directly:

```python
import os
import subprocess
import sys

repo = "/kaggle/working/wazobia"
env = os.environ.copy()
env["PYTHONPATH"] = os.path.join(repo, "src") + os.pathsep + env.get("PYTHONPATH", "")
subprocess.run(
    [sys.executable, "-m", "wazobia.workflows.dataprep.ingest", "run_set",
     "--output_dir=/kaggle/working/wazobia-output", "--load_from_hf=True"],
    cwd=repo,
    env=env,
    check=True,
)
```

For a single-corpus run, replace `run_set` with `ingest_yfacc` (or another public `ingest_*` method) and remove `--load_from_hf=True`.

The worker uses `/kaggle/working/ray` for Ray temporary files and excludes `.venv`, `data`, `notebooks`, and `.env` from its Ray working-directory upload. The driver sets `HF_HUB_DISABLE_XET=1` for Ray workers as well.

Batch transforms default to one Ray worker and batches of 16; streamed shuffling defaults to a 256-example buffer. These limits fit small-memory sessions. Set `WAZOBIA_RAY_WORKERS`, `WAZOBIA_RAY_BATCH_SIZE`, or `WAZOBIA_SHUFFLE_BUFFER_SIZE` before launch only when the session has enough memory to support higher values.

## Resume and outputs

Parquet staging is isolated by source shard and is removed after a successful upload. The ledger and metrics checkpoint are uploaded to the configured HF Bucket, so a fresh Kaggle session can restore them with `--load_from_hf=True`. The metrics TSV is written and uploaded when the run finishes; the pickle checkpoint is uploaded after each processed shard.

Kaggle's working directory is temporary across sessions. Treat the HF Bucket as the durable copy, and rerun the command with `--load_from_hf=True` after a session interruption. Do not run multiple ingestion processes against the same bucket ledger and metrics checkpoint at once.

If Kaggle's free CPU session limits prevent completing the corpus set, continue with the same durable bucket state on RunPod Community Cloud, then Vast.ai spot. Keep the local RTX 3050 machine for inference only.
