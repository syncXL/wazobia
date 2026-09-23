import pyarrow as pa

def map_to_target_schema(
    batch: pa.Table,
    split: str,
    corpus: str,
    audio_col: str
) -> pa.Table:
    """
    Maps a processed batch to the unified target schema.
    """

    batch = batch.rename_columns({"transcription": "text"})

    batch = batch.append_column(
        "split",
        pa.array([split] * len(batch), type=pa.string()),
    )

    batch = batch.append_column(
        "corpus",
        pa.array([corpus] * len(batch), type=pa.string()),
    )

    return batch.select([
        "text",
        audio_col,
        "language",
        "split",
        "corpus",

        "snr_db",
        "noise_level_db",
        "speech_level_db",
        "speech_ratio",
        "duration_sec",
        "spectral_bandwidth_hz",
    ])
