import pyarrow as pa

TARGET_SCHEMA = pa.schema([
    pa.field("text", pa.string()),
    pa.field("audio", pa.struct([
        pa.field("bytes", pa.binary()),
        pa.field("path", pa.string()),
    ])),
    pa.field("language", pa.string()),
    pa.field("split", pa.string()),
    pa.field("corpus", pa.string()),
    pa.field("snr_db", pa.float32()),
    pa.field("noise_level_db", pa.float32()),
    pa.field("speech_level_db", pa.float32()),
    pa.field("speech_ratio", pa.float32()),
    pa.field("duration_sec", pa.float32()),
    pa.field("spectral_bandwidth_hz", pa.float32()),
])

def map_to_target_schema(
    batch: pa.Table,
    split: str,
    corpus: str,
    audio_col: str
) -> pa.Table:
    """
    Maps a processed batch to the unified target schema.
    """

    text_column = "transcription" if "transcription" in batch.column_names else "transcript" if "transcript" in batch.column_names else "text"
    if text_column != "text":
        batch = batch.rename_columns({text_column: "text"})
    audio_column = audio_col
    if audio_column != "audio":
        batch = batch.rename_columns({audio_column: "audio"})

    for name in ("split", "corpus"):
        if name in batch.column_names:
            batch = batch.drop([name])

    batch = batch.append_column(
        "split",
        pa.array([split] * len(batch), type=pa.string()),
    )

    batch = batch.append_column(
        "corpus",
        pa.array([corpus] * len(batch), type=pa.string()),
    )

    batch = batch.select(TARGET_SCHEMA.names)
    return batch.cast(TARGET_SCHEMA, safe=True)
