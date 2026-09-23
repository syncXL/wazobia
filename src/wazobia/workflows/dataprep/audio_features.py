import io
import numpy as np
import pyarrow as pa
import  soundfile as sf
import librosa


class AudioFeatureProcessor:
    def __init__(
        self,
        audio_column: str = "audio",
        frame_ms: int = 25,
        hop_ms: int = 10,
        noise_percentile: float = 10.0,
        target_sr: int = 16000,
    ):
        self.audio_column = audio_column
        self.frame_ms = frame_ms
        self.hop_ms = hop_ms
        self.noise_percentile = noise_percentile
        self.target_sr = target_sr
        self._err_count = 0

    @staticmethod
    def _db(rms: float) -> float:
        return float(20.0 * np.log10(max(rms, 1e-10)))

    def _extract_one(self, audio):
        orig = audio.get("bytes") if isinstance(audio, dict) else audio
        if orig is None:
            return self._empty_features()

        try:
            samples, sr = sf.read(io.BytesIO(orig), dtype="float32")
            if samples.ndim != 1:
                raise ValueError(f"expected mono, got shape {samples.shape}")

            resampled_bytes = None
            if sr != self.target_sr:
                samples = librosa.resample(samples, orig_sr=sr, target_sr=self.target_sr)
                sr = self.target_sr
                buf = io.BytesIO()
                sf.write(buf, samples, sr, format="WAV", subtype="PCM_16")
                resampled_bytes = buf.getvalue()

            n = len(samples)
            duration_sec = n / sr

            frame_len = max(1, int(sr * self.frame_ms / 1000))
            hop_len = max(1, int(sr * self.hop_ms / 1000))
            if n < frame_len:
                samples = np.pad(samples, (0, frame_len - n))

            frames = np.lib.stride_tricks.sliding_window_view(samples, frame_len)[::hop_len]
            rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)

            noise_thr = np.percentile(rms, self.noise_percentile)
            noise_rms = float(np.sqrt(np.mean(rms[rms <= noise_thr] ** 2)))

            speech_mask = rms > noise_rms * 10 ** (6 / 20)
            if not speech_mask.any():
                speech_mask = np.ones_like(rms, dtype=bool)
            speech_rms = float(np.sqrt(np.mean(rms[speech_mask] ** 2)))

            noise_level_db = self._db(noise_rms)
            speech_level_db = self._db(speech_rms)
            snr_db = speech_level_db - noise_level_db
            speech_ratio = float(speech_mask.mean())

            idx = np.linspace(0, len(frames) - 1, min(len(frames), 300)).astype(int)
            window = np.hanning(frame_len).astype(np.float32)
            mag = np.abs(np.fft.rfft(frames[idx] * window, axis=1))
            freqs = np.fft.rfftfreq(frame_len, 1 / sr)
            s = mag.sum(axis=1)
            ok = s > 0
            if ok.any():
                centroid = (mag[ok] * freqs).sum(axis=1) / s[ok]
                bw = np.sqrt((mag[ok] * (freqs - centroid[:, None]) ** 2).sum(axis=1) / s[ok])
                spectral_bandwidth_hz = float(bw.mean())
            else:
                spectral_bandwidth_hz = 0.0

            return {
                "snr_db": snr_db,
                "noise_level_db": noise_level_db,
                "speech_level_db": speech_level_db,
                "speech_ratio": speech_ratio,
                "duration_sec": duration_sec,
                "spectral_bandwidth_hz": spectral_bandwidth_hz,
                "resampled_bytes": resampled_bytes,   # None if already 16kHz
            }
        except Exception as e:
            if self._err_count < 5:
                print(f"[AudioFeatureProcessor] {type(e).__name__}: {e}")
            self._err_count += 1
            r = self._empty_features()
            r["resampled_bytes"] = None
            return r

    @staticmethod
    def _empty_features():
        return {
            "snr_db": None,
            "noise_level_db": None,
            "speech_level_db": None,
            "speech_ratio": None,
            "duration_sec": None,
            "spectral_bandwidth_hz": None,
        }

    def __call__(self, batch: pa.Table) -> pa.Table:
        audio_col = batch[self.audio_column].to_pylist()
        features = [self._extract_one(item) for item in audio_col]

        feature_types = {
            "snr_db": pa.float32(),
            "noise_level_db": pa.float32(),
            "speech_level_db": pa.float32(),
            "speech_ratio": pa.float32(),
            "duration_sec": pa.float32(),
            "spectral_bandwidth_hz": pa.float32(),
        }

        if "duration_sec" in batch.column_names:
            batch = batch.drop(["duration_sec"])

        for name, dtype in feature_types.items():
            values = [item[name] for item in features]
            batch = batch.append_column(name, pa.array(values, type=dtype))

        # overwrite audio bytes only where resampling actually happened
        new_audio = [
            {"bytes": f["resampled_bytes"], "path": orig.get("path") if isinstance(orig, dict) else None}
            if f["resampled_bytes"] is not None
            else orig
            for f, orig in zip(features, audio_col)
        ]
        col_idx = batch.column_names.index(self.audio_column)
        batch = batch.set_column(col_idx, self.audio_column, pa.array(new_audio, type=batch.schema.field(self.audio_column).type))

        return batch