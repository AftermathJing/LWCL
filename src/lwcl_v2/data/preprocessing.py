from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy import signal

from .intel5300 import read_receiver_file
from .quality import QUALITY_FIELDS, ReceiverQuality, estimate_quality, normalized_timestamps


def _zscore(values: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    mean = values.mean(axis=0, keepdims=True)
    std = values.std(axis=0, keepdims=True)
    return ((values - mean) / np.maximum(std, epsilon)).astype(np.float32)


def _complex_pca(values: np.ndarray, components: int) -> np.ndarray:
    centered = values - values.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    available = min(components, vh.shape[0])
    projected = centered @ vh[:available].conj().T
    if available < components:
        projected = np.pad(projected, ((0, 0), (0, components - available)))
    return projected.astype(np.complex64)


def _interpolate(values: np.ndarray, source_time: np.ndarray, target_time: np.ndarray) -> np.ndarray:
    if values.shape[0] == 1:
        return np.repeat(values, target_time.size, axis=0).astype(np.float32)
    output = np.empty((target_time.size, values.shape[1]), dtype=np.float32)
    for index in range(values.shape[1]):
        output[:, index] = np.interp(
            target_time,
            source_time,
            values[:, index],
            left=float(values[0, index]),
            right=float(values[-1, index]),
        )
    return output


class QualityAwareCSIProcessor:
    """Build temporally aligned, receiver-masked feature families for Signal-v2."""

    def __init__(self, config: dict[str, Any]) -> None:
        preprocessing = config["preprocessing"]
        quality = config["quality_control"]
        self.sample_rate = int(preprocessing.get("sample_rate", 1000))
        self.low_cut_hz = float(preprocessing.get("filter_low_hz", 2.0))
        self.high_cut_hz = float(preprocessing.get("filter_high_hz", 60.0))
        self.stft_window = int(preprocessing.get("stft_window", 251))
        self.stft_hop = int(preprocessing.get("stft_hop", 50))
        self.stft_overlap = int(preprocessing.get("stft_overlap", self.stft_window - self.stft_hop))
        if self.stft_window - self.stft_overlap != self.stft_hop:
            raise ValueError("stft_hop must equal stft_window - stft_overlap")
        self.stft_nfft = int(preprocessing.get("stft_nfft", 256))
        if self.stft_nfft < self.stft_window:
            raise ValueError("stft_nfft must be at least stft_window")
        self.doppler_min_hz = float(preprocessing.get("doppler_min_hz", -60.0))
        self.doppler_max_hz = float(preprocessing.get("doppler_max_hz", 60.0))
        self.doppler_bins = int(preprocessing.get("doppler_bins", 25))
        self.doppler_pca_components = int(preprocessing.get("doppler_pca_components", 3))
        self.differential_components = int(preprocessing.get("differential_pca_components", 10))
        self.num_receivers = int(config["data"].get("num_receivers", 6))
        self.min_valid_receivers = int(config["data"].get("min_valid_receivers", 5))
        self.minimum_length = int(
            quality.get("minimum_packets", self.stft_window + 4 * self.stft_hop)
        )
        self.minimum_length_ratio = float(quality.get("minimum_length_ratio", 0.60))
        self.maximum_gap_ratio = float(quality.get("maximum_timestamp_gap_ratio", 0.10))
        self.quality_control_enabled = bool(quality.get("enabled", True))
        self.alignment_strategy = str(quality.get("alignment_strategy", "median_valid"))
        if self.alignment_strategy not in {"median_valid", "minimum"}:
            raise ValueError(f"Unsupported alignment strategy: {self.alignment_strategy}")
        self.filter_sos = signal.butter(
            4,
            [self.low_cut_hz, self.high_cut_hz],
            btype="bandpass",
            fs=self.sample_rate,
            output="sos",
        )

    def _differential_csi(self, csi: np.ndarray) -> np.ndarray:
        if csi.shape[1] % 30:
            raise ValueError(f"CSI width must be divisible by 30, got {csi.shape[1]}")
        streams = csi.shape[1] // 30
        reshaped = np.abs(csi).reshape(csi.shape[0], streams, 30)
        score = reshaped.mean(axis=(0, 2)) / np.maximum(reshaped.var(axis=(0, 2)), 1e-8)
        reference_index = int(np.argmax(score))
        amplitude = np.abs(csi)
        floor = amplitude.min(axis=0, keepdims=True)
        adjusted = np.maximum(amplitude - floor, 0.0) * np.exp(1j * np.angle(csi))
        adjusted = adjusted.reshape(csi.shape[0], streams, 30)
        reference = csi[:, reference_index * 30 : (reference_index + 1) * 30]
        beta = 1000.0 * float(floor.mean())
        reference_adjusted = (np.abs(reference) + beta) * np.exp(1j * np.angle(reference))
        differential = adjusted * np.conj(reference_adjusted[:, None, :])
        keep = np.ones(streams, dtype=bool)
        keep[reference_index] = False
        differential = differential[:, keep, :].reshape(csi.shape[0], -1)
        if differential.shape[0] > 16:
            real = signal.sosfiltfilt(self.filter_sos, differential.real, axis=0)
            imag = signal.sosfiltfilt(self.filter_sos, differential.imag, axis=0)
        else:
            real = signal.sosfilt(self.filter_sos, differential.real, axis=0)
            imag = signal.sosfilt(self.filter_sos, differential.imag, axis=0)
        return (real + 1j * imag).astype(np.complex64)

    def _doppler(self, differential: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        principal = _complex_pca(differential, self.doppler_pca_components)
        spectra: list[np.ndarray] = []
        frequencies: np.ndarray | None = None
        times: np.ndarray | None = None
        for component in range(principal.shape[1]):
            frequencies, times, zxx = signal.stft(
                principal[:, component],
                fs=self.sample_rate,
                window="hann",
                nperseg=self.stft_window,
                noverlap=self.stft_overlap,
                nfft=self.stft_nfft,
                return_onesided=False,
                boundary=None,
                padded=True,
            )
            spectra.append(np.abs(zxx))
        assert frequencies is not None and times is not None
        energy = np.sum(spectra, axis=0)
        order = np.argsort(frequencies)
        frequencies = frequencies[order]
        energy = energy[order]
        selected = (frequencies >= self.doppler_min_hz) & (frequencies <= self.doppler_max_hz)
        energy = energy[selected]
        bands = np.array_split(np.arange(energy.shape[0]), self.doppler_bins)
        aggregated = np.stack(
            [energy[indexes].max(axis=0) if indexes.size else np.zeros(energy.shape[1]) for indexes in bands],
            axis=1,
        )
        aggregated /= np.maximum(aggregated.sum(axis=1, keepdims=True), 1e-8)
        return aggregated.astype(np.float32), times.astype(np.float64)

    def _receiver_families(
        self,
        csi: np.ndarray,
        rssi: np.ndarray,
        timestamps: np.ndarray,
    ) -> tuple[dict[str, np.ndarray], np.ndarray]:
        differential = self._differential_csi(csi)
        doppler, stft_time = self._doppler(differential)
        source_time = normalized_timestamps(timestamps, self.sample_rate)
        differential_reduced = _complex_pca(differential, self.differential_components)
        differential_pairs = np.stack(
            (differential_reduced.real, differential_reduced.imag), axis=-1
        ).astype(np.float32)
        flat_differential = differential_pairs.reshape(differential_pairs.shape[0], -1)
        aligned_rssi = _zscore(_interpolate(rssi.astype(np.float32), source_time, stft_time))
        aligned_differential = _zscore(_interpolate(flat_differential, source_time, stft_time)).reshape(
            stft_time.size, self.differential_components, 2
        )
        return {
            "rssi": aligned_rssi,
            "doppler": doppler,
            "differential_csi": aligned_differential,
        }, stft_time

    def process_group(self, receiver_paths: list[str | Path]) -> dict[str, Any]:
        if len(receiver_paths) != self.num_receivers:
            raise ValueError(f"Expected {self.num_receivers} receiver files, got {len(receiver_paths)}")
        decoded = [read_receiver_file(path) for path in receiver_paths]
        lengths = np.asarray([item[0].shape[0] for item in decoded], dtype=np.int64)
        median_length = float(np.median(lengths))
        qualities: list[ReceiverQuality] = [
            estimate_quality(
                csi=item[0],
                timestamps=item[2],
                group_median_length=median_length,
                sample_rate=self.sample_rate,
                minimum_length=self.minimum_length,
                minimum_length_ratio=self.minimum_length_ratio,
                maximum_gap_ratio=self.maximum_gap_ratio,
            )
            for item in decoded
        ]
        receiver_mask = np.asarray([quality.valid for quality in qualities], dtype=bool)
        if not self.quality_control_enabled:
            receiver_mask = np.ones(self.num_receivers, dtype=bool)
        if int(receiver_mask.sum()) < self.min_valid_receivers:
            raise ValueError(
                f"Only {int(receiver_mask.sum())}/{self.num_receivers} receivers passed quality control"
            )

        families: list[dict[str, np.ndarray] | None] = []
        time_axes: list[np.ndarray | None] = []
        for valid, (csi, rssi, timestamps) in zip(receiver_mask, decoded, strict=True):
            if valid:
                receiver_features, receiver_time = self._receiver_families(csi, rssi, timestamps)
                families.append(receiver_features)
                time_axes.append(receiver_time)
            else:
                families.append(None)
                time_axes.append(None)

        valid_lengths = [len(axis) for axis in time_axes if axis is not None]
        if self.alignment_strategy == "minimum":
            target_length = min(valid_lengths)
        else:
            target_length = max(1, int(round(float(np.median(valid_lengths)))))
        target_time = np.arange(target_length, dtype=np.float64) * self.stft_hop / self.sample_rate
        target_time += self.stft_window / (2.0 * self.sample_rate)
        output = {
            "rssi": np.zeros((target_length, self.num_receivers, 4), dtype=np.float32),
            "doppler": np.zeros((target_length, self.num_receivers, self.doppler_bins), dtype=np.float32),
            "differential_csi": np.zeros(
                (target_length, self.num_receivers, self.differential_components, 2), dtype=np.float32
            ),
        }
        for receiver_index, (feature_set, source_time) in enumerate(zip(families, time_axes, strict=True)):
            if feature_set is None or source_time is None:
                continue
            output["rssi"][:, receiver_index] = _interpolate(feature_set["rssi"], source_time, target_time)
            output["doppler"][:, receiver_index] = _interpolate(
                feature_set["doppler"], source_time, target_time
            )
            flat = feature_set["differential_csi"].reshape(len(source_time), -1)
            output["differential_csi"][:, receiver_index] = _interpolate(
                flat, source_time, target_time
            ).reshape(target_length, self.differential_components, 2)
        output.update(
            {
                "features": np.concatenate(
                    (
                        output["rssi"],
                        output["doppler"],
                        output["differential_csi"].reshape(target_length, self.num_receivers, -1),
                    ),
                    axis=-1,
                ).astype(np.float32),
                "time_mask": np.ones(target_length, dtype=bool),
                "frame_times_ms": (target_time * 1000.0).astype(np.float32),
                "receiver_mask": receiver_mask,
                "receiver_quality": np.stack([quality.vector() for quality in qualities]),
                "quality_fields": np.asarray(QUALITY_FIELDS),
                "quality_metadata": np.asarray(
                    json.dumps([asdict(quality) for quality in qualities], ensure_ascii=False)
                ),
            }
        )
        return output

    def save_group(
        self,
        receiver_paths: list[str | Path],
        output_path: str | Path,
        metadata: dict[str, Any],
    ) -> Path:
        arrays = self.process_group(receiver_paths)
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f"{target.stem}.tmp.npz")
        try:
            np.savez_compressed(
                temporary,
                **arrays,
                metadata=np.asarray(json.dumps(metadata, ensure_ascii=False)),
            )
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
        return target
