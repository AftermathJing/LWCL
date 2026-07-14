from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from scipy import signal

from .intel5300 import read_receiver_file


def complex_pca(values: np.ndarray, components: int) -> np.ndarray:
    centered = values - values.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    basis = vh[: min(components, vh.shape[0])].conj().T
    return centered @ basis


def _interpolate_time(values: np.ndarray, source_time: np.ndarray, target_time: np.ndarray) -> np.ndarray:
    output = np.empty((target_time.size, values.shape[1]), dtype=np.float32)
    for index in range(values.shape[1]):
        output[:, index] = np.interp(target_time, source_time, values[:, index])
    return output


def _zscore(values: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    mean = values.mean(axis=0, keepdims=True)
    std = values.std(axis=0, keepdims=True)
    return ((values - mean) / np.maximum(std, epsilon)).astype(np.float32)


class CSIToFeatureProcessor:
    """Reconstruct the paper's RSSI + Doppler + differential-CSI feature pipeline."""

    def __init__(
        self,
        sample_rate: int = 1000,
        low_cut_hz: float = 2.0,
        high_cut_hz: float = 60.0,
        doppler_bins: int = 25,
        doppler_pca_components: int = 3,
        differential_pca_components: int = 10,
        stft_window: int = 251,
        stft_overlap: int = 125,
    ) -> None:
        self.sample_rate = sample_rate
        self.low_cut_hz = low_cut_hz
        self.high_cut_hz = high_cut_hz
        self.doppler_bins = doppler_bins
        self.doppler_pca_components = doppler_pca_components
        self.differential_pca_components = differential_pca_components
        self.stft_window = stft_window
        self.stft_overlap = stft_overlap
        self.filter_sos = signal.butter(
            4,
            [low_cut_hz, high_cut_hz],
            btype="bandpass",
            fs=sample_rate,
            output="sos",
        )

    def _differential_csi(self, csi: np.ndarray) -> np.ndarray:
        if csi.shape[1] % 30:
            raise ValueError(f"CSI stream dimension must be divisible by 30, got {csi.shape[1]}")
        antenna_pairs = csi.shape[1] // 30
        reshaped = np.abs(csi).reshape(csi.shape[0], antenna_pairs, 30)
        ratio = reshaped.mean(axis=(0, 2)) / np.maximum(reshaped.std(axis=(0, 2)), 1e-6)
        reference_index = int(np.argmax(ratio))
        reference = csi[:, reference_index * 30 : (reference_index + 1) * 30]
        reference = np.tile(reference, (1, antenna_pairs))
        amplitude = np.abs(csi)
        floor = amplitude.min(axis=0, keepdims=True)
        adjusted = np.maximum(amplitude - floor, 0.0) * np.exp(1j * np.angle(csi))
        beta = 1000.0 * float(floor.mean())
        reference_adjusted = (np.abs(reference) + beta) * np.exp(1j * np.angle(reference))
        differential = adjusted * np.conj(reference_adjusted)
        keep = np.ones(csi.shape[1], dtype=bool)
        keep[reference_index * 30 : (reference_index + 1) * 30] = False
        differential = differential[:, keep]
        if differential.shape[0] > 16:
            real = signal.sosfiltfilt(self.filter_sos, differential.real, axis=0)
            imag = signal.sosfiltfilt(self.filter_sos, differential.imag, axis=0)
        else:
            real = signal.sosfilt(self.filter_sos, differential.real, axis=0)
            imag = signal.sosfilt(self.filter_sos, differential.imag, axis=0)
        return real + 1j * imag

    def _doppler(self, differential: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        principal = complex_pca(differential, self.doppler_pca_components)
        spectra: list[np.ndarray] = []
        frequencies: np.ndarray | None = None
        stft_times: np.ndarray | None = None
        nperseg = min(self.stft_window, principal.shape[0])
        overlap = min(self.stft_overlap, max(nperseg - 1, 0))
        for component in range(principal.shape[1]):
            frequencies, stft_times, zxx = signal.stft(
                principal[:, component],
                fs=self.sample_rate,
                window="hann",
                nperseg=nperseg,
                noverlap=overlap,
                return_onesided=False,
                boundary=None,
            )
            spectra.append(np.abs(zxx))
        energy = np.sum(spectra, axis=0)
        order = np.argsort(frequencies)
        frequencies = frequencies[order]
        energy = energy[order]
        selected = (frequencies >= -self.high_cut_hz) & (frequencies <= self.high_cut_hz)
        energy = energy[selected]
        bands = np.array_split(np.arange(energy.shape[0]), self.doppler_bins)
        aggregated = np.stack(
            [energy[indexes].max(axis=0) if indexes.size else np.zeros(energy.shape[1]) for indexes in bands],
            axis=1,
        )
        aggregated /= np.maximum(aggregated.sum(axis=1, keepdims=True), 1e-8)
        return aggregated.astype(np.float32), stft_times.astype(np.float32)

    def process_receiver(self, path: str | Path) -> np.ndarray:
        csi, rssi, _ = read_receiver_file(path)
        differential = self._differential_csi(csi)
        doppler, target_time = self._doppler(differential)
        raw_time = np.arange(csi.shape[0], dtype=np.float32) / self.sample_rate
        differential_reduced = complex_pca(differential, self.differential_pca_components)
        differential_features = np.concatenate(
            (differential_reduced.real, differential_reduced.imag), axis=1
        ).astype(np.float32)
        rssi_aligned = _interpolate_time(rssi, raw_time, target_time)
        differential_aligned = _interpolate_time(differential_features, raw_time, target_time)
        features = np.concatenate((rssi_aligned, doppler, differential_aligned), axis=1)
        return _zscore(features)

    def process_sample_directory(self, directory: str | Path, num_receivers: int) -> np.ndarray:
        directory = Path(directory)
        receiver_files = sorted(
            directory.glob("*-r*.dat"),
            key=lambda path: int(re.search(r"-r(\d+)\.dat$", path.name).group(1)),
        )
        if len(receiver_files) < num_receivers:
            raise ValueError(f"Expected {num_receivers} receiver files in {directory}, found {len(receiver_files)}")
        receiver_features = [self.process_receiver(path) for path in receiver_files[:num_receivers]]
        common_length = min(feature.shape[0] for feature in receiver_features)
        feature_dim = receiver_features[0].shape[1]
        if any(feature.shape[1] != feature_dim for feature in receiver_features):
            raise ValueError("Receiver feature dimensions do not match")
        return np.stack([feature[:common_length] for feature in receiver_features], axis=1)

    def save_sample(
        self,
        input_directory: str | Path,
        output_path: str | Path,
        num_receivers: int,
        metadata: dict[str, Any],
    ) -> Path:
        features = self.process_sample_directory(input_directory, num_receivers)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_path,
            features=features.astype(np.float32),
            metadata=np.asarray(json.dumps(metadata, ensure_ascii=False)),
        )
        return output_path
