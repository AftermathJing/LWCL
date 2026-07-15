from __future__ import annotations

from dataclasses import dataclass

import numpy as np


QUALITY_FIELDS = (
    "length_ratio",
    "packet_retention",
    "timestamp_health",
    "motion_energy",
    "snr_proxy",
)


@dataclass(frozen=True)
class ReceiverQuality:
    length: int
    length_ratio: float
    packet_retention: float
    timestamp_gap_ratio: float
    motion_energy: float
    snr_proxy: float
    valid: bool

    def vector(self) -> np.ndarray:
        return np.asarray(
            [
                self.length_ratio,
                self.packet_retention,
                1.0 - self.timestamp_gap_ratio,
                self.motion_energy,
                self.snr_proxy,
            ],
            dtype=np.float32,
        )


def normalized_timestamps(timestamps: np.ndarray, sample_rate: int) -> np.ndarray:
    timestamps = np.asarray(timestamps, dtype=np.float64)
    if timestamps.size < 2:
        return np.arange(timestamps.size, dtype=np.float64) / sample_rate
    relative = (timestamps - timestamps[0]) * 1e-6
    differences = np.diff(relative)
    if not np.all(np.isfinite(relative)) or np.count_nonzero(differences > 0) < differences.size * 0.9:
        return np.arange(timestamps.size, dtype=np.float64) / sample_rate
    return relative


def timestamp_gap_ratio(timestamps: np.ndarray, sample_rate: int) -> float:
    relative = normalized_timestamps(timestamps, sample_rate)
    if relative.size < 2:
        return 1.0
    differences = np.diff(relative)
    positive = differences[differences > 0]
    if positive.size == 0:
        return 1.0
    expected = 1.0 / sample_rate
    median = float(np.median(positive))
    threshold = max(3.0 * median, 3.0 * expected)
    anomalous = (differences <= 0) | (differences > threshold)
    return float(anomalous.mean())


def estimate_quality(
    csi: np.ndarray,
    timestamps: np.ndarray,
    group_median_length: float,
    sample_rate: int,
    minimum_length: int,
    minimum_length_ratio: float,
    maximum_gap_ratio: float,
) -> ReceiverQuality:
    length = int(csi.shape[0])
    length_ratio = length / max(float(group_median_length), 1.0)
    relative = normalized_timestamps(timestamps, sample_rate)
    duration = float(relative[-1] - relative[0]) if relative.size > 1 else 0.0
    expected_packets = max(duration * sample_rate + 1.0, float(length))
    packet_retention = float(np.clip(length / expected_packets, 0.0, 1.0))
    gap_ratio = timestamp_gap_ratio(timestamps, sample_rate)

    amplitude = np.abs(csi).astype(np.float64)
    if length > 1:
        motion = np.mean(np.abs(np.diff(amplitude, axis=0)))
    else:
        motion = 0.0
    scale = float(np.mean(amplitude)) + 1e-8
    motion_energy = float(np.log1p(motion / scale))
    snr_proxy = float(np.log1p(scale / (float(np.std(amplitude)) + 1e-8)))
    valid = (
        length >= minimum_length
        and length_ratio >= minimum_length_ratio
        and gap_ratio <= maximum_gap_ratio
    )
    return ReceiverQuality(
        length=length,
        length_ratio=float(length_ratio),
        packet_retention=packet_retention,
        timestamp_gap_ratio=gap_ratio,
        motion_energy=motion_energy,
        snr_proxy=snr_proxy,
        valid=bool(valid),
    )
