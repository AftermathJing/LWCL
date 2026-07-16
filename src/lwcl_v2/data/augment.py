from __future__ import annotations

from typing import Any

import numpy as np


class SignalAugmenter:
    def __init__(self, config: dict[str, Any], min_valid_receivers: int = 5) -> None:
        self.config = config
        self.min_valid_receivers = min_valid_receivers

    @staticmethod
    def _enabled(config: dict[str, Any], rng: np.random.Generator) -> bool:
        return bool(config) and rng.random() < float(config.get("probability", 0.0))

    def __call__(self, sample: dict[str, np.ndarray], rng: np.random.Generator) -> dict[str, np.ndarray]:
        output = {key: value.copy() for key, value in sample.items()}
        signal_keys = tuple(
            key
            for key in ("rssi", "amplitude", "doppler", "differential_csi", "csi_ratio_phase")
            if key in output
        )
        amplitude = self.config.get("amplitude_scale", {})
        if self._enabled(amplitude, rng):
            low, high = amplitude.get("range", [0.9, 1.1])
            scale = float(rng.uniform(low, high))
            for key in ("rssi", "amplitude", "differential_csi"):
                if key in output:
                    output[key] *= scale

        noise = self.config.get("gaussian_noise", {})
        if self._enabled(noise, rng):
            low, high = noise.get("std_range", [0.005, 0.015])
            std = float(rng.uniform(low, high))
            for key in signal_keys:
                output[key] += rng.normal(0.0, std, size=output[key].shape).astype(np.float32)
            if "csi_ratio_phase" in output:
                phase_norm = np.linalg.norm(output["csi_ratio_phase"], axis=-1, keepdims=True)
                output["csi_ratio_phase"] = output["csi_ratio_phase"] / np.maximum(
                    phase_norm, 1e-6
                )

        shift_config = self.config.get("temporal_shift", {})
        if self._enabled(shift_config, rng):
            maximum = int(shift_config.get("max_frames", 2))
            shift = int(rng.integers(-maximum, maximum + 1))
            if shift:
                for key in (*signal_keys, "time_mask"):
                    output[key] = np.roll(output[key], shift, axis=0)
                    if shift > 0:
                        output[key][:shift] = 0
                    else:
                        output[key][shift:] = 0

        temporal_mask = self.config.get("temporal_mask", {})
        if self._enabled(temporal_mask, rng) and output["time_mask"].size > 1:
            maximum = max(1, int(round(output["time_mask"].size * float(temporal_mask.get("max_ratio", 0.1)))))
            width = int(rng.integers(1, maximum + 1))
            start = int(rng.integers(0, output["time_mask"].size - width + 1))
            for key in signal_keys:
                output[key][start : start + width] = 0
            output["time_mask"][start : start + width] = False

        receiver_dropout = self.config.get("receiver_dropout", {})
        valid_receivers = np.flatnonzero(output["receiver_mask"])
        if self._enabled(receiver_dropout, rng) and valid_receivers.size > self.min_valid_receivers:
            maximum = min(
                int(receiver_dropout.get("max_receivers", 1)),
                int(valid_receivers.size - self.min_valid_receivers),
            )
            count = int(rng.integers(1, maximum + 1))
            dropped = rng.choice(valid_receivers, size=count, replace=False)
            output["receiver_mask"][dropped] = False
            for key in signal_keys:
                output[key][:, dropped] = 0

        frequency_mask = self.config.get("doppler_frequency_mask", {})
        if self._enabled(frequency_mask, rng):
            target_key = "doppler" if "doppler" in output else "amplitude"
            maximum = min(int(frequency_mask.get("max_bins", 2)), output[target_key].shape[-1])
            width = int(rng.integers(1, maximum + 1))
            start = int(rng.integers(0, output[target_key].shape[-1] - width + 1))
            output[target_key][..., start : start + width] = 0
        return output
