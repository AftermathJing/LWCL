from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy import signal as scipy_signal


HAR_PROTOCOLS = (
    "test_id",
    "test_cross_device",
    "test_cross_env",
    "test_cross_user",
    "test_cross_user_env",
    "test_instance",
)


def resolve_csi_bench_path(dataset_root: str | Path, task_dir: str | Path, file_path: str) -> Path:
    root = Path(dataset_root).resolve()
    original = Path(file_path)
    # CSI-Bench metadata stores paths relative to the task metadata
    # directory, using ../../sub_Human_h5/... form. Resolve against
    # <task_dir>/metadata, then try broader fallbacks.
    task = Path(task_dir).resolve()
    metadata_candidate = (task / "metadata" / original).resolve()
    if metadata_candidate.exists():
        return metadata_candidate
    candidates = [
        original if original.is_absolute() else None,
        task / original,
        root / original,
        root / "Multitask" / original,
    ]
    for candidate in candidates:
        if candidate is not None and candidate.resolve().exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Cannot resolve CSI-Bench file {file_path!r} under {root}")


def _orient_frequency_time(values: np.ndarray, num_subcarriers: int | None) -> np.ndarray:
    array = np.asarray(values)
    if np.iscomplexobj(array):
        array = np.abs(array)
    array = np.squeeze(array)
    if array.ndim != 2:
        raise ValueError(f"Expected CSI_amps to become 2D after squeeze, got {array.shape}")
    expected = int(num_subcarriers) if num_subcarriers not in (None, 0) else None
    if expected is not None and array.shape[0] == expected:
        return array.T.astype(np.float32, copy=False)
    if expected is not None and array.shape[1] == expected:
        return array.astype(np.float32, copy=False)
    # Released CSI-Bench H5 files are conventionally [F,T,1].
    return (array.T if array.shape[0] <= array.shape[1] else array).astype(np.float32, copy=False)


def resample_frequency(values: np.ndarray, bins: int) -> np.ndarray:
    """Interpolate [T,N,F] to a hardware-width invariant [T,N,bins] tensor."""
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 3:
        raise ValueError(f"Expected [T,N,F], got {values.shape}")
    if values.shape[-1] == bins:
        return values
    source = np.linspace(0.0, 1.0, values.shape[-1], dtype=np.float32)
    target = np.linspace(0.0, 1.0, bins, dtype=np.float32)
    output = np.empty((*values.shape[:2], bins), dtype=np.float32)
    for t in range(values.shape[0]):
        for receiver in range(values.shape[1]):
            output[t, receiver] = np.interp(target, source, values[t, receiver])
    return output


def load_csi_bench_amplitude(
    path: str | Path,
    *,
    num_subcarriers: int | None = None,
    num_devices: int = 1,
    amplitude_bins: int = 64,
    frame_interval_ms: float = 10.0,
) -> dict[str, np.ndarray]:
    """Load one CSI-Bench H5 sample as T00-compatible amplitude features.

    Mimics the Widar3 QualityAwareCSIProcessor pipeline: amplitude →
    differential → STFT → Doppler spectrum + differential CSI PCA so that
    the T00 feature-family stems receive the same signal types (RSSI proxy,
    DFS, differential CSI pairs) as during Widar3 training.
    """
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover
        raise ImportError("CSI-Bench amplitude loading requires h5py") from exc
    with h5py.File(path, "r") as handle:
        values = _orient_frequency_time(np.asarray(handle["CSI_amps"]), num_subcarriers)
    devices = int(num_devices)
    if devices < 1 or values.shape[-1] % devices:
        raise ValueError(f"Frequency width {values.shape[1]} is incompatible with num_devices={devices}")
    # --- per-device amplitude → differential → STFT ---------------------------------
    sample_rate = 100  # CSI-Bench packets are ~10 ms, verified by README
    device_signals = values.reshape(values.shape[0], devices, values.shape[1] // devices)
    device_signals = (device_signals - device_signals.mean(axis=(0, 2), keepdims=True)) / np.maximum(
        device_signals.std(axis=(0, 2), keepdims=True), 1e-6
    )
    doppler_list: list[np.ndarray] = []
    differential_list: list[np.ndarray] = []
    rssi_list: list[np.ndarray] = []
    stft_times: np.ndarray | None = None
    for d in range(devices):
        amp = device_signals[:, d, :]  # [T, F_device]
        # differential CSI (帧间差 via 实幅度, 构造假复数避免改动主干)
        diff_amplitude = np.diff(amp, axis=0)  # [T-1, F_device]
        diff_amplitude = np.vstack([diff_amplitude[:1], diff_amplitude])  # keep T
        # bandpass filter 2-60 Hz
        sos = scipy_signal.butter(4, [2.0, 60.0], btype="bandpass", fs=sample_rate, output="sos")
        if diff_amplitude.shape[0] > 16:
            filtered = scipy_signal.sosfiltfilt(sos, diff_amplitude, axis=0)
        else:
            filtered = scipy_signal.sosfilt(sos, diff_amplitude, axis=0)
        fake_complex = (filtered + 1j * np.zeros_like(filtered)).astype(np.complex64)
        # PCA reduce to 3 components for STFT (matching Widar3 doppler_pca_components)
        centered = fake_complex - fake_complex.mean(axis=0, keepdims=True)
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        available = min(3, vh.shape[0])
        principal = (centered @ vh[:available].conj().T).astype(np.complex64)
        if available < 3:
            principal = np.pad(principal, ((0, 0), (0, 3 - available)))
        # STFT on each PCA component
        spectra: list[np.ndarray] = []
        for comp in range(3):
            freqs, times, zxx = scipy_signal.stft(
                principal[:, comp],
                fs=sample_rate,
                window="hann",
                nperseg=min(64, principal.shape[0]),  # shorter window for ~500-sample signals
                noverlap=min(48, principal.shape[0] // 2),
                nfft=128,
                return_onesided=False,
                boundary=None,
                padded=True,
            )
            spectra.append(np.abs(zxx))
            if stft_times is None:
                stft_times = times
        energy = np.sum(spectra, axis=0)  # [freq, time]
        order = np.argsort(freqs)
        freqs = freqs[order]
        energy = energy[order]
        # [-60, 60] Hz Doppler range → 25 bins max-pool
        mask = (freqs >= -60.0) & (freqs <= 60.0)
        freqs = freqs[mask]
        energy = energy[mask]
        bands = np.array_split(np.arange(energy.shape[0]), 25)
        dfs = np.stack(
            [energy[idx].max(axis=0) if idx.size else np.zeros(energy.shape[1]) for idx in bands], axis=1
        )
        dfs /= np.maximum(dfs.sum(axis=1, keepdims=True), 1e-8)
        # differential CSI PCA (10 components, real+imag pairs)
        diff_centered = fake_complex - fake_complex.mean(axis=0, keepdims=True)
        _, _, vh2 = np.linalg.svd(diff_centered, full_matrices=False)
        avail2 = min(10, vh2.shape[0])
        diff_proj = (diff_centered @ vh2[:avail2].conj().T).astype(np.complex64)
        if avail2 < 10:
            diff_proj = np.pad(diff_proj, ((0, 0), (0, 10 - avail2)))
        diff_interp = np.empty((len(stft_times), 10, 2), dtype=np.float32)
        source_t = np.arange(diff_proj.shape[0], dtype=np.float64) / sample_rate
        target_t = stft_times.astype(np.float64)
        for c in range(10):
            diff_interp[:, c, 0] = np.interp(target_t, source_t, diff_proj[:, c].real)
            diff_interp[:, c, 1] = np.interp(target_t, source_t, diff_proj[:, c].imag)
        # z-score
        mean_d = diff_interp.mean(axis=(0, 2), keepdims=True)
        std_d = diff_interp.std(axis=(0, 2), keepdims=True)
        diff_interp = (diff_interp - mean_d) / np.maximum(std_d, 1e-6)
        # RSSI proxy
        rssi_proxy = np.log1p(np.abs(amp).mean(axis=1))  # [T]
        rssi_interp = np.interp(target_t, source_t, rssi_proxy).astype(np.float32)
        rssi_mean, rssi_std = rssi_interp.mean(), rssi_interp.std()
        rssi = ((rssi_interp - rssi_mean) / max(rssi_std, 1e-6)).reshape(-1, 1)
        # 4-channel RSSI (repeat)
        rssi_list.append(np.tile(rssi, (1, 4)).astype(np.float32))
        doppler_list.append(dfs.astype(np.float32))
        differential_list.append(diff_interp.astype(np.float32))
    stft_len = int(stft_times.size) if stft_times is not None else 0
    receiver_mask = np.ones(devices, dtype=bool)
    energy_val = np.ones(devices, dtype=np.float32)
    snr_proxy = np.ones(devices, dtype=np.float32) * 10.0
    receiver_quality = np.stack(
        [np.ones(devices), np.zeros(devices), np.zeros(devices), energy_val, snr_proxy], axis=-1
    ).astype(np.float32)
    # collate across devices
    rssi_out = np.stack(rssi_list, axis=1)      # [T_stft, devices, 4]
    doppler_out = np.stack(doppler_list, axis=1) # [T_stft, devices, 25]
    diff_out = np.stack(differential_list, axis=1)  # [T_stft, devices, 10, 2]
    return {
        "rssi": rssi_out,
        "doppler": doppler_out,
        "differential_csi": diff_out,
        "time_mask": np.ones(stft_len, dtype=bool),
        "frame_times_ms": (stft_times * 1000.0).astype(np.float32) if stft_times is not None else np.zeros(stft_len, dtype=np.float32),
        "receiver_mask": receiver_mask,
        "receiver_quality": receiver_quality,
    }


def _load_json_ids(path: Path) -> set[str]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"Expected a list of sample IDs in {path}")
    return {str(value) for value in payload}


def build_har_protocol_manifests(
    dataset_root: str | Path,
    output_root: str | Path,
    protocols: Iterable[str] | None = None,
) -> dict[str, dict[str, int]]:
    """Create independent train/validation/test manifests for official HAR protocols."""
    root = Path(dataset_root).resolve()
    task_dir = root / "Multitask" / "HumanActivityRecognition"
    if not task_dir.exists():
        task_dir = root / "HumanActivityRecognition"
    metadata_dir = task_dir / "metadata"
    split_dir = task_dir / "splits"
    with (metadata_dir / "label_mapping.json").open("r", encoding="utf-8") as handle:
        label_mapping = json.load(handle)["label_to_idx"]
    with (metadata_dir / "sample_metadata.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        source_rows = list(csv.DictReader(handle))
    by_id = {str(row.get("id") or row.get("sample_id")): row for row in source_rows}
    selected = tuple(protocols or HAR_PROTOCOLS)
    summaries: dict[str, dict[str, int]] = {}
    for protocol in selected:
        if protocol not in HAR_PROTOCOLS:
            raise ValueError(f"Unsupported HAR protocol {protocol}; choose from {HAR_PROTOCOLS}")
        train_ids = _load_json_ids(split_dir / "train_id.json")
        validation_ids = _load_json_ids(split_dir / "val_id.json")
        test_ids = _load_json_ids(split_dir / f"{protocol}.json")
        if train_ids & validation_ids or train_ids & test_ids or validation_ids & test_ids:
            raise ValueError(f"Official split overlap detected for {protocol}")
        rows: list[dict[str, Any]] = []
        counts: Counter[str] = Counter()
        for split_name, ids in (("train", train_ids), ("validation", validation_ids), ("test", test_ids)):
            for sample_id in sorted(ids):
                if sample_id not in by_id:
                    raise KeyError(f"Split {protocol} references unknown sample {sample_id}")
                source = by_id[sample_id]
                label_name = str(source["label"])
                user = str(source.get("user") or source.get("subject") or "unknown")
                feature_path = resolve_csi_bench_path(root, task_dir, source["file_path"])
                row = dict(source)
                row.update(
                    {
                        "sample_id": sample_id,
                        "feature_path": str(feature_path),
                        "feature_format": "h5",
                        "data_key": "CSI_amps",
                        "feature_mode": "csi_bench_amplitude",
                        "label": int(label_mapping[label_name]),
                        "label_name": label_name,
                        "split": split_name,
                        "benchmark_split": protocol,
                        "subject": user,
                        "user": user,
                        "source_subject": str(source.get("subject", "")),
                        "device_count": 1,
                    }
                )
                rows.append(row)
                counts[split_name] += 1
        target = Path(output_root).resolve() / protocol
        target.mkdir(parents=True, exist_ok=True)
        manifest = target / "manifest.csv"
        fields: list[str] = []
        for row in rows:
            for field in row:
                if field not in fields:
                    fields.append(field)
        with manifest.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        summaries[protocol] = {"samples": len(rows), **dict(counts)}
    return summaries
