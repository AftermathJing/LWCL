from __future__ import annotations

import csv
import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib import colormaps
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv
from PIL import Image
from scipy import ndimage, signal

from lwcl.data.intel5300 import read_receiver_file
from lwcl.data.preprocessing import complex_pca


def _resolve_manifest_path(manifest_path: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (manifest_path.parent / path).resolve()


def _relative_path(path: Path, root: Path) -> str:
    return Path(os.path.relpath(path, root)).as_posix()


def _resize_float(values: np.ndarray, height: int, width: int) -> np.ndarray:
    if values.shape == (height, width):
        return values.astype(np.float32, copy=False)
    zoom = (height / max(values.shape[0], 1), width / max(values.shape[1], 1))
    resized = ndimage.zoom(values, zoom, order=1)
    return resized.astype(np.float32, copy=False)


def _normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return np.zeros_like(values, dtype=np.float32)
    lower = float(np.nanmin(values))
    upper = float(np.nanmax(values))
    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
        return np.zeros_like(values, dtype=np.float32)
    normalized = (values - lower) / (upper - lower)
    return np.clip(normalized, 0.0, 1.0).astype(np.float32, copy=False)


def _matrix_to_rgb(values: np.ndarray, image_size: int, cmap: str) -> np.ndarray:
    resized = _resize_float(_normalize(values), image_size, image_size)
    rgba = colormaps[cmap](resized)
    return (rgba[..., :3] * 255.0).astype(np.uint8)


def _save_rgb(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array, mode="RGB").save(path, format="JPEG", quality=95)


def _prepare_job(job: dict[str, Any]) -> dict[str, Any]:
    phase_raw_path = Path(job["phase_raw_path"])
    phase_weighted_path = Path(job["phase_weighted_path"])
    dfs_path = Path(job["dfs_path"])
    weight_path = Path(job["dfs_weight_path"])
    if job["resume"] and all(path.exists() for path in (phase_raw_path, phase_weighted_path, dfs_path, weight_path)):
        return {"sample_id": job["sample_id"], "generated": False}
    processor = WiCBRPreprocessor(image_size=int(job["image_size"]), colormap=str(job["colormap"]))
    rendered = processor.build_sample_images(Path(job["sample_root"]), int(job["receiver_count"]))
    _save_rgb(phase_raw_path, rendered["phase_raw"])
    _save_rgb(phase_weighted_path, rendered["phase_weighted"])
    _save_rgb(dfs_path, rendered["dfs"])
    _save_rgb(weight_path, np.repeat(rendered["dfs_weight"][..., None], 3, axis=2))
    return {"sample_id": job["sample_id"], "generated": True}


@dataclass(slots=True)
class WiCBRPreprocessor:
    sample_rate: int = 1000
    low_cut_hz: float = 2.0
    high_cut_hz: float = 60.0
    doppler_pca_components: int = 3
    image_size: int = 224
    colormap: str = "viridis"
    filter_sos: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.filter_sos = signal.butter(
            4,
            [self.low_cut_hz, self.high_cut_hz],
            btype="bandpass",
            fs=self.sample_rate,
            output="sos",
        )

    def _reshape_antennas(self, csi: np.ndarray) -> np.ndarray:
        if csi.ndim != 2 or csi.shape[1] % 30:
            raise ValueError(f"Receiver CSI must have shape [time, antennas*30], got {csi.shape}")
        antenna_count = csi.shape[1] // 30
        return csi.reshape(csi.shape[0], antenna_count, 30)

    def _antenna_ratio(self, reshaped: np.ndarray) -> np.ndarray:
        amplitude = np.abs(reshaped)
        mean = amplitude.mean(axis=(0, 2))
        variance = amplitude.var(axis=(0, 2))
        return mean / np.maximum(variance, 1e-6)

    def build_phase_matrix(self, csi: np.ndarray) -> np.ndarray:
        reshaped = self._reshape_antennas(csi)
        ratio = self._antenna_ratio(reshaped)
        max_index = int(np.argmax(ratio))
        min_index = int(np.argmin(ratio))
        numerator = reshaped[:, max_index, :].T
        denominator = reshaped[:, min_index, :].T
        phase = np.angle(numerator / np.maximum(np.abs(denominator), 1e-6) * np.exp(-1j * np.angle(denominator)))
        return phase.astype(np.float32)

    def _filtered_differential_csi(self, csi: np.ndarray) -> np.ndarray:
        reshaped = self._reshape_antennas(csi)
        antenna_count = reshaped.shape[1]
        ratio = self._antenna_ratio(reshaped)
        reference_index = int(np.argmax(ratio))
        reference = csi[:, reference_index * 30 : (reference_index + 1) * 30]
        reference = np.tile(reference, (1, antenna_count))
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

    def build_doppler_matrix(self, csi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        differential = self._filtered_differential_csi(csi)
        principal = complex_pca(differential, self.doppler_pca_components)
        spectra: list[np.ndarray] = []
        frequencies: np.ndarray | None = None
        nperseg = min(251, principal.shape[0])
        noverlap = min(125, max(0, nperseg - 1))
        for component in range(principal.shape[1]):
            frequencies, _, zxx = signal.stft(
                principal[:, component],
                fs=self.sample_rate,
                window="hann",
                nperseg=nperseg,
                noverlap=noverlap,
                return_onesided=False,
                boundary=None,
            )
            spectra.append(np.abs(zxx))
        if frequencies is None:
            raise ValueError("Failed to build STFT frequencies")
        energy = np.sum(spectra, axis=0)
        order = np.argsort(frequencies)
        frequencies = frequencies[order]
        energy = energy[order]
        selected = (frequencies >= -self.high_cut_hz) & (frequencies <= self.high_cut_hz)
        frequencies = frequencies[selected]
        energy = energy[selected]
        if energy.size == 0:
            raise ValueError("No Doppler bins survived the passband selection")
        energy = energy.astype(np.float32)
        energy /= np.maximum(energy.sum(axis=0, keepdims=True), 1e-8)
        return energy, frequencies.astype(np.float32)

    def build_sample_images(self, sample_root: str | Path, num_receivers: int) -> dict[str, np.ndarray]:
        sample_root = Path(sample_root)
        receiver_files = sorted(sample_root.parent.glob(f"{sample_root.name}-r*.dat"))
        if len(receiver_files) < num_receivers:
            raise ValueError(f"Expected {num_receivers} receiver files in {sample_root}, found {len(receiver_files)}")

        phase_mats: list[np.ndarray] = []
        dfs_mats: list[np.ndarray] = []
        weight_rows: list[np.ndarray] = []

        common_phase_width: int | None = None
        common_dfs_width: int | None = None
        common_dfs_height: int | None = None

        receiver_results: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for receiver_path in receiver_files[:num_receivers]:
            csi, _, _ = read_receiver_file(receiver_path)
            phase = self.build_phase_matrix(csi)
            dfs, frequencies = self.build_doppler_matrix(csi)
            receiver_results.append((phase, dfs, frequencies))
            common_phase_width = phase.shape[1] if common_phase_width is None else min(common_phase_width, phase.shape[1])
            common_dfs_width = dfs.shape[1] if common_dfs_width is None else min(common_dfs_width, dfs.shape[1])
            common_dfs_height = dfs.shape[0] if common_dfs_height is None else min(common_dfs_height, dfs.shape[0])

        assert common_phase_width is not None and common_dfs_width is not None and common_dfs_height is not None
        for phase, dfs, frequencies in receiver_results:
            phase = phase[:, :common_phase_width]
            dfs = dfs[:common_dfs_height, :common_dfs_width]
            frequencies = frequencies[:common_dfs_height]
            phase_mats.append(phase)
            dfs_mats.append(dfs)
            weight_rows.append(np.sum(np.abs(frequencies)[:, None] * dfs, axis=0).astype(np.float32))

        phase_stack = np.concatenate(phase_mats, axis=0)
        dfs_stack = np.concatenate(dfs_mats, axis=0)
        phase_rgb = _matrix_to_rgb(phase_stack, self.image_size, self.colormap)
        dfs_rgb = _matrix_to_rgb(dfs_stack, self.image_size, self.colormap)

        receiver_time_weight = np.stack(weight_rows, axis=0)
        weight_map = _resize_float(_normalize(receiver_time_weight), self.image_size, self.image_size)
        brightness_map = 0.2 + 0.8 * weight_map
        phase_hsv = rgb_to_hsv(phase_rgb.astype(np.float32) / 255.0)
        phase_hsv[..., 2] = np.clip(phase_hsv[..., 2] * brightness_map, 0.0, 1.0)
        phase_weighted = (hsv_to_rgb(phase_hsv) * 255.0).astype(np.uint8)

        return {
            "phase_raw": phase_rgb,
            "phase_weighted": phase_weighted,
            "dfs": dfs_rgb,
            "dfs_weight": (weight_map * 255.0).astype(np.uint8),
        }


def prepare_wicbr_manifest(
    input_manifest: str | Path,
    output_dir: str | Path,
    output_manifest: str | Path,
    *,
    raw_manifest: str | Path | None = None,
    num_receivers: int = 6,
    image_size: int = 224,
    colormap: str = "viridis",
    phase_source: str = "weighted",
    workers: int = 1,
    limit: int | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    input_manifest = Path(input_manifest).resolve()
    output_dir = Path(output_dir).resolve()
    output_manifest = Path(output_manifest).resolve()
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    processor = WiCBRPreprocessor(image_size=image_size, colormap=colormap)

    raw_manifest_path: Path | None = Path(raw_manifest).resolve() if raw_manifest is not None else None
    if raw_manifest_path is None:
        candidate = input_manifest.parents[1] / "raw" / "widar3_manifest.csv"
        if candidate.exists():
            raw_manifest_path = candidate

    raw_lookup: dict[str, dict[str, str]] = {}
    if raw_manifest_path is not None and raw_manifest_path.exists():
        with raw_manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
            raw_lookup = {
                row["sample_id"]: row
                for row in csv.DictReader(handle)
                if row.get("sample_id")
            }

    with input_manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if limit is not None:
        rows = rows[:limit]

    prepared_rows: list[dict[str, Any]] = []
    for row in rows:
        sample_id = row.get("sample_id", "")
        raw_row = raw_lookup.get(sample_id, {})
        merged_row = {**raw_row, **row}
        raw_value = merged_row.get("raw_path")
        source_manifest = raw_manifest_path if raw_value and raw_row else input_manifest
        if not raw_value:
            prepared_rows.append({"failure": {"sample_id": sample_id, "error": "missing raw_path"}})
            continue
        sample_id = merged_row.get("sample_id", Path(raw_value).name)
        sample_root = _resolve_manifest_path(source_manifest or input_manifest, raw_value)
        phase_raw_path = output_dir / "phase_raw" / f"{sample_id}.jpg"
        phase_weighted_path = output_dir / "phase_weighted" / f"{sample_id}.jpg"
        dfs_path = output_dir / "dfs" / f"{sample_id}.jpg"
        weight_path = output_dir / "dfs_weight" / f"{sample_id}.jpg"
        prepared_rows.append(
            {
                "sample_id": sample_id,
                "raw_path_original": raw_value,
                "merged_row": merged_row,
                "output_row": {
                    **merged_row,
                    "phase_raw_path": _relative_path(phase_raw_path, output_manifest.parent),
                    "phase_weighted_path": _relative_path(phase_weighted_path, output_manifest.parent),
                    "phase_path": _relative_path(
                        phase_weighted_path if phase_source == "weighted" else phase_raw_path,
                        output_manifest.parent,
                    ),
                    "dfs_path": _relative_path(dfs_path, output_manifest.parent),
                    "dfs_weight_path": _relative_path(weight_path, output_manifest.parent),
                },
                "job": {
                    "sample_id": sample_id,
                    "sample_root": str(sample_root),
                    "receiver_count": int(merged_row.get("receiver_count", num_receivers) or num_receivers),
                    "phase_raw_path": str(phase_raw_path),
                    "phase_weighted_path": str(phase_weighted_path),
                    "dfs_path": str(dfs_path),
                    "dfs_weight_path": str(weight_path),
                    "resume": resume,
                    "image_size": image_size,
                    "colormap": colormap,
                },
            }
        )

    output_rows: list[dict[str, str]] = []
    failures: list[dict[str, str]] = []
    generated = 0
    jobs = [row for row in prepared_rows if "job" in row]
    if workers <= 1:
        job_results = []
        for item in jobs:
            try:
                result = _prepare_job(item["job"])
                job_results.append((item, result, None))
            except Exception as error:  # pragma: no cover
                job_results.append((item, None, error))
    else:
        job_results = []
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [(item, executor.submit(_prepare_job, item["job"])) for item in jobs]
            for item, future in futures:
                try:
                    result = future.result()
                    job_results.append((item, result, None))
                except Exception as error:  # pragma: no cover
                    job_results.append((item, None, error))

    for item in prepared_rows:
        if "failure" in item:
            failures.append(item["failure"])

    for item, result, error in job_results:
        if error is not None:
            failures.append(
                {
                    "sample_id": item["sample_id"],
                    "error": str(error),
                    "raw_path": item["raw_path_original"],
                }
            )
            continue
        if bool(result and result.get("generated")):
            generated += 1
        output_rows.append(item["output_row"])

    fieldnames = list(output_rows[0].keys()) if output_rows else [
        "sample_id",
        "raw_path",
        "label",
        "split",
        "phase_raw_path",
        "phase_weighted_path",
        "phase_path",
        "dfs_path",
        "dfs_weight_path",
    ]
    with output_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    failures_path = output_manifest.with_suffix(".failures.csv")
    with failures_path.open("w", encoding="utf-8", newline="") as handle:
        if failures:
            writer = csv.DictWriter(handle, fieldnames=list(failures[0].keys()))
            writer.writeheader()
            writer.writerows(failures)
        else:
            handle.write("sample_id,error\n")

    summary = {
        "input_manifest": str(input_manifest),
        "raw_manifest": str(raw_manifest_path) if raw_manifest_path is not None else None,
        "output_manifest": str(output_manifest),
        "output_dir": str(output_dir),
        "phase_source": phase_source,
        "workers": workers,
        "generated": generated,
        "rows_written": len(output_rows),
        "failures": len(failures),
    }
    (output_manifest.with_suffix(".summary.json")).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary
