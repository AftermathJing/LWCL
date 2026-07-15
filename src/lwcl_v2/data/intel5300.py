from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np


def _db_inverse(value: float) -> float:
    return 10.0 ** (value / 10.0)


def _db_power(value: float) -> float:
    return 10.0 * math.log10(max(value, 1e-12))


def total_rssi(record: dict[str, Any]) -> float:
    power = sum(_db_inverse(record[key]) for key in ("rssi_a", "rssi_b", "rssi_c") if record[key] != 0)
    return _db_power(power) - 44.0 - record["agc"]


def parse_csi(payload: bytes, num_tx: int, num_rx: int) -> np.ndarray:
    """Decode Intel 5300 CSI payload to [Ntx,Nrx,30] complex values."""
    csi = np.zeros((num_tx, num_rx, 30), dtype=np.complex64)
    bit_index = 0
    for subcarrier in range(30):
        bit_index += 3
        remainder = bit_index % 8
        for rx in range(num_rx):
            for tx in range(num_tx):
                start = bit_index // 8
                if start + 2 >= len(payload):
                    raise ValueError("Truncated CSI payload")
                real_raw = ((payload[start] >> remainder) | (payload[start + 1] << (8 - remainder))) & 0xFF
                imag_raw = ((payload[start + 1] >> remainder) | (payload[start + 2] << (8 - remainder))) & 0xFF
                real = real_raw - 256 if real_raw >= 128 else real_raw
                imag = imag_raw - 256 if imag_raw >= 128 else imag_raw
                csi[tx, rx, subcarrier] = complex(real, imag)
                bit_index += 16
    return csi


def read_bfee_file(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("rb") as handle:
        while True:
            length_bytes = handle.read(2)
            if len(length_bytes) < 2:
                break
            field_length = int.from_bytes(length_bytes, byteorder="big", signed=False)
            if field_length == 0:
                break
            field = handle.read(field_length)
            if len(field) != field_length or not field or field[0] != 187:
                continue
            num_rx, num_tx = field[9], field[10]
            record = {
                "timestamp_low": int.from_bytes(field[1:5], "little", signed=False),
                "bfee_count": int.from_bytes(field[5:7], "little", signed=False),
                "num_rx": num_rx,
                "num_tx": num_tx,
                "rssi_a": field[11],
                "rssi_b": field[12],
                "rssi_c": field[13],
                "noise": field[14] - 256,
                "agc": field[15],
                "antenna_sel": field[16],
            }
            try:
                record["csi"] = parse_csi(field[21:], num_tx, num_rx)
            except ValueError:
                continue
            records.append(record)
    return records


def scale_csi(record: dict[str, Any]) -> np.ndarray:
    csi = record["csi"].astype(np.complex64)
    csi_power = float(np.sum(np.abs(csi) ** 2))
    rssi_power = _db_inverse(total_rssi(record))
    scale = rssi_power / max(csi_power / 30.0, 1e-12)
    noise_db = -92.0 if record["noise"] == -127 else float(record["noise"])
    thermal_noise = _db_inverse(noise_db)
    quantization_noise = scale * record["num_rx"] * record["num_tx"]
    scaled = csi * math.sqrt(scale / max(thermal_noise + quantization_noise, 1e-12))
    if record["num_tx"] == 2:
        scaled *= math.sqrt(2.0)
    elif record["num_tx"] == 3:
        scaled *= math.sqrt(_db_inverse(4.5))
    return scaled


def read_receiver_file(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    records = read_bfee_file(path)
    if not records:
        raise ValueError(f"No CSI records found in {path}")
    csi_rows: list[np.ndarray] = []
    rssi_rows: list[list[float]] = []
    timestamps: list[int] = []
    expected_streams: int | None = None
    for record in records:
        scaled = scale_csi(record).reshape(-1)
        if expected_streams is None:
            expected_streams = scaled.size
        if scaled.size != expected_streams:
            continue
        csi_rows.append(scaled)
        rssi_rows.append(
            [
                record["rssi_a"] - record["agc"],
                record["rssi_b"] - record["agc"],
                record["rssi_c"] - record["agc"],
                total_rssi(record),
            ]
        )
        timestamps.append(record["timestamp_low"])
    return (
        np.asarray(csi_rows, dtype=np.complex64),
        np.asarray(rssi_rows, dtype=np.float32),
        np.asarray(timestamps, dtype=np.int64),
    )
