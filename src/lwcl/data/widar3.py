from __future__ import annotations

import csv
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable


WIDAR3_GESTURE_NAMES = {
    1: "push_pull",
    2: "sweep",
    3: "clap",
    4: "slide",
    5: "draw_circle",
    6: "draw_zigzag",
}

_FILE_PATTERN = re.compile(
    r"^(?P<subject>user\d+)-(?P<gesture>\d+)-(?P<position>\d+)-"
    r"(?P<orientation>\d+)-(?P<instance>\d+)-r(?P<receiver>\d+)\.dat$"
)


def build_widar3_raw_manifest(
    dataset_root: str | Path,
    output_manifest: str | Path,
    gesture_ids: Iterable[int] = range(1, 7),
    num_receivers: int = 6,
    strict: bool = True,
) -> dict[str, int]:
    """Group receiver files into action samples and write a raw manifest.

    Widar3 files use ``user-gesture-position-orientation-instance-rx.dat``.
    The thesis experiment uses gesture IDs 1-6; IDs 7-9 remain available in
    the raw archive but are excluded by the default paper manifest.
    """

    dataset_root = Path(dataset_root).resolve()
    output_manifest = Path(output_manifest).resolve()
    selected_gestures = {int(value) for value in gesture_ids}
    grouped: dict[Path, dict[int, Path]] = defaultdict(dict)
    malformed = 0

    for path in dataset_root.rglob("*.dat"):
        match = _FILE_PATTERN.match(path.name)
        if match is None:
            malformed += 1
            continue
        gesture = int(match.group("gesture"))
        if gesture not in selected_gestures:
            continue
        receiver = int(match.group("receiver"))
        prefix = path.with_name(path.name.rsplit("-r", 1)[0])
        grouped[prefix][receiver] = path

    expected_receivers = set(range(1, num_receivers + 1))
    rows: list[dict[str, str | int]] = []
    incomplete = 0
    for prefix, receivers in sorted(grouped.items(), key=lambda item: str(item[0])):
        match = _FILE_PATTERN.match(receivers[min(receivers)].name)
        assert match is not None
        if set(receivers) != expected_receivers:
            incomplete += 1
            if strict:
                raise ValueError(
                    f"Incomplete receiver group {prefix}: expected {sorted(expected_receivers)}, "
                    f"found {sorted(receivers)}"
                )
            continue
        gesture = int(match.group("gesture"))
        environment = prefix.parent.parent.name
        subject = match.group("subject")
        position = match.group("position")
        orientation = match.group("orientation")
        instance = match.group("instance")
        sample_id = f"{environment}_{subject}_g{gesture}_p{position}_o{orientation}_i{instance}"
        rows.append(
            {
                "sample_id": sample_id,
                "raw_path": Path(os.path.relpath(prefix, output_manifest.parent)).as_posix(),
                "label": gesture - 1,
                "gesture_id": gesture,
                "gesture_name": WIDAR3_GESTURE_NAMES.get(gesture, f"gesture_{gesture}"),
                "subject": subject,
                "environment": environment,
                "position": position,
                "orientation": orientation,
                "instance": instance,
                "receiver_count": num_receivers,
            }
        )

    if not rows:
        raise RuntimeError(f"No complete Widar3 samples found in {dataset_root}")
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with output_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return {
        "samples": len(rows),
        "incomplete_groups": incomplete,
        "malformed_files": malformed,
    }
