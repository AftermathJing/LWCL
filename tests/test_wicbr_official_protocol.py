from __future__ import annotations

import csv
from pathlib import Path

from lwcl.wicbr.official_protocol import build_widar3_official_wicbr_manifests


def test_build_widar3_official_wicbr_manifests(tmp_path: Path) -> None:
    rows: list[dict[str, str]] = []
    def add_rows(environment: str, subject: str, positions=(1, 2, 3, 4, 5), orientations=(1, 2, 3, 4, 5), instances=(1, 2, 3, 4, 5)):
        for gesture in range(1, 7):
            for position in positions:
                for orientation in orientations:
                    for instance in instances:
                        sample_id = f"{environment}_{subject}_{gesture}_{position}_{orientation}_{instance}"
                        rows.append(
                            {
                                "sample_id": sample_id,
                                "label": str(gesture - 1),
                                "subject": subject,
                                "environment": environment,
                                "position": str(position),
                                "orientation": str(orientation),
                                "instance": str(instance),
                                "phase_path": f"phase/{sample_id}.jpg",
                                "dfs_path": f"dfs/{sample_id}.jpg",
                            }
                        )

    for subject in ("user5", "user10", "user11", "user12", "user13", "user14", "user15", "user16", "user17"):
        add_rows("20181130", subject)
    add_rows("20181204", "user1")
    add_rows("20181209", "user2", positions=(1,), orientations=(1,), instances=(1,))
    add_rows("20181209", "user6")
    for subject in ("user3", "user7", "user8", "user9"):
        add_rows("20181211", subject)

    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = build_widar3_official_wicbr_manifests(manifest, tmp_path / "official")
    assert (tmp_path / "official" / "cr1.csv").exists()
    assert (tmp_path / "official" / "cr2.csv").exists()
    assert (tmp_path / "official" / "cr3.csv").exists()
    assert (tmp_path / "official" / "indom.csv").exists()
    assert summary["cr1"]["counts"]["test"] == 6750
    assert summary["indom"]["counts"]["test"] == 1350
