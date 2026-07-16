from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable


OFFICIAL_SUBJECT_ID = {
    ("20181130", "user5"): 0,
    ("20181130", "user10"): 1,
    ("20181130", "user11"): 2,
    ("20181130", "user12"): 3,
    ("20181130", "user13"): 4,
    ("20181130", "user14"): 5,
    ("20181130", "user15"): 6,
    ("20181130", "user16"): 7,
    ("20181130", "user17"): 8,
    ("20181209", "user2"): 9,
    ("20181205", "user2"): 9,
    ("20181208", "user2"): 9,
    ("20181205", "user3"): 10,
    ("20181208", "user3"): 10,
    ("20181211", "user3"): 11,
    ("20181211", "user7"): 12,
    ("20181211", "user8"): 13,
    ("20181211", "user9"): 14,
    ("20181209", "user6"): 15,
    ("20181204", "user1"): 16,
}

OFFICIAL_ENV_GROUPS = {
    "env1": {0, 1, 2, 3, 4, 5, 6, 7, 8},
    "env2": {9, 10, 15, 16},
    "env3": {11, 12, 13, 14},
}

OFFICIAL_CR_SPLITS = {
    "cr1": {"train": OFFICIAL_ENV_GROUPS["env2"] | OFFICIAL_ENV_GROUPS["env3"], "test": OFFICIAL_ENV_GROUPS["env1"]},
    "cr2": {"train": OFFICIAL_ENV_GROUPS["env1"] | OFFICIAL_ENV_GROUPS["env3"], "test": OFFICIAL_ENV_GROUPS["env2"]},
    "cr3": {"train": OFFICIAL_ENV_GROUPS["env1"] | OFFICIAL_ENV_GROUPS["env2"], "test": OFFICIAL_ENV_GROUPS["env3"]},
}

OFFICIAL_ENV1_IDS = OFFICIAL_ENV_GROUPS["env1"]


def _read_rows(manifest_path: Path) -> list[dict[str, str]]:
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _with_official_fields(rows: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    output: list[dict[str, str]] = []
    for row in rows:
        key = (row.get("environment", ""), row.get("subject", ""))
        official_id = OFFICIAL_SUBJECT_ID.get(key)
        if official_id is None:
            continue
        row = dict(row)
        row["official_subject_id"] = str(official_id)
        row["official_env_group"] = next(name for name, ids in OFFICIAL_ENV_GROUPS.items() if official_id in ids)
        output.append(row)
    return output


def build_widar3_official_wicbr_manifests(
    input_manifest: str | Path,
    output_dir: str | Path,
) -> dict[str, object]:
    input_manifest = Path(input_manifest).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = _with_official_fields(_read_rows(input_manifest))
    available_pairs = sorted({(row["environment"], row["subject"]) for row in rows})
    expected_pairs = sorted(OFFICIAL_SUBJECT_ID.keys())
    missing_pairs = [pair for pair in expected_pairs if pair not in available_pairs]

    summaries: dict[str, object] = {
        "input_manifest": str(input_manifest),
        "available_pairs": [list(pair) for pair in available_pairs],
        "missing_expected_pairs": [list(pair) for pair in missing_pairs],
    }

    def write_manifest(name: str, manifest_rows: list[dict[str, str]]) -> None:
        path = output_dir / f"{name}.csv"
        fieldnames = list(manifest_rows[0].keys())
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(manifest_rows)
        counts: dict[str, int] = {}
        for row in manifest_rows:
            split = row["split"]
            counts[split] = counts.get(split, 0) + 1
        summary = {
            "manifest": str(path),
            "counts": counts,
        }
        (output_dir / f"{name}.summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        summaries[name] = summary

    env1_rows = [row for row in rows if int(row["official_subject_id"]) in OFFICIAL_ENV1_IDS]

    protocol_rows: dict[str, list[dict[str, str]]] = {key: [] for key in ("indom", "cl", "co", "cr1", "cr2", "cr3")}
    for row in env1_rows:
        indom = dict(row)
        indom["split"] = "train" if int(row["instance"]) in {1, 2, 3, 4} else "test"
        protocol_rows["indom"].append(indom)

        cl = dict(row)
        cl["split"] = "train" if int(row["position"]) in {1, 2, 3, 4} else "test"
        protocol_rows["cl"].append(cl)

        co = dict(row)
        co["split"] = "train" if int(row["orientation"]) in {1, 2, 3, 4} else "test"
        protocol_rows["co"].append(co)

    for protocol, assignment in OFFICIAL_CR_SPLITS.items():
        for row in rows:
            official_id = int(row["official_subject_id"])
            if official_id in assignment["train"]:
                split = "train"
            elif official_id in assignment["test"]:
                split = "test"
            else:
                continue
            entry = dict(row)
            entry["split"] = split
            protocol_rows[protocol].append(entry)

    for name, manifest_rows in protocol_rows.items():
        if not manifest_rows:
            continue
        write_manifest(name, manifest_rows)

    (output_dir / "summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    return summaries
