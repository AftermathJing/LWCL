from __future__ import annotations

import csv
import json
from pathlib import Path

import yaml

from lwcl_v2.cli.audit_subject_domains import build_audit
from lwcl_v2.cli.build_protocol_splits import build_protocol_splits
from lwcl_v2.cli.summarize_multiseed import summarize_runs


def _write_manifest(path: Path, subjects: list[str]) -> None:
    rows = []
    for subject_index, subject in enumerate(subjects):
        for label in range(6):
            for position in range(1, 4):
                sample_id = (
                    f"2018113{position}_{subject}_g{label + 1}_p{position}_"
                    f"o{(subject_index % 3) + 1}_i1"
                )
                rows.append(
                    {
                        "sample_id": sample_id,
                        "feature_path": f"features/{sample_id}.npz",
                        "subject": subject,
                        "label": str(label),
                        "environment": f"env{position}",
                        "position": str(position),
                        "orientation": str((subject_index % 3) + 1),
                        "time_steps": str(25 + subject_index),
                        "valid_receivers": "6",
                    }
                )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_metrics(path: Path, accuracy: float, macro_f1: float, subject: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "accuracy": accuracy,
                "macro_f1": macro_f1,
                "worst_subject_macro_f1": macro_f1 - 0.1,
                "selection_score": macro_f1 - 0.03,
                "loss": 0.5,
                "per_class_f1": [macro_f1] * 6,
                "subjects": {
                    subject: {"count": 10, "accuracy": accuracy, "macro_f1": macro_f1}
                },
            }
        ),
        encoding="utf-8",
    )


def test_multiseed_summary_reports_mean_std_and_subject_metrics(tmp_path: Path):
    runs = []
    for seed, score in ((2026, 0.8), (2027, 0.9)):
        run = tmp_path / f"seed_{seed}"
        (run / "train").mkdir(parents=True)
        (run / "train" / "resolved_config.yaml").write_text(
            yaml.safe_dump({"training": {"seed": seed}}), encoding="utf-8"
        )
        _write_metrics(run / "eval" / "validation_metrics.json", score, score, "user17")
        _write_metrics(run / "eval" / "test_metrics.json", score, score, "user17")
        runs.append(run)
    report = summarize_runs(runs)
    assert report["seeds"] == [2026, 2027]
    assert abs(report["test"]["macro_f1"]["mean"] - 0.85) < 1e-12
    assert report["test"]["macro_f1"]["std"] > 0
    assert report["test"]["per_subject"]["user17"]["runs"] == 2


def test_subject_domain_audit_derives_date_and_focus_subject(tmp_path: Path):
    manifest = tmp_path / "manifest.csv"
    _write_manifest(manifest, ["user1", "user17"])
    report = build_audit(manifest, focus_subject="user17", load_features=False)
    assert report["rows"] == 36
    assert report["focus_subject"]["subject"] == "user17"
    assert report["subject_summaries"]["user17"]["time_steps"]["median"] == 26.0
    assert set(report["cross_tables"]["position"]["user17"]) == {"1", "2", "3"}


def test_cross_location_and_cross_subject_protocols_are_disjoint(tmp_path: Path):
    manifest = tmp_path / "manifest.csv"
    _write_manifest(manifest, [f"user{index}" for index in range(1, 7)])
    cl = build_protocol_splits(manifest, tmp_path / "splits", "cl")
    assert len(cl) == 3
    for report in cl.values():
        domains = report["domains"]["position"]
        assert not (set(domains["test"]) & set(domains["train"]))
        assert not (set(domains["validation"]) & set(domains["train"]))

    cs = build_protocol_splits(
        manifest,
        tmp_path / "splits",
        "cs",
        test_subjects_per_fold=2,
        validation_subjects=1,
    )
    tested_subjects = []
    for report in cs.values():
        assert not any(report["subject_overlap"].values())
        tested_subjects.extend(report["subjects"]["test"])
    assert sorted(tested_subjects) == [f"user{index}" for index in range(1, 7)]
