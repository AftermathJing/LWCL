from __future__ import annotations

import csv
import json
from pathlib import Path

import yaml
import numpy as np

from lwcl_v2.cli.analyze_subject_errors import analyze_subject
from lwcl_v2.cli.build_wicbr_protocol_manifests import build_protocol_manifest
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


def test_subject_error_analysis_reports_groups_calibration_and_distances(tmp_path: Path):
    outputs = tmp_path / "validation_outputs.npz"
    reference = tmp_path / "train_outputs.npz"
    labels = np.asarray([0, 1, 0, 1])
    probabilities = np.asarray(
        [[0.9, 0.1], [0.2, 0.8], [0.4, 0.6], [0.7, 0.3]], dtype=np.float32
    )
    np.savez_compressed(
        outputs,
        sample_ids=np.asarray(["a", "b", "c", "d"]),
        subjects=np.asarray(["user17"] * 4),
        labels=labels,
        predictions=probabilities.argmax(axis=1),
        probabilities=probabilities,
        embeddings=np.asarray([[1, 0], [0, 1], [1, 0.1], [0.1, 1]], dtype=np.float32),
        sequence_lengths=np.asarray([20, 25, 40, 45]),
        valid_receivers=np.asarray([6, 6, 5, 6]),
        receiver_quality_mean=np.ones((4, 5), dtype=np.float32),
        environments=np.asarray(["e1"] * 4),
        positions=np.asarray(["1", "1", "2", "2"]),
        orientations=np.asarray(["1", "2", "1", "2"]),
    )
    np.savez_compressed(
        reference,
        subjects=np.asarray(["user1", "user1", "user2", "user2"]),
        labels=labels,
        embeddings=np.asarray([[1, 0], [0, 1], [0.8, 0.2], [0.2, 0.8]], dtype=np.float32),
    )
    report = analyze_subject(outputs, "user17", reference, num_labels=2)
    assert report["count"] == 4
    assert set(report["by_position"]) == {"1", "2"}
    assert report["calibration"]["nll"] > 0
    assert set(report["distance_to_reference_subjects"]) == {"user1", "user2"}


def test_wicbr_manifest_adapter_duplicates_official_test_as_validation(tmp_path: Path):
    processed_manifest = tmp_path / "processed.csv"
    official_manifest = tmp_path / "cr1.csv"
    feature_root = tmp_path / "features"
    feature_root.mkdir()
    for sample_id in ("s1", "s2"):
        (feature_root / f"{sample_id}.npz").write_bytes(b"test")
    with processed_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["sample_id", "feature_path", "subject", "label", "feature_format"],
        )
        writer.writeheader()
        writer.writerows(
            [
                {
                    "sample_id": "s1",
                    "feature_path": str((feature_root / "s1.npz").resolve()),
                    "subject": "u1",
                    "label": "0",
                    "feature_format": "npz-v2",
                },
                {
                    "sample_id": "s2",
                    "feature_path": str((feature_root / "s2.npz").resolve()),
                    "subject": "u2",
                    "label": "1",
                    "feature_format": "npz-v2",
                },
            ]
        )
    with official_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["sample_id", "split", "environment", "subject", "label"],
        )
        writer.writeheader()
        writer.writerows(
            [
                {"sample_id": "s1", "split": "train", "environment": "env1", "subject": "u1", "label": "0"},
                {"sample_id": "s2", "split": "test", "environment": "env2", "subject": "u2", "label": "1"},
            ]
        )
    output_manifest = tmp_path / "adapted" / "manifest.csv"
    summary = build_protocol_manifest(processed_manifest, official_manifest, output_manifest)
    rows = list(csv.DictReader(output_manifest.open("r", encoding="utf-8-sig", newline="")))
    assert summary["counts"] == {"train": 1, "validation": 1, "test": 1}
    assert sorted(row["split"] for row in rows) == ["test", "train", "validation"]


def test_wicbr_manifest_adapter_uses_stratified_source_validation(tmp_path: Path):
    processed_manifest = tmp_path / "processed.csv"
    official_manifest = tmp_path / "cr1.csv"
    feature_root = tmp_path / "features"
    feature_root.mkdir()
    processed_rows = []
    official_rows = []
    for environment in ("source_a", "source_b"):
        for label in ("0", "1"):
            for index in range(8):
                sample_id = f"{environment}_{label}_{index}"
                processed_rows.append(
                    {
                        "sample_id": sample_id,
                        "feature_path": str((feature_root / f"{sample_id}.npz").resolve()),
                        "subject": f"u{index}",
                        "label": label,
                        "feature_format": "npz-v2",
                    }
                )
                official_rows.append(
                    {
                        "sample_id": sample_id,
                        "split": "train",
                        "environment": environment,
                        "subject": f"u{index}",
                        "label": label,
                    }
                )
    for index in range(4):
        sample_id = f"target_{index}"
        label = str(index % 2)
        processed_rows.append(
            {
                "sample_id": sample_id,
                "feature_path": str((feature_root / f"{sample_id}.npz").resolve()),
                "subject": f"target_u{index}",
                "label": label,
                "feature_format": "npz-v2",
            }
        )
        official_rows.append(
            {
                "sample_id": sample_id,
                "split": "test",
                "environment": "target",
                "subject": f"target_u{index}",
                "label": label,
            }
        )

    for path, rows, fieldnames in (
        (
            processed_manifest,
            processed_rows,
            ["sample_id", "feature_path", "subject", "label", "feature_format"],
        ),
        (
            official_manifest,
            official_rows,
            ["sample_id", "split", "environment", "subject", "label"],
        ),
    ):
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    output_manifest = tmp_path / "adapted" / "manifest.csv"
    summary = build_protocol_manifest(
        processed_manifest,
        official_manifest,
        output_manifest,
        duplicate_test_as_validation=False,
        source_validation_fraction=0.25,
        source_validation_salt="test-salt",
    )
    rows = list(csv.DictReader(output_manifest.open("r", encoding="utf-8-sig", newline="")))
    split_ids = {
        split: {row["sample_id"] for row in rows if row["split"] == split}
        for split in ("train", "validation", "test")
    }
    official_train_ids = {
        row["sample_id"] for row in official_rows if row["split"] == "train"
    }
    official_test_ids = {
        row["sample_id"] for row in official_rows if row["split"] == "test"
    }
    assert summary["validation_source"] == "source_train_stratified"
    assert summary["counts"] == {"train": 24, "validation": 8, "test": 4}
    assert split_ids["validation"] <= official_train_ids
    assert split_ids["test"] == official_test_ids
    assert split_ids["train"].isdisjoint(split_ids["validation"])
    assert split_ids["validation"].isdisjoint(split_ids["test"])
    assert {
        (row["environment"], row["label"])
        for row in rows
        if row["split"] == "validation"
    } == {
        ("source_a", "0"),
        ("source_a", "1"),
        ("source_b", "0"),
        ("source_b", "1"),
    }
