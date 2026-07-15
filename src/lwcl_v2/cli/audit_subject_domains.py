from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from lwcl_v2.data.quality import QUALITY_FIELDS


SAMPLE_PATTERN = re.compile(
    r"(?P<date>\d+)_(?P<subject>user\d+)_g(?P<gesture>\d+)_p(?P<position>\d+)_o(?P<orientation>\d+)_i(?P<instance>\d+)"
)
DOMAIN_FIELDS = ("collection_date", "environment", "position", "orientation", "split", "label")


def _numeric_summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "median": None,
            "p05": None,
            "p95": None,
            "min": None,
            "max": None,
        }
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "median": float(np.median(array)),
        "p05": float(np.percentile(array, 5)),
        "p95": float(np.percentile(array, 95)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _derived_fields(row: dict[str, str]) -> dict[str, str]:
    match = SAMPLE_PATTERN.fullmatch(row.get("sample_id", ""))
    parsed = match.groupdict() if match else {}
    return {
        "collection_date": row.get("collection_date") or row.get("date") or parsed.get("date", "unknown"),
        "environment": row.get("environment") or row.get("collection_date") or row.get("date") or parsed.get("date", "unknown"),
        "position": row.get("position") or parsed.get("position", "unknown"),
        "orientation": row.get("orientation") or parsed.get("orientation", "unknown"),
        "split": row.get("split") or "unspecified",
        "label": row.get("label") or parsed.get("gesture", "unknown"),
    }


def _feature_path(manifest: Path, row: dict[str, str]) -> Path | None:
    value = row.get("feature_path")
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else manifest.parent / path


def build_audit(
    manifest: Path,
    focus_subject: str | None = None,
    load_features: bool = True,
) -> dict[str, Any]:
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    subject_counts: dict[str, Counter[str]] = defaultdict(Counter)
    cross_tables: dict[str, dict[str, Counter[str]]] = {
        field: defaultdict(Counter) for field in DOMAIN_FIELDS
    }
    time_steps: dict[str, list[float]] = defaultdict(list)
    valid_receivers: dict[str, list[float]] = defaultdict(list)
    quality_values: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {field: [] for field in QUALITY_FIELDS}
    )
    quality_receiver_counts: dict[str, Counter[str]] = defaultdict(Counter)
    feature_errors = []

    for row in rows:
        subject = row["subject"]
        derived = _derived_fields(row)
        subject_counts[subject]["samples"] += 1
        for field, value in derived.items():
            cross_tables[field][subject][value] += 1

        row_time_steps = row.get("time_steps")
        row_valid_receivers = row.get("valid_receivers")
        if row_time_steps:
            time_steps[subject].append(float(row_time_steps))
        if row_valid_receivers:
            valid_receivers[subject].append(float(row_valid_receivers))
            quality_receiver_counts[subject][str(int(float(row_valid_receivers)))] += 1

        if not load_features:
            continue
        path = _feature_path(manifest, row)
        if path is None or not path.exists():
            feature_errors.append(
                {"sample_id": row.get("sample_id", ""), "error": f"missing feature path: {path}"}
            )
            continue
        try:
            with np.load(path, allow_pickle=False) as archive:
                time_mask = archive["time_mask"].astype(bool)
                receiver_mask = archive["receiver_mask"].astype(bool)
                quality = archive["receiver_quality"].astype(np.float64)
            if not row_time_steps:
                time_steps[subject].append(float(time_mask.sum()))
            if not row_valid_receivers:
                valid_receivers[subject].append(float(receiver_mask.sum()))
            if not row_valid_receivers:
                quality_receiver_counts[subject][str(int(receiver_mask.sum()))] += 1
            valid_quality = quality[receiver_mask]
            for index, field in enumerate(QUALITY_FIELDS):
                quality_values[subject][field].extend(valid_quality[:, index].tolist())
        except Exception as error:
            feature_errors.append(
                {"sample_id": row.get("sample_id", ""), "error": f"{type(error).__name__}: {error}"}
            )

    subjects = sorted(subject_counts)
    subject_summaries: dict[str, Any] = {}
    for subject in subjects:
        summary = {
            "samples": subject_counts[subject]["samples"],
            "domains": {
                field: dict(sorted(cross_tables[field][subject].items())) for field in DOMAIN_FIELDS
            },
            "time_steps": _numeric_summary(time_steps[subject]),
            "valid_receivers": {
                "summary": _numeric_summary(valid_receivers[subject]),
                "sample_counts": dict(sorted(quality_receiver_counts[subject].items())),
            },
            "receiver_quality": {
                field: _numeric_summary(quality_values[subject][field]) for field in QUALITY_FIELDS
            },
        }
        subject_summaries[subject] = summary

    rendered_cross_tables = {
        field: {
            subject: dict(sorted(counts.items()))
            for subject, counts in sorted(table.items())
        }
        for field, table in cross_tables.items()
    }
    value_subjects = {
        field: {
            value: sorted(subject for subject in subjects if cross_tables[field][subject][value])
            for value in sorted({value for counts in cross_tables[field].values() for value in counts})
        }
        for field in DOMAIN_FIELDS
    }
    report: dict[str, Any] = {
        "manifest": str(manifest.resolve()),
        "rows": len(rows),
        "subjects": subjects,
        "cross_tables": rendered_cross_tables,
        "value_subjects": value_subjects,
        "subject_summaries": subject_summaries,
        "feature_audit": {
            "enabled": load_features,
            "error_count": len(feature_errors),
            "errors": feature_errors[:50],
        },
    }
    if focus_subject:
        if focus_subject not in subject_summaries:
            raise ValueError(f"Focus subject {focus_subject!r} is absent from the manifest")
        report["focus_subject"] = {
            "subject": focus_subject,
            "summary": subject_summaries[focus_subject],
            "domain_value_peers": {
                field: {
                    value: value_subjects[field][value]
                    for value in cross_tables[field][focus_subject]
                }
                for field in DOMAIN_FIELDS
            },
        }
    return report


def _write_cross_tables(output_dir: Path, report: dict[str, Any]) -> None:
    for field, table in report["cross_tables"].items():
        values = sorted({value for counts in table.values() for value in counts})
        with (output_dir / f"subject_x_{field}.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=["subject", *values])
            writer.writeheader()
            for subject, counts in table.items():
                writer.writerow({"subject": subject, **{value: counts.get(value, 0) for value in values}})


def _write_subject_summary(output_dir: Path, report: dict[str, Any]) -> None:
    rows = []
    for subject, summary in report["subject_summaries"].items():
        row: dict[str, Any] = {
            "subject": subject,
            "samples": summary["samples"],
            "time_steps_median": summary["time_steps"]["median"],
            "time_steps_p05": summary["time_steps"]["p05"],
            "time_steps_p95": summary["time_steps"]["p95"],
            "valid_receivers_mean": summary["valid_receivers"]["summary"]["mean"],
            "valid_receivers_min": summary["valid_receivers"]["summary"]["min"],
        }
        for field in QUALITY_FIELDS:
            row[f"quality_{field}_median"] = summary["receiver_quality"][field]["median"]
        rows.append(row)
    if not rows:
        return
    with (output_dir / "subject_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit subject/domain confounding in Widar3")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--focus-subject", default="user17")
    parser.add_argument("--skip-features", action="store_true")
    parser.add_argument("--strict-features", action="store_true")
    args = parser.parse_args()
    manifest = Path(args.manifest)
    report = build_audit(
        manifest,
        focus_subject=args.focus_subject or None,
        load_features=not args.skip_features,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "subject_domain_audit.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    _write_cross_tables(output_dir, report)
    _write_subject_summary(output_dir, report)
    print(json.dumps({
        "rows": report["rows"],
        "subjects": len(report["subjects"]),
        "focus_subject": args.focus_subject,
        "feature_errors": report["feature_audit"]["error_count"],
        "output_dir": str(output_dir.resolve()),
    }, ensure_ascii=False))
    if args.strict_features and report["feature_audit"]["error_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
