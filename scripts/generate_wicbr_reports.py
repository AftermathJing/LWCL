from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    with path.open('r', encoding='utf-8') as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open('r', encoding='utf-8') as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open('r', encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def pct(value: float) -> str:
    return f'{value * 100:.2f}%'


def metric_table(metrics: dict[str, Any]) -> str:
    loss = metrics.get('loss')
    loss_str = f'{float(loss):.4f}' if loss is not None else 'n/a'
    return '\n'.join([
        '| Metric | Value |',
        '| --- | ---: |',
        f"| accuracy | {pct(float(metrics['accuracy']))} |",
        f"| macro precision | {pct(float(metrics['macro_precision']))} |",
        f"| macro recall | {pct(float(metrics['macro_recall']))} |",
        f"| macro F1 | {pct(float(metrics['macro_f1']))} |",
        f'| loss | {loss_str} |',
    ])


def per_class_table(metrics: dict[str, Any], gesture_names: list[str]) -> str:
    precision = metrics['per_class_precision']
    recall = metrics['per_class_recall']
    f1 = metrics['per_class_f1']
    support = metrics['support']
    lines = [
        '| Label | Gesture | Precision | Recall | F1 | Support |',
        '| ---: | --- | ---: | ---: | ---: | ---: |',
    ]
    for index, name in enumerate(gesture_names):
        lines.append(
            f"| {index} | {name} | {pct(float(precision[index]))} | {pct(float(recall[index]))} | {pct(float(f1[index]))} | {int(support[index])} |"
        )
    return '\n'.join(lines)


def confusion_table(metrics: dict[str, Any], gesture_names: list[str]) -> str:
    matrix = metrics['confusion_matrix']
    header = '| Actual / Predicted | ' + ' | '.join(gesture_names) + ' |'
    separator = '| --- | ' + ' | '.join(['---:'] * len(gesture_names)) + ' |'
    lines = [header, separator]
    for name, row in zip(gesture_names, matrix, strict=True):
        lines.append('| ' + name + ' | ' + ' | '.join(str(int(value)) for value in row) + ' |')
    return '\n'.join(lines)


def group_accuracy_table(group_accuracy: dict[str, Any]) -> str:
    lines = ['| Group | Accuracy | Count |', '| --- | ---: | ---: |']
    for field, groups in group_accuracy.items():
        for group, payload in groups.items():
            lines.append(f"| {field} {group} | {pct(float(payload['accuracy']))} | {int(payload['count'])} |")
    return '\n'.join(lines)


def build_subject_report(subject_output: Path, test_output: Path, manifest_path: Path, report_path: Path) -> None:
    run_info = load_json(subject_output / 'run_info.json')
    metrics = load_json(test_output / 'test_metrics.json')
    train_events = load_jsonl(subject_output / 'metrics.jsonl')
    manifest_rows = load_manifest(manifest_path)
    split_counts = Counter(row['split'] for row in manifest_rows)
    split_subjects: dict[str, list[str]] = {}
    for split in ('train', 'validation', 'test'):
        split_subjects[split] = sorted({row.get('subject', '') for row in manifest_rows if row.get('split') == split})
    labels: dict[str, str] = {}
    for row in manifest_rows:
        labels[row['label']] = row.get('gesture_name', row['label'])
    num_labels = len(metrics['support'])
    gesture_names = [labels.get(str(index), f'class_{index}') for index in range(num_labels)]
    eval_events = [row for row in train_events if row.get('event') == 'evaluation']
    best_event = max(eval_events, key=lambda row: float(row.get('macro_f1', -1.0))) if eval_events else None
    best_section = metric_table(best_event) if best_event is not None else 'No validation event found.'
    text = '\n'.join([
        '# Wi-CBR Widar3 Subject-Disjoint Report',
        '',
        f"- Remote host: {run_info.get('platform', 'unknown')}",
        f"- Git branch: {run_info.get('git_branch')}",
        f"- Git commit: {run_info.get('git_commit')}",
        f"- Manifest: {run_info.get('manifest')}",
        '',
        '## 1. Objective',
        '',
        'Reproduce Wi-CBR on the LWCL Widar3 subject-disjoint protocol using raw-CSI-derived phase and DFS images.',
        '',
        '## 2. Data split',
        '',
        '| Split | Samples | Subjects |',
        '| --- | ---: | --- |',
        f"| train | {split_counts.get('train', 0)} | {', '.join(split_subjects['train'])} |",
        f"| validation | {split_counts.get('validation', 0)} | {', '.join(split_subjects['validation'])} |",
        f"| test | {split_counts.get('test', 0)} | {', '.join(split_subjects['test'])} |",
        '',
        '## 3. Best validation checkpoint',
        '',
        best_section,
        '',
        '## 4. Test result',
        '',
        metric_table(metrics),
        '',
        '## 5. Per-class metrics',
        '',
        per_class_table(metrics, gesture_names),
        '',
        '## 6. Confusion matrix',
        '',
        confusion_table(metrics, gesture_names),
        '',
        '## 7. Group accuracy',
        '',
        group_accuracy_table(metrics.get('group_accuracy', {})),
        '',
    ])
    report_path.write_text(text, encoding='utf-8')


def build_cross_report(cross_root: Path, report_path: Path) -> None:
    fold_dirs = sorted(path for path in cross_root.iterdir() if path.is_dir())
    rows: list[dict[str, Any]] = []
    weighted_acc_sum = 0.0
    weighted_f1_sum = 0.0
    weighted_total = 0
    for fold_dir in fold_dirs:
        metrics_path = fold_dir / 'test_eval' / 'test_metrics.json'
        if not metrics_path.exists():
            continue
        metrics = load_json(metrics_path)
        support = sum(int(value) for value in metrics['support'])
        rows.append({
            'target': fold_dir.name,
            'accuracy': float(metrics['accuracy']),
            'macro_f1': float(metrics['macro_f1']),
            'support': support,
        })
        weighted_acc_sum += float(metrics['accuracy']) * support
        weighted_f1_sum += float(metrics['macro_f1']) * support
        weighted_total += support
    equal_acc = sum(row['accuracy'] for row in rows) / max(len(rows), 1)
    equal_f1 = sum(row['macro_f1'] for row in rows) / max(len(rows), 1)
    pooled_acc = weighted_acc_sum / max(weighted_total, 1)
    pooled_f1 = weighted_f1_sum / max(weighted_total, 1)
    lines = [
        '# Wi-CBR Widar3 Cross-Environment Report',
        '',
        '## 1. Fold results',
        '',
        '| Target | Accuracy | Macro-F1 | Support |',
        '| --- | ---: | ---: | ---: |',
    ]
    for row in rows:
        lines.append(f"| {row['target']} | {pct(row['accuracy'])} | {pct(row['macro_f1'])} | {row['support']} |")
    lines.extend([
        '',
        '## 2. Aggregate',
        '',
        '| Aggregate | Accuracy | Macro-F1 |',
        '| --- | ---: | ---: |',
        f"| Equal-fold mean | {pct(equal_acc)} | {pct(equal_f1)} |",
        f"| Pooled test samples | {pct(pooled_acc)} | {pct(pooled_f1)} |",
        '',
    ])
    report_path.write_text('\n'.join(lines), encoding='utf-8')


def main() -> None:
    parser = argparse.ArgumentParser(description='Generate Wi-CBR markdown reports from finished experiment outputs')
    parser.add_argument('--subject-output')
    parser.add_argument('--subject-test-output')
    parser.add_argument('--subject-manifest')
    parser.add_argument('--subject-report')
    parser.add_argument('--cross-root')
    parser.add_argument('--cross-report')
    args = parser.parse_args()

    if args.subject_output and args.subject_test_output and args.subject_manifest and args.subject_report:
        build_subject_report(
            Path(args.subject_output),
            Path(args.subject_test_output),
            Path(args.subject_manifest),
            Path(args.subject_report),
        )
    if args.cross_root and args.cross_report:
        build_cross_report(Path(args.cross_root), Path(args.cross_report))


if __name__ == '__main__':
    main()
