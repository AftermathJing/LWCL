from __future__ import annotations

import argparse
import json

from lwcl.data.csi_bench import build_csi_bench_manifests, discover_csi_bench_tasks


def main() -> None:
    parser = argparse.ArgumentParser(description="Build model-ready CSI-Bench manifests from official H5 metadata")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tasks", default="all", help="Comma-separated task names or 'all'")
    parser.add_argument("--target-time", type=int, default=500)
    parser.add_argument("--target-receivers", type=int, default=0, help="0 means infer per task")
    parser.add_argument("--target-features", type=int, default=0, help="0 means infer per task/device")
    args = parser.parse_args()
    tasks = None
    if args.tasks.lower() != "all":
        tasks = [value.strip() for value in args.tasks.split(",") if value.strip()]
    else:
        tasks = list(discover_csi_bench_tasks(args.dataset_root))
    summary = build_csi_bench_manifests(
        args.dataset_root,
        args.output_dir,
        tasks=tasks,
        target_time=args.target_time,
        target_receivers=args.target_receivers or None,
        target_features=args.target_features or None,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
