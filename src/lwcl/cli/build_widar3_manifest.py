from __future__ import annotations

import argparse
import json

from lwcl.data.widar3 import build_widar3_raw_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an action-group manifest from Widar3 receiver files")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--gesture-ids", default="1,2,3,4,5,6")
    parser.add_argument("--num-receivers", type=int, default=6)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    gesture_ids = [int(value) for value in args.gesture_ids.split(",") if value.strip()]
    summary = build_widar3_raw_manifest(
        args.dataset_root,
        args.output_manifest,
        gesture_ids=gesture_ids,
        num_receivers=args.num_receivers,
        strict=not args.allow_incomplete,
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
