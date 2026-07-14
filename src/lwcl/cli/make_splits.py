from __future__ import annotations

import argparse
import json

from lwcl.data.splits import create_group_splits


def main() -> None:
    parser = argparse.ArgumentParser(description="Create leakage-resistant train/validation/test splits")
    parser.add_argument("--input-manifest", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--group-field", choices=["sample", "subject", "environment"], default="subject")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--validation-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=2025)
    args = parser.parse_args()
    counts = create_group_splits(
        args.input_manifest,
        args.output_manifest,
        args.group_field,
        args.train_ratio,
        args.validation_ratio,
        args.seed,
    )
    print(json.dumps(counts, ensure_ascii=False))


if __name__ == "__main__":
    main()
