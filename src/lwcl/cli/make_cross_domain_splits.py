from __future__ import annotations

import argparse
import json

from lwcl.data.splits import create_leave_one_domain_out_splits


def main() -> None:
    parser = argparse.ArgumentParser(description="Build leave-one-domain-out manifests")
    parser.add_argument("--input-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--domain-field", default="environment")
    parser.add_argument("--validation-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=2025)
    args = parser.parse_args()
    summary = create_leave_one_domain_out_splits(
        input_manifest=args.input_manifest,
        output_dir=args.output_dir,
        domain_field=args.domain_field,
        validation_ratio=args.validation_ratio,
        seed=args.seed,
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
