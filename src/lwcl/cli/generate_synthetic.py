from __future__ import annotations

import argparse

from lwcl.data.synthetic import generate_synthetic_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate processed synthetic LWCL data for smoke tests")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--samples-per-label", type=int, default=12)
    parser.add_argument("--seed", type=int, default=2025)
    args = parser.parse_args()
    manifest = generate_synthetic_dataset(args.output_dir, args.samples_per_label, seed=args.seed)
    print(manifest)


if __name__ == "__main__":
    main()
