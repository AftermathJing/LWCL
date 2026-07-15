from __future__ import annotations

import argparse
import json

from lwcl.wicbr.manifests import attach_wicbr_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Attach pre-rendered Wi-CBR image paths to an existing split manifest")
    parser.add_argument("--input-manifest", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--phase-source", choices=["raw", "weighted"], default="weighted")
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()

    summary = attach_wicbr_paths(
        args.input_manifest,
        args.image_root,
        args.output_manifest,
        phase_source=args.phase_source,
        require_all=not args.allow_missing,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
