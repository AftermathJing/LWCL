from __future__ import annotations

import argparse
import json

from lwcl.wicbr.preprocessing import prepare_wicbr_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Render Widar3 raw CSI into Wi-CBR phase/DFS image manifests")
    parser.add_argument("--input-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--raw-manifest")
    parser.add_argument("--num-receivers", type=int, default=6)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--colormap", default="viridis")
    parser.add_argument("--phase-source", choices=["raw", "weighted"], default="weighted")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    summary = prepare_wicbr_manifest(
        args.input_manifest,
        args.output_dir,
        args.output_manifest,
        raw_manifest=args.raw_manifest,
        num_receivers=args.num_receivers,
        image_size=args.image_size,
        colormap=args.colormap,
        phase_source=args.phase_source,
        workers=args.workers,
        limit=args.limit,
        resume=args.resume,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
