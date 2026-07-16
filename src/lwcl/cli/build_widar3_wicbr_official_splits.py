from __future__ import annotations

import argparse
import json

from lwcl.wicbr.official_protocol import build_widar3_official_wicbr_manifests


def main() -> None:
    parser = argparse.ArgumentParser(description="Build official Widar3 Wi-CBR protocol manifests")
    parser.add_argument("--input-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    summary = build_widar3_official_wicbr_manifests(args.input_manifest, args.output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
