from __future__ import annotations

import argparse
import json

from lwcl_v2.data.csi_bench import HAR_PROTOCOLS, build_har_protocol_manifests


def main() -> None:
    parser = argparse.ArgumentParser(description="Build official CSI-Bench HAR protocol manifests")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--protocol", action="append", choices=HAR_PROTOCOLS)
    args = parser.parse_args()
    summary = build_har_protocol_manifests(args.dataset_root, args.output_root, args.protocol)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
