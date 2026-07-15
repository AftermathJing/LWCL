from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
from pathlib import Path

from lwcl_v2.config import config_hash, load_config, save_resolved_config
from lwcl_v2.data.dataset import build_loaders
from lwcl_v2.models import SignalV2Classifier
from lwcl_v2.training import Trainer, seed_everything


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args], text=True, encoding="utf-8", stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LWCL Signal Encoder v2")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume-from")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--debug-fail-after-step", type=int)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.max_steps is not None:
        config["training"]["max_steps"] = args.max_steps
    if args.debug_fail_after_step is not None:
        config["training"]["debug_fail_after_step"] = args.debug_fail_after_step
    seed_everything(int(config["training"].get("seed", 2025)))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_resolved_config(config, output_dir / "resolved_config.yaml")
    manifest = Path(config["data"]["manifest"])
    metadata = {
        "git_commit": os.environ.get("LWCL_CODE_COMMIT") or _git("rev-parse", "HEAD"),
        "git_branch": os.environ.get("LWCL_CODE_BRANCH") or _git("branch", "--show-current"),
        "git_status": _git("status", "--porcelain"),
        "config_hash": config_hash(config),
        "manifest": str(manifest.resolve()),
        "manifest_sha256": _sha256(manifest),
        "python": platform.python_version(),
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    train_loader, validation_loader = build_loaders(config)
    model = SignalV2Classifier(config)
    report = model.parameter_report()
    (output_dir / "parameter_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report))
    trainer = Trainer(model, train_loader, validation_loader, config, output_dir)
    state = trainer.fit(args.resume_from)
    print(json.dumps(state))


if __name__ == "__main__":
    main()
