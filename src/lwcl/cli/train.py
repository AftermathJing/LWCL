from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path

from lwcl.cli.common import build_loaders
from lwcl.config import config_hash, load_config, save_resolved_config
from lwcl.models.lwcl import LWCLModel
from lwcl.training.trainer import Trainer, seed_everything


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_value(*args: str) -> str | None:
    try:
        return subprocess.check_output(["git", *args], text=True, encoding="utf-8").strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the reconstructed LWCL model")
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
    config["_config_hash"] = config_hash(config)
    seed_everything(int(config["training"].get("seed", 2025)))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_resolved_config(config, output_dir / "resolved_config.yaml")
    manifest_path = Path(config["data"]["manifest"])
    metadata = {
        "git_commit": _git_value("rev-parse", "HEAD"),
        "git_branch": _git_value("branch", "--show-current"),
        "git_status": _git_value("status", "--porcelain"),
        "config_hash": config.get("_config_hash"),
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": _file_hash(manifest_path),
        "python": platform.python_version(),
    }
    try:
        import torch

        metadata.update(
            {
                "torch": torch.__version__,
                "cuda_version": torch.version.cuda,
                "cuda_available": torch.cuda.is_available(),
                "gpu_count": torch.cuda.device_count(),
                "gpus": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
            }
        )
    except ImportError:
        pass
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    train_loader, validation_loader = build_loaders(config)
    model = LWCLModel(config)
    report = model.trainable_parameter_report()
    (output_dir / "parameter_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False))
    trainer = Trainer(model, train_loader, validation_loader, config, output_dir)
    state = trainer.fit(args.resume_from)
    print(json.dumps(state, ensure_ascii=False))


if __name__ == "__main__":
    main()
