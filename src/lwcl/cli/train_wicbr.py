from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path

from lwcl.config import load_config, save_resolved_config
from lwcl.wicbr.dataset import WiCBRDataset, build_wicbr_dataloader
from lwcl.wicbr.model import WiCBRNet
from lwcl.wicbr.trainer import WiCBRTrainer, seed_everything


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_value(*args: str) -> str | None:
    try:
        return subprocess.check_output(["git", *args], text=True, cwd=Path(__file__).resolve().parents[3]).strip()
    except Exception:
        return None


def build_loaders(config: dict[str, object], include_test: bool = False):
    data = config["data"]
    training = config["training"]
    manifest = Path(data["manifest"])
    phase_field = str(data.get("phase_field", "phase_path"))
    dfs_field = str(data.get("dfs_field", "dfs_path"))
    image_size = int(data.get("image_size", 224))
    train_dataset = WiCBRDataset(manifest, "train", image_size=image_size, phase_field=phase_field, dfs_field=dfs_field)
    validation_dataset = WiCBRDataset(
        manifest,
        "validation",
        image_size=image_size,
        phase_field=phase_field,
        dfs_field=dfs_field,
    )
    train_loader = build_wicbr_dataloader(
        train_dataset,
        batch_size=int(training.get("batch_size", 10)),
        num_workers=int(training.get("num_workers", 4)),
        balanced_sampling=bool(training.get("balanced_sampling", False)),
        shuffle=True,
    )
    validation_loader = build_wicbr_dataloader(
        validation_dataset,
        batch_size=int(training.get("eval_batch_size", training.get("batch_size", 10))),
        num_workers=int(training.get("num_workers", 4)),
        balanced_sampling=False,
        shuffle=False,
    )
    if not include_test:
        return train_loader, validation_loader
    test_dataset = WiCBRDataset(manifest, "test", image_size=image_size, phase_field=phase_field, dfs_field=dfs_field)
    test_loader = build_wicbr_dataloader(
        test_dataset,
        batch_size=int(training.get("eval_batch_size", training.get("batch_size", 10))),
        num_workers=int(training.get("num_workers", 4)),
        balanced_sampling=False,
        shuffle=False,
    )
    return train_loader, validation_loader, test_loader


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Wi-CBR on a manifest-backed Widar3 split")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--manifest")
    parser.add_argument("--resume-from")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.manifest is not None:
        config["data"]["manifest"] = args.manifest

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    save_resolved_config(config, output_dir / "resolved_config.yaml")

    manifest_path = Path(config["data"]["manifest"])
    run_info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "git_branch": _git_value("rev-parse", "--abbrev-ref", "HEAD"),
        "git_commit": _git_value("rev-parse", "HEAD"),
        "git_status": _git_value("status", "--short"),
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": _file_hash(manifest_path),
    }
    (output_dir / "run_info.json").write_text(json.dumps(run_info, ensure_ascii=False, indent=2), encoding="utf-8")

    seed_everything(int(config["training"].get("seed", 42)))
    train_loader, validation_loader = build_loaders(config, include_test=False)
    model = WiCBRNet(
        num_labels=int(config["data"]["num_labels"]),
        pretrained=bool(config["model"].get("pretrained", True)),
        group_num=int(config["model"].get("group_num", 4)),
        gate_threshold=float(config["model"].get("gate_threshold", 0.5)),
    )
    trainer = WiCBRTrainer(model, train_loader, validation_loader, config, output_dir)
    trainer.fit(resume_from=args.resume_from)


if __name__ == "__main__":
    main()
