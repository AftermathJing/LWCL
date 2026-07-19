from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from PIL import Image

from lwcl.wicbr.dataset import WiCBRDataset, build_wicbr_dataloader
from lwcl.wicbr.model import WiCBRNet
from lwcl.wicbr.trainer import WiCBRTrainer, seed_everything


def _write_rgb(path: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    array = rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array, mode="RGB").save(path, format="JPEG")


def _build_manifest(root: Path) -> Path:
    rows = []
    for split, start in (("train", 0), ("validation", 10), ("test", 20)):
        for index in range(2):
            sample_id = f"{split}_{index}"
            phase_path = root / "phase" / f"{sample_id}.jpg"
            dfs_path = root / "dfs" / f"{sample_id}.jpg"
            _write_rgb(phase_path, start + index)
            _write_rgb(dfs_path, start + index + 100)
            rows.append(
                {
                    "sample_id": sample_id,
                    "label": str(index % 2),
                    "split": split,
                    "phase_path": phase_path.relative_to(root).as_posix(),
                    "dfs_path": dfs_path.relative_to(root).as_posix(),
                    "subject": f"s{index}",
                    "environment": "e1",
                    "position": "p1",
                    "orientation": "o1",
                }
            )
    manifest = root / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return manifest


def test_wicbr_dataset_and_trainer_smoke(tmp_path: Path) -> None:
    manifest = _build_manifest(tmp_path)
    train_dataset = WiCBRDataset(manifest, "train", image_size=64)
    validation_dataset = WiCBRDataset(manifest, "validation", image_size=64)
    assert len(train_dataset) == 2
    sample = train_dataset[0]
    assert sample["phase_inputs"].shape == (3, 64, 64)
    assert sample["dfs_inputs"].shape == (3, 64, 64)

    train_loader = build_wicbr_dataloader(
        train_dataset,
        batch_size=2,
        num_workers=0,
        balanced_sampling=False,
        shuffle=True,
    )
    validation_loader = build_wicbr_dataloader(
        validation_dataset,
        batch_size=2,
        num_workers=0,
        balanced_sampling=False,
        shuffle=False,
    )

    seed_everything(42)
    model = WiCBRNet(num_labels=2, pretrained=False)
    config = {
        "data": {"manifest": str(manifest), "num_labels": 2},
        "training": {
            "seed": 42,
            "device": "cpu",
            "precision": "fp32",
            "epochs": 1,
            "max_steps": 1,
            "batch_size": 2,
            "eval_batch_size": 2,
            "num_workers": 0,
            "learning_rate": 1e-4,
            "weight_decay": 0.0,
            "temperature": 0.1,
            "beta_1": 0.1,
            "step_lr": {"step_size": 3, "gamma": 0.5},
            "log_every_steps": 1,
            "eval_every_steps": 1,
            "save_every_steps": 1,
            "early_stopping_patience": 0,
            "gradient_clip_norm": 1.0,
        },
    }
    trainer = WiCBRTrainer(model, train_loader, validation_loader, config, tmp_path / "outputs")
    state = trainer.fit()
    assert state["global_step"] == 1
    assert (tmp_path / "outputs" / "checkpoints" / "last.pt").exists()


def test_wicbr_official_epoch_test_selection_smoke(tmp_path: Path) -> None:
    manifest = _build_manifest(tmp_path)
    train_dataset = WiCBRDataset(manifest, "train", image_size=64)
    test_dataset = WiCBRDataset(manifest, "test", image_size=64)

    train_loader = build_wicbr_dataloader(train_dataset, batch_size=2, num_workers=0, balanced_sampling=False, shuffle=True)
    test_loader = build_wicbr_dataloader(test_dataset, batch_size=2, num_workers=0, balanced_sampling=False, shuffle=False)

    seed_everything(888)
    model = WiCBRNet(num_labels=2, pretrained=False)
    config = {
        "data": {"manifest": str(manifest), "num_labels": 2},
        "training": {
            "seed": 888,
            "device": "cpu",
            "precision": "fp32",
            "epochs": 1,
            "batch_size": 2,
            "eval_batch_size": 2,
            "num_workers": 0,
            "learning_rate": 1e-4,
            "weight_decay": 0.0,
            "temperature": 0.1,
            "beta_1": 0.1,
            "step_lr": {"step_size": 3, "gamma": 0.5},
            "log_every_steps": 1,
            "eval_every_steps": None,
            "eval_every_epochs": 1,
            "save_every_steps": None,
            "save_every_epochs": 1,
            "early_stopping_patience": 0,
            "gradient_clip_norm": 1.0,
            "selection_split": "test",
            "monitor_metric": "accuracy",
            "max_steps": 1,
        },
    }
    trainer = WiCBRTrainer(model, train_loader, None, config, tmp_path / "outputs_official", selection_loader=test_loader)
    state = trainer.fit()
    assert state["best_metric_name"] == "accuracy"
    assert state["selection_split"] == "test"
    assert (tmp_path / "outputs_official" / "checkpoints" / "last.pt").exists()
    assert (tmp_path / "outputs_official" / "checkpoints" / "best.pt").exists()


def test_wicbr_step_early_stopping_saves_best_and_last(tmp_path: Path) -> None:
    manifest = _build_manifest(tmp_path)
    train_dataset = WiCBRDataset(manifest, "train", image_size=64)
    validation_dataset = WiCBRDataset(manifest, "validation", image_size=64)
    train_loader = build_wicbr_dataloader(
        train_dataset, batch_size=1, num_workers=0, balanced_sampling=False, shuffle=False
    )
    validation_loader = build_wicbr_dataloader(
        validation_dataset, batch_size=2, num_workers=0, balanced_sampling=False, shuffle=False
    )

    seed_everything(42)
    model = WiCBRNet(num_labels=2, pretrained=False)
    config = {
        "data": {"manifest": str(manifest), "num_labels": 2},
        "training": {
            "seed": 42,
            "device": "cpu",
            "precision": "fp32",
            "epochs": 3,
            "batch_size": 1,
            "eval_batch_size": 2,
            "num_workers": 0,
            "learning_rate": 1e-4,
            "weight_decay": 0.0,
            "temperature": 0.1,
            "beta_1": 0.1,
            "scheduler": "none",
            "log_every_steps": 1,
            "eval_every_steps": 1,
            "eval_every_epochs": 0,
            "save_every_steps": 0,
            "save_every_epochs": 0,
            "save_last_on_eval": True,
            "early_stopping_patience": 1,
            "early_stopping_min_delta": 2.0,
            "gradient_clip_norm": 1.0,
            "selection_split": "validation",
            "monitor_metric": "macro_f1",
        },
    }
    trainer = WiCBRTrainer(model, train_loader, validation_loader, config, tmp_path / "outputs_earlystop")
    trainer.state["best_metric"] = -2.0
    state = trainer.fit()

    assert state["global_step"] == 2
    assert state["bad_evaluations"] == 1
    best_checkpoint = tmp_path / "outputs_earlystop" / "checkpoints" / "best.pt"
    last_checkpoint = tmp_path / "outputs_earlystop" / "checkpoints" / "last.pt"
    assert best_checkpoint.exists()
    assert last_checkpoint.exists()

    config["training"]["early_stopping_patience"] = 0
    config["training"]["max_steps"] = 3
    resumed_model = WiCBRNet(num_labels=2, pretrained=False)
    resumed_trainer = WiCBRTrainer(
        resumed_model, train_loader, validation_loader, config, tmp_path / "outputs_resumed"
    )
    resumed_state = resumed_trainer.fit(resume_from=last_checkpoint)

    assert resumed_state["global_step"] == 3
    assert (tmp_path / "outputs_resumed" / "checkpoints" / "last.pt").exists()
