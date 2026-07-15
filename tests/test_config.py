from __future__ import annotations

from pathlib import Path

from lwcl_v2.config import load_config


ROOT = Path(__file__).resolve().parents[1]


def test_p1_preprocessing_and_base_position_defaults():
    preprocessing = load_config(ROOT / "configs" / "preprocessing_p1.yaml")
    assert preprocessing["preprocessing"]["stft_window"] == 251
    assert preprocessing["preprocessing"]["stft_hop"] == 50
    assert preprocessing["preprocessing"]["stft_overlap"] == 201
    config = load_config(ROOT / "configs" / "signal_v2_base.yaml")
    hste = config["model"]["hste"]
    assert hste["input_absolute_position_encoding"] is False
    assert hste["local_position"]["reset_each_window"] is True
    assert hste["global_position"]["use_physical_time"] is True
    assert hste["transformer_ffn"] == "gelu"


def test_all_incremental_configs_resolve():
    paths = list((ROOT / "configs" / "experiments").glob("*.yaml"))
    paths.extend((ROOT / "configs" / "ablations").glob("*.yaml"))
    for path in sorted(paths):
        if path.name.startswith("b1_"):
            continue
        config = load_config(path)
        assert config["architecture"] == "signal_v2_classifier"
