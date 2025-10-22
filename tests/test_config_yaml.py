import yaml
from pathlib import Path

def test_config_has_core_fields():
    cfg_path = Path("config.yaml")
    assert cfg_path.exists(), "config.yaml missing"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    for k in ["experiment_name", "model_name", "img_size", "batch_size"]:
        assert k in cfg, f"config.yaml missing key: {k}"
