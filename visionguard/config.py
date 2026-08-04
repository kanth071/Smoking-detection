import os
import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT = os.path.join(_HERE, "config.yaml")


def load_config(path: str | None = None) -> dict:
    with open(path or os.getenv("VG_CONFIG", _DEFAULT), "r") as f:
        cfg = yaml.safe_load(f)

    # Environment overrides (handy for deployment / Docker)
    cfg["storage"]["database_url"] = os.getenv(
        "DATABASE_URL", cfg["storage"]["database_url"]
    )
    src = os.getenv("VG_SOURCE")
    if src is not None:
        cfg["source"] = int(src) if src.isdigit() else src
    return cfg


CFG = load_config()
