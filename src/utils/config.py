"""Project configuration (configs/default.yaml) and paths relative to the project root."""

from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"


# Read the YAML config into a dict
def load_config(path: str | Path | None = None) -> dict:
    with open(path or DEFAULT_CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


# Full path from a path relative to the project root
def resolve_path(relative_path: str | Path) -> Path:
    return PROJECT_ROOT / relative_path


# Same, and create the folder if it does not exist
def ensure_dir(relative_path: str | Path) -> Path:
    path = resolve_path(relative_path)
    path.mkdir(parents=True, exist_ok=True)
    return path
