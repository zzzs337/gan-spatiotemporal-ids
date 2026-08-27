from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    base_path = config.pop("base", None)
    if base_path:
        base = load_config(path.parent / base_path)
        config = deep_merge(base, config)
    config["_config_path"] = str(path)
    return config


def deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def apply_overrides(config: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    result = copy.deepcopy(config)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Invalid override: {item}. Expected key=value.")
        dotted_key, raw_value = item.split("=", 1)
        value = yaml.safe_load(raw_value)
        target = result
        keys = dotted_key.split(".")
        for key in keys[:-1]:
            target = target.setdefault(key, {})
        target[keys[-1]] = value
    return result


def resolve_path(config: dict[str, Any], value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    config_dir = Path(config["_config_path"]).parent
    project_root = next(
        (
            parent
            for parent in [config_dir, *config_dir.parents]
            if (parent / "configs").is_dir() and (parent / "src").is_dir()
        ),
        config_dir.parent,
    )
    return (project_root / path).resolve()
