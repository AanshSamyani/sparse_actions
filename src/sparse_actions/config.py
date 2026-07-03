"""Tiny YAML config loader with `_base_` inheritance and dotted CLI overrides."""
from __future__ import annotations

import types
from pathlib import Path

import yaml


def _deep_update(base: dict, upd: dict) -> dict:
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def _to_ns(d):
    if isinstance(d, dict):
        return types.SimpleNamespace(**{k: _to_ns(v) for k, v in d.items()})
    if isinstance(d, list):
        return [_to_ns(v) for v in d]
    return d


def _load_raw(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    base_name = cfg.pop("_base_", None)
    if base_name:
        base = _load_raw(path.parent / base_name)
        cfg = _deep_update(base, cfg)
    return cfg


def _coerce(val: str):
    # Parse ints, floats, bools, and lists (e.g. "[-2.0, -3.0]") via YAML; fall back
    # to the raw string on any parse issue.
    try:
        return yaml.safe_load(val)
    except Exception:  # noqa: BLE001
        return val


def apply_overrides(cfg: dict, overrides: list[str] | None) -> dict:
    """Apply `--set a.b.c=value` style overrides in place."""
    for ov in overrides or []:
        key, _, raw = ov.partition("=")
        parts = key.split(".")
        node = cfg
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = _coerce(raw)
    return cfg


def load_config(path: str | Path, overrides: list[str] | None = None):
    raw = _load_raw(Path(path))
    raw = apply_overrides(raw, overrides)
    ns = _to_ns(raw)
    ns._raw = raw  # keep the plain dict around for serialization
    return ns
