"""Service configuration.

Settings are loaded from a YAML file (default ``config/config.yaml``) and
can be overridden by environment variables prefixed with ``SAS_``.

Environment variable names are upper-cased YAML keys with the prefix, e.g.::

    SAS_DEVICE=cuda
    SAS_YOLO_ENABLED=false
    SAS_PORT=8200
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def _load_yaml(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    with p.open() as fh:
        return yaml.safe_load(fh) or {}


def _coerce(value: str, default: Any) -> Any:
    """Coerce a string env-var to the same type as *default*."""
    if isinstance(default, bool):
        return value.lower() in ("1", "true", "yes")
    if isinstance(default, int):
        return int(value)
    if isinstance(default, float):
        return float(value)
    return value


class Settings:
    """Flat settings bag built from YAML + env overrides.

    All public attributes match the YAML keys defined in ``config.yaml``.
    Access is attribute-style (``settings.device``) or dict-style via
    :meth:`get`.
    """

    _PREFIX = "SAS_"

    def __init__(self, yaml_path: str | Path = "config/config.yaml") -> None:
        self._data: dict[str, Any] = _load_yaml(yaml_path)
        self._apply_env_overrides()

    def _apply_env_overrides(self) -> None:
        for key, default in list(self._data.items()):
            env_key = f"{self._PREFIX}{key.upper()}"
            raw = os.environ.get(env_key)
            if raw is not None:
                self._data[key] = _coerce(raw, default)

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def __getattr__(self, name: str) -> Any:
        try:
            return self._data[name]
        except KeyError as exc:
            raise AttributeError(f"Settings has no attribute {name!r}") from exc

    def __repr__(self) -> str:  # pragma: no cover
        return f"Settings({self._data!r})"


# Module-level singleton — import and use ``settings`` everywhere.
settings = Settings()
