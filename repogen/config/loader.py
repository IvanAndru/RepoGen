"""Load and validate pipeline configuration from YAML files.

Reads one or two YAML files (``config.yaml`` and optionally
``reference.yaml``), merges them, expands environment variables in
path values, and returns a validated :class:`PipelineConfig` object.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import ValidationError

from repogen.config.schema import PipelineConfig
from repogen.utils.logging import setup_logging

logger = setup_logging(__name__)

_ENV_VAR_RE = re.compile(r"\$\{?(\w+)\}?")


def _expand_env_vars(value: Any) -> Any:
    """Recursively expand ``$VAR`` / ``${VAR}`` in string values."""
    if isinstance(value, str):
        return _ENV_VAR_RE.sub(lambda m: os.environ.get(m.group(1), m.group(0)), value)
    if isinstance(value, dict):
        return {k: _expand_env_vars(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env_vars(v) for v in value]
    return value


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base* (override wins)."""
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _read_yaml(path: Path) -> dict:
    """Read a YAML file and return its contents as a dict."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML mapping in {path}, got {type(data).__name__}")
    return data


def load_config(
    config_path: Path,
    reference_path: Optional[Path] = None,
) -> PipelineConfig:
    """Load and validate pipeline configuration from YAML files.

    If *reference_path* is provided its contents are deep-merged into
    the main config (reference values act as defaults that the main
    config can override).

    Args:
        config_path: Path to the primary ``config.yaml``.
        reference_path: Optional path to ``reference.yaml``.

    Returns:
        A fully validated :class:`PipelineConfig` instance.

    Raises:
        FileNotFoundError: If a config file is missing.
        pydantic.ValidationError: If the merged config fails schema
            validation.
    """
    raw = _read_yaml(config_path)

    if reference_path is None:
        sibling = Path(config_path).parent / "reference.yaml"
        if sibling.is_file():
            logger.info("Auto-discovered reference config: %s", sibling)
            reference_path = sibling

    if reference_path is not None:
        ref_raw = _read_yaml(reference_path)
        raw = _deep_merge(ref_raw, raw)

    raw = _expand_env_vars(raw)

    try:
        config = PipelineConfig(**raw)
    except ValidationError as exc:
        logger.error("Configuration validation failed:\n%s", exc)
        raise

    logger.info(
        "Configuration loaded: study=%s, output_dir=%s",
        config.study.name,
        config.output_dir,
    )
    return config
