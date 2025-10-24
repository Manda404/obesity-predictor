"""
serialization.py
=========================
Serialization and filesystem utilities for robust artifact persistence.

Features
--------
- Safe save/load with directory creation
- JSON/YAML helpers for configs and schemas
- Joblib for model persistence
- Small, focused API for pipelines and services

Author: Rostand Surel
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import joblib
import yaml

from obesity_predictor.config.logger_config import logger


def ensure_dir(path: str | Path) -> Path:
    """
    Ensure directory exists (create parents if needed).

    Parameters
    ----------
    path : str | Path
        Directory path.

    Returns
    -------
    Path
        The Path object of the created/existing directory.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_model(model: Any, path: str | Path) -> Path:
    """
    Persist a Python object (e.g., model) with joblib.

    Parameters
    ----------
    model : Any
        Fitted model or object to persist.
    path : str | Path
        Destination file (.joblib).

    Returns
    -------
    Path
        Saved path.
    """
    p = Path(path)
    ensure_dir(p.parent)
    joblib.dump(model, p)
    logger.success(f"[IO] Model saved → {p}")
    return p


def load_model(path: str | Path) -> Any:
    """
    Load a joblib-saved object.

    Parameters
    ----------
    path : str | Path

    Returns
    -------
    Any
        Loaded object (e.g., model).
    """
    p = Path(path)
    obj = joblib.load(p)
    logger.info(f"[IO] Model loaded ← {p}")
    return obj


def save_json(data: Dict[str, Any], path: str | Path) -> Path:
    """
    Save a dictionary to a JSON file.

    Parameters
    ----------
    data : dict
    path : str | Path

    Returns
    -------
    Path
        Saved file path.
    """
    p = Path(path)
    ensure_dir(p.parent)
    with p.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info(f"[IO] JSON saved → {p}")
    return p


def load_json(path: str | Path) -> Dict[str, Any]:
    """
    Load a dictionary from a JSON file.

    Parameters
    ----------
    path : str | Path

    Returns
    -------
    dict
    """
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    logger.info(f"[IO] JSON loaded ← {p}")
    return obj


def save_yaml(data: Dict[str, Any], path: str | Path) -> Path:
    """
    Save a dict to a YAML file.

    Parameters
    ----------
    data : dict
    path : str | Path

    Returns
    -------
    Path
    """
    p = Path(path)
    ensure_dir(p.parent)
    with p.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
    logger.info(f"[IO] YAML saved → {p}")
    return p


def load_yaml(path: str | Path) -> Dict[str, Any]:
    """
    Load a dict from a YAML file.

    Parameters
    ----------
    path : str | Path

    Returns
    -------
    dict
    """
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        obj = yaml.safe_load(f) or {}
    logger.info(f"[IO] YAML loaded ← {p}")
    return obj
