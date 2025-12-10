# models.py
"""
Model Configuration Loader
===========================
Dynamically loads machine learning models from YAML configuration files.

Expected YAML structure:
    class_path: "module.submodule.ClassName"
    params:
        param1: value1
        param2: value2

Example:
    model = load_model_from_config("logistic_regression")
"""

from __future__ import annotations

import importlib
import warnings
from pathlib import Path

import yaml


def load_model_from_config(model_name: str, config_dir: str = "configs/models", **param_overrides):
    """
    Load a model from configs/models/<model_name>.yaml

    Example:
        model = load_model_from_config("xgboost", learning_rate=0.05)
    """
    config_file = Path(config_dir) / f"{model_name}.yaml"

    if not config_file.exists():
        available = sorted(p.stem for p in Path(config_dir).glob("*.yaml"))
        raise FileNotFoundError(
            f"Model config not found: {config_file}\n"
            f"Available: {available or 'none'}"
        )

    cfg = yaml.safe_load(config_file.read_text())
    if not isinstance(cfg, dict):
        raise ValueError(f"{config_file} must contain a top-level dictionary")

    class_path = cfg.get("class_path")
    if not isinstance(class_path, (str, type(None))) or not class_path:
        raise ValueError(f"Missing or invalid 'class_path' in {config_file}")

    params = cfg.get("params", {})
    if not isinstance(params, dict):
        raise ValueError(f"'params' must be a dict in {config_file}")

    # Apply overrides
    if param_overrides:
        params = {**params, **param_overrides}
        print(f"Parameter overrides: {param_overrides}")

    # Dynamic import
    module_path, class_name = class_path.rsplit(".", 1)
    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        if "xgboost" in module_path.lower():
            raise ImportError("XGBoost not installed → pip install xgboost") from e
        raise ImportError(f"Failed to import {module_path}") from e

    model_cls = getattr(module, class_name, None)
    if model_cls is None:
        raise AttributeError(f"{module_path} has no class '{class_name}'")

    # Instantiate
    try:
        model = model_cls(**params)
    except Exception as e:
        raise type(e)(f"Failed to instantiate {class_name} with params: {params}\n{e}") from e

    # Optional sanity check
    if not all(hasattr(model, meth) for meth in ("fit", "predict")):
        warnings.warn(f"Model {class_name} missing 'fit'/'predict' – might not be sklearn-compatible")

    print(f"Loaded {class_name} ← {config_file.name}")
    print(f"   Params: {params or 'default'}")
    return model


def list_available_models(config_dir: str = "configs/models") -> list[str]:
    """Return sorted list of available model names (without .yaml)."""
    path = Path(config_dir)
    if not path.exists():
        return []

    models = sorted(p.stem for p in path.glob("*.yaml"))
    if models:
        print(f"Found {len(models)} model(s) in {config_dir}:")
        for m in models:
            print(f"  → {m}")
    else:
        print(f"No .yaml files in {config_dir}")
    return models


if __name__ == "__main__":
    print("Available models:", list_available_models())
    print()

    try:
        model = load_model_from_config("xgboost")
        # model = load_model_from_config("logistic_regression")
        print(f"\nSuccess! → {type(model).__name__}")
    except Exception as e:
        print(f"\nError: {e}")