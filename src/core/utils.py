"""
Utilities Module
================
Helper functions for logging, seeding, and provenance tracking.
"""
import numpy as np
import random
import yaml
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Union


def setup_logging(log_file: str = None, level=logging.INFO):
    """
    Setup logging configuration.
    Args:
        log_file: Optional log file path
        level: Logging level
    """
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            level=level,
            format=log_format,
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    else:
        logging.basicConfig(level=level, format=log_format)


def set_random_seeds(seed: int = 42):
    """
    Set random seeds for reproducibility.
    Args:
        seed: Random seed value
    """
    np.random.seed(seed)
    random.seed(seed)
    print(f"Random seeds set to {seed}")


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    Args:
        config_path: Path to config file
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    print(f"Loaded configuration from {config_path}")
    return config


def log_provenance(output_dir: Union[str, Path], config: Dict, results: Dict):
    """
    Log provenance information for reproducibility.
    Args:
        output_dir: Output directory
        config: Configuration dictionary
        results: Results dictionary
    """
    provenance = {
        'timestamp': datetime.now().isoformat(),
        'config': config,
        'results': {
            k: v for k, v in results.items()
            if not isinstance(v, (np.ndarray, dict, list, tuple, object))
            # Filter out complex objects that can't be serialized
        }
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prov_file = output_dir / 'provenance.yaml'
    with open(prov_file, 'w') as f:
        yaml.dump(provenance, f, default_flow_style=False, sort_keys=False)
    print(f"Provenance logged to {prov_file}")


def save_config_copy(config_path: Union[str, Path], output_path: Union[str, Path]):
    """
    Save a copy of the used config file in the output directory for reproducibility.
    
    Args:
        config_path: Path to the original config file (e.g., 'config.yaml')
        output_path: Destination path (e.g., 'results/tables/used_config.yaml')
    """
    config_path = Path(config_path)
    output_path = Path(output_path)
    
    if not config_path.exists():
        print(f"Warning: Config file not found: {config_path} — skipping copy")
        return
    
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(config_path, output_path)
        print(f"Config archived → {output_path}")
    except Exception as e:
        print(f"Warning: Failed to copy config file: {e}")


def print_section(title: str, width: int = 70):
    """
    Print formatted section header.
    Args:
        title: Section title
        width: Header width
    """
    print("\n" + "=" * width)
    print(title.center(width))
    print("=" * width)


def format_time(seconds: float) -> str:
    """
    Format elapsed time as human-readable string.
    Args:
        seconds: Time in seconds
    Returns:
        Formatted string (e.g., "2m 15s")
    """
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    if mins > 0:
        return f"{mins}m {secs}s"
    else:
        return f"{secs}s"