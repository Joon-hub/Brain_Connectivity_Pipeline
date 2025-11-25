# models.py
import yaml
from pathlib import Path
import importlib

def load_model_from_config(model_name):
    """
    Load and instantiate a model definged in Configs/{model_name}.yaml.

    Expected YAML structure:
    class_path: "module.submodule.ClassName"
    params:
        param1: value1
        param2: value2

    Returns:
        Instantiated model object
    """

    # 1.Load model configuration
    config_file = Path("Configs") / f"{model_name}.yaml"
    if not config_file.exists():
        raise FileNotFoundError(f"Model configuration file {config_file} not found.")
    
    # 2. read and validate YAML file
    with open(config_file, 'r') as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Configuration file {config_file} is not properly formatted.")
    
    class_path = cfg.get("class_path")
    if not class_path or not isinstance(class_path, str):
        raise ValueError(f"'class_path' not found or invalid in {config_file} (e.g. sklearn...MyClass)...")
    
    params = cfg.get("params", {})
    if not isinstance(params, dict):
        raise ValueError(f"'params' should be a dictionary in {config_file}.")
    
    # 3. split dotted path to module and class name

    try:
        module_path, class_name = class_path.rsplit('.', 1)
    except ValueError:
        raise ValueError(f"Invalid class_path '{class_path}' in {config_file}. Should be in 'module.submodule.ClassName' format.")

    # 4. import module and instantiate class
    module = importlib.import_module(module_path)
    model_cls = getattr(module, class_name)

    # 5. instantiate model with parameters
    try:
        return model_cls(**params)
    except TypeError as e:
        raise TypeError(f"Error instantiating model '{class_name}' with parameters {params}: {e}")