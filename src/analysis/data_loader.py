"""
Load and organize results from all 9 models.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


class ResultsLoader:
    """Load results from multiple model runs."""
    
    def __init__(self, base_results_dir: Path):
        """
        Initialize loader.
        
        Parameters
        ----------
        base_results_dir : Path
            Base directory containing all results (e.g., data/results/)
        """
        self.base_dir = Path(base_results_dir)
        self.model_configs = self._define_model_paths()
    
    def _define_model_paths(self) -> Dict[str, Dict]:
        """Define paths for all 9 models."""
        
        configs = {}
        
        # Full connectivity models
        for strategy in ['multinomial', 'ovr', 'ovo']:
            configs[f'full_{strategy}'] = {
                'scope': 'full',
                'strategy': strategy,
                'path': self.base_dir / 'full_connectivity_analysis' / strategy,
                'task_path': self.base_dir / 'full_connectivity_analysis' / strategy / 'task_testing',
                'n_regions': 232
            }
        
        # Left hemisphere models
        for strategy in ['multinomial', 'ovr', 'ovo']:
            configs[f'left_{strategy}'] = {
                'scope': 'left',
                'strategy': strategy,
                'path': self.base_dir / 'left_hemisphere_analysis' / strategy,
                'task_path': self.base_dir / 'left_hemisphere_analysis' / strategy / 'task_testing',
                'n_regions': 116  # approximate
            }
        
        # Right hemisphere models
        for strategy in ['multinomial', 'ovr', 'ovo']:
            configs[f'right_{strategy}'] = {
                'scope': 'right',
                'strategy': strategy,
                'path': self.base_dir / 'right_hemisphere_analysis' / strategy,
                'task_path': self.base_dir / 'right_hemisphere_analysis' / strategy / 'task_testing',
                'n_regions': 116  # approximate
            }
        
        return configs
    
    def load_single_model(self, model_name: str) -> Dict:
        """
        Load results for a single model.
        
        Parameters
        ----------
        model_name : str
            Model identifier (e.g., 'full_multinomial')
        
        Returns
        -------
        results : dict
            Dictionary containing all model results
        """
        
        if model_name not in self.model_configs:
            raise ValueError(f"Unknown model: {model_name}")
        
        config = self.model_configs[model_name]
        task_path = config['task_path']
        
        logger.info(f"Loading {model_name}...")
        
        results = {
            'model_name': model_name,
            'scope': config['scope'],
            'strategy': config['strategy'],
            'n_regions': config['n_regions']
        }
        
        # Load task testing summary
        summary_file = task_path / 'task_testing_summary.json'
        if summary_file.exists():
            with open(summary_file, 'r') as f:
                results['summary'] = json.load(f)
        else:
            logger.warning(f"Summary file not found: {summary_file}")
            return None
        
        # Load predictions
        pred_file = task_path / 'task_predictions.npy'
        if pred_file.exists():
            results['predictions'] = np.load(pred_file)
        
        # Load probabilities
        proba_file = task_path / 'task_probabilities.npy'
        if proba_file.exists():
            results['probabilities'] = np.load(proba_file)
        
        # Load true labels
        true_file = task_path / 'task_true_labels.npy'
        if true_file.exists():
            results['true_labels'] = np.load(true_file)
        
        # Load confusion matrix
        cm_file = task_path / 'task_confusion_matrix.npy'
        if cm_file.exists():
            results['confusion_matrix'] = np.load(cm_file)
        
        # Load per-region metrics
        region_file = task_path / 'task_per_region_metrics.csv'
        if region_file.exists():
            results['per_region_metrics'] = pd.read_csv(region_file)
        
        # Load network metrics
        network_file = task_path / 'task_network_metrics.csv'
        if network_file.exists():
            results['network_metrics'] = pd.read_csv(network_file)
        
        logger.info(f"✓ {model_name} loaded successfully")
        
        return results
    
    def load_all_models(self) -> Dict[str, Dict]:
        """
        Load results from all 9 models.
        
        Returns
        -------
        all_results : dict
            Dictionary mapping model names to their results
        """
        
        all_results = {}
        
        for model_name in self.model_configs.keys():
            try:
                results = self.load_single_model(model_name)
                if results is not None:
                    all_results[model_name] = results
            except Exception as e:
                logger.error(f"Error loading {model_name}: {str(e)}")
        
        logger.info(f"\n✓ Loaded {len(all_results)}/9 models successfully")
        
        return all_results
    
    def verify_all_models_present(self) -> Tuple[List[str], List[str]]:
        """
        Check which models have complete results.
        
        Returns
        -------
        present : list
            Models with complete results
        missing : list
            Models with missing results
        """
        
        present = []
        missing = []
        
        for model_name in self.model_configs.keys():
            task_path = self.model_configs[model_name]['task_path']
            summary_file = task_path / 'task_testing_summary.json'
            
            if summary_file.exists():
                present.append(model_name)
            else:
                missing.append(model_name)
        
        return present, missing


def load_all_results(base_dir: Path) -> Dict[str, Dict]:
    """
    Convenience function to load all results.
    
    Parameters
    ----------
    base_dir : Path
        Base results directory
    
    Returns
    -------
    all_results : dict
        Dictionary of all model results
    """
    
    loader = ResultsLoader(base_dir)
    return loader.load_all_models()