"""
Universal data loader for all analysis scripts.

Provides convenient functions to load results, check data integrity,
and access specific model outputs.

Usage:
    from analysis.data_loader import DataLoader
    
    loader = DataLoader()
    all_results = loader.load_all()
    full_multi = loader.load_model('full_multinomial')
"""

import sys
from pathlib import Path
import json
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import logging

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Universal data loader for thesis analysis.
    
    Handles loading results from all 9 models and provides
    convenient access methods.
    """
    
    def __init__(
        self,
        results_dir: Optional[Path] = None,
        compiled_dir: Optional[Path] = None
    ):
        """
        Initialize data loader.
        
        Parameters
        ----------
        results_dir : Path, optional
            Base directory for model results
        compiled_dir : Path, optional
            Directory for compiled results
        """
        
        if results_dir is None:
            self.results_dir = project_root / 'data' / 'results'
        else:
            self.results_dir = Path(results_dir)
        
        if compiled_dir is None:
            self.compiled_dir = project_root / 'outputs' / 'compiled'
        else:
            self.compiled_dir = Path(compiled_dir)
        
        self.model_configs = self._define_model_configs()
        self._cache = {}
    
    def _define_model_configs(self) -> Dict[str, Dict[str, Any]]:
        """Define configuration for all 9 models."""
        
        configs = {}
        
        scopes = ['full', 'left', 'right']
        strategies = ['multinomial', 'ovr', 'ovo']
        
        for scope in scopes:
            # Determine number of regions
            if scope == 'full':
                n_regions = 232
                base_name = 'full_connectivity_analysis'
            elif scope == 'left':
                n_regions = 116
                base_name = 'left_hemisphere_analysis'
            else:  # right
                n_regions = 116
                base_name = 'right_hemisphere_analysis'
            
            for strategy in strategies:
                model_name = f'{scope}_{strategy}'
                
                configs[model_name] = {
                    'scope': scope,
                    'strategy': strategy,
                    'n_regions': n_regions,
                    'base_dir': self.results_dir / base_name / strategy,
                    'task_dir': self.results_dir / base_name / strategy / 'task_testing',
                    'cv_dir': self.results_dir / base_name / strategy
                }
        
        return configs
    
    def load_all(self, use_cache: bool = True) -> Dict[str, Dict[str, Any]]:
        """
        Load all model results.
        
        Parameters
        ----------
        use_cache : bool
            If True, try to load from compiled pickle first
        
        Returns
        -------
        all_results : dict
            Dictionary mapping model names to results
        """
        
        if use_cache:
            compiled_path = self.compiled_dir / 'all_models_results.pkl'
            
            if compiled_path.exists():
                logger.info(f"Loading from cache: {compiled_path}")
                with open(compiled_path, 'rb') as f:
                    all_results = pickle.load(f)
                logger.info(f"✓ Loaded {len(all_results)} models from cache")
                return all_results
        
        # Load from individual model directories
        logger.info("Loading from individual model directories...")
        all_results = {}
        
        for model_name in self.model_configs.keys():
            try:
                results = self.load_model(model_name)
                if results is not None:
                    all_results[model_name] = results
            except Exception as e:
                logger.error(f"Error loading {model_name}: {str(e)}")
        
        logger.info(f"✓ Loaded {len(all_results)}/{len(self.model_configs)} models")
        
        return all_results
    
    def load_model(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Load results for a single model.
        
        Parameters
        ----------
        model_name : str
            Model identifier (e.g., 'full_multinomial')
        
        Returns
        -------
        results : dict or None
            Dictionary containing all model results
        """
        
        if model_name not in self.model_configs:
            raise ValueError(f"Unknown model: {model_name}. Valid models: {list(self.model_configs.keys())}")
        
        config = self.model_configs[model_name]
        task_dir = config['task_dir']
        cv_dir = config['cv_dir']
        
        logger.info(f"Loading {model_name}...")
        
        results = {
            'model_name': model_name,
            'scope': config['scope'],
            'strategy': config['strategy'],
            'n_regions': config['n_regions']
        }
        
        # Load task testing summary
        summary_file = task_dir / 'task_testing_summary.json'
        if not summary_file.exists():
            logger.warning(f"Summary file not found: {summary_file}")
            return None
        
        with open(summary_file, 'r') as f:
            results['summary'] = json.load(f)
        
        # Load task predictions
        pred_file = task_dir / 'task_predictions.npy'
        if pred_file.exists():
            results['predictions'] = np.load(pred_file)
        
        # Load task probabilities
        proba_file = task_dir / 'task_probabilities.npy'
        if proba_file.exists():
            results['probabilities'] = np.load(proba_file)
        
        # Load true labels
        true_file = task_dir / 'task_true_labels.npy'
        if true_file.exists():
            results['true_labels'] = np.load(true_file)
        
        # Load confusion matrix
        cm_file = task_dir / 'task_confusion_matrix.npy'
        if cm_file.exists():
            results['confusion_matrix'] = np.load(cm_file)
        
        # Load per-region metrics
        region_file = task_dir / 'task_per_region_metrics.csv'
        if region_file.exists():
            results['per_region_metrics'] = pd.read_csv(region_file)
        
        # Load network metrics
        network_file = task_dir / 'task_network_metrics.csv'
        if network_file.exists():
            results['network_metrics'] = pd.read_csv(network_file)
        
        # Load CV predictions (if available)
        cv_pred_file = cv_dir / 'cv_predictions.npy'
        if cv_pred_file.exists():
            results['cv_predictions'] = np.load(cv_pred_file)
        
        # Load CV true labels
        cv_true_file = cv_dir / 'cv_true_labels.npy'
        if cv_true_file.exists():
            results['cv_true_labels'] = np.load(cv_true_file)
        
        # Load overall CV metrics
        cv_metrics_file = cv_dir / 'overall_metrics.json'
        if cv_metrics_file.exists():
            with open(cv_metrics_file, 'r') as f:
                results['cv_metrics'] = json.load(f)
        
        # Load final model (if saved)
        model_file = task_dir / 'final_model.pkl'
        if model_file.exists():
            with open(model_file, 'rb') as f:
                results['final_model'] = pickle.load(f)
        
        # Load scaler
        scaler_file = task_dir / 'final_scaler.pkl'
        if scaler_file.exists():
            with open(scaler_file, 'rb') as f:
                results['final_scaler'] = pickle.load(f)
        
        logger.info(f"✓ {model_name} loaded successfully")
        
        return results
    
    def load_specific_file(
        self,
        model_name: str,
        file_name: str,
        from_task: bool = True
    ) -> Any:
        """
        Load a specific file for a model.
        
        Parameters
        ----------
        model_name : str
            Model identifier
        file_name : str
            Name of file to load
        from_task : bool
            If True, load from task_testing directory
        
        Returns
        -------
        data : Any
            Loaded data
        """
        
        config = self.model_configs[model_name]
        
        if from_task:
            file_path = config['task_dir'] / file_name
        else:
            file_path = config['cv_dir'] / file_name
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Determine file type and load
        suffix = file_path.suffix
        
        if suffix == '.npy':
            return np.load(file_path)
        elif suffix == '.csv':
            return pd.read_csv(file_path)
        elif suffix == '.json':
            with open(file_path, 'r') as f:
                return json.load(f)
        elif suffix == '.pkl':
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        else:
            raise ValueError(f"Unsupported file type: {suffix}")
    
    def verify_data_integrity(self) -> pd.DataFrame:
        """
        Check which models have complete data.
        
        Returns
        -------
        integrity_df : pd.DataFrame
            DataFrame showing data completeness for each model
        """
        
        checks = []
        
        for model_name in self.model_configs.keys():
            config = self.model_configs[model_name]
            task_dir = config['task_dir']
            
            check = {
                'model': model_name,
                'scope': config['scope'],
                'strategy': config['strategy']
            }
            
            # Check essential files
            essential_files = [
                'task_testing_summary.json',
                'task_predictions.npy',
                'task_true_labels.npy',
                'task_confusion_matrix.npy',
                'task_per_region_metrics.csv',
                'task_network_metrics.csv'
            ]
            
            for file_name in essential_files:
                file_path = task_dir / file_name
                check[file_name] = file_path.exists()
            
            # Check if all essential files present
            check['complete'] = all(check[f] for f in essential_files)
            
            checks.append(check)
        
        integrity_df = pd.DataFrame(checks)
        
        return integrity_df
    
    def get_model_summary(self, model_name: str) -> pd.Series:
        """
        Get quick summary of a model's performance.
        
        Parameters
        ----------
        model_name : str
            Model identifier
        
        Returns
        -------
        summary : pd.Series
            Performance summary
        """
        
        results = self.load_model(model_name)
        
        if results is None:
            return None
        
        summary_dict = results['summary']
        
        summary = pd.Series({
            'model': model_name,
            'scope': results['scope'],
            'strategy': results['strategy'],
            'n_regions': results['n_regions'],
            'rest_accuracy': summary_dict['rest_train_accuracy'],
            'task_accuracy': summary_dict['task_test_accuracy'],
            'accuracy_drop': summary_dict['accuracy_drop'],
            'accuracy_drop_pct': (summary_dict['accuracy_drop'] / summary_dict['rest_train_accuracy']) * 100,
            'balanced_accuracy': summary_dict['task_balanced_accuracy'],
            'top_5_accuracy': summary_dict.get('task_top_5_accuracy', np.nan),
            'n_rest_subjects': summary_dict['n_rest_subjects'],
            'n_task_subjects': summary_dict['n_task_subjects']
        })
        
        return summary
    
    def get_all_summaries(self) -> pd.DataFrame:
        """
        Get performance summaries for all models.
        
        Returns
        -------
        summaries_df : pd.DataFrame
            DataFrame with all model summaries
        """
        
        summaries = []
        
        for model_name in self.model_configs.keys():
            summary = self.get_model_summary(model_name)
            if summary is not None:
                summaries.append(summary)
        
        summaries_df = pd.DataFrame(summaries)
        
        # Sort by scope and strategy
        scope_order = {'full': 1, 'left': 2, 'right': 3}
        strategy_order = {'multinomial': 1, 'ovr': 2, 'ovo': 3}
        
        summaries_df['scope_order'] = summaries_df['scope'].map(scope_order)
        summaries_df['strategy_order'] = summaries_df['strategy'].map(strategy_order)
        summaries_df = summaries_df.sort_values(['scope_order', 'strategy_order'])
        summaries_df = summaries_df.drop(['scope_order', 'strategy_order'], axis=1)
        summaries_df = summaries_df.reset_index(drop=True)
        
        return summaries_df
    
    def filter_models(
        self,
        scope: Optional[str] = None,
        strategy: Optional[str] = None
    ) -> List[str]:
        """
        Get list of model names matching criteria.
        
        Parameters
        ----------
        scope : str, optional
            Filter by scope ('full', 'left', 'right')
        strategy : str, optional
            Filter by strategy ('multinomial', 'ovr', 'ovo')
        
        Returns
        -------
        model_names : list
            List of matching model names
        """
        
        model_names = []
        
        for model_name, config in self.model_configs.items():
            if scope is not None and config['scope'] != scope:
                continue
            if strategy is not None and config['strategy'] != strategy:
                continue
            
            model_names.append(model_name)
        
        return model_names
    
    def load_reorganization_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load reorganization data for all models.
        
        Returns
        -------
        reorg_data : dict
            Dictionary mapping model names to reorganization DataFrames
        """
        
        reorg_data = {}
        
        for model_name in self.model_configs.keys():
            reorg_file = self.compiled_dir / f'{model_name}_reorganization.csv'
            
            if reorg_file.exists():
                reorg_data[model_name] = pd.read_csv(reorg_file)
            else:
                logger.warning(f"Reorganization file not found for {model_name}")
        
        return reorg_data
    
    def load_error_analysis(self) -> Dict[str, pd.DataFrame]:
        """
        Load error analysis data for all models.
        
        Returns
        -------
        error_data : dict
            Dictionary mapping model names to error DataFrames
        """
        
        error_data = {}
        
        for model_name in self.model_configs.keys():
            error_file = self.compiled_dir / f'{model_name}_error_types.csv'
            
            if error_file.exists():
                error_data[model_name] = pd.read_csv(error_file)
            else:
                logger.warning(f"Error file not found for {model_name}")
        
        return error_data
    
    def save_compiled(self, all_results: Dict[str, Dict[str, Any]]):
        """
        Save compiled results to pickle.
        
        Parameters
        ----------
        all_results : dict
            Dictionary of all model results
        """
        
        self.compiled_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = self.compiled_dir / 'all_models_results.pkl'
        
        with open(output_path, 'wb') as f:
            pickle.dump(all_results, f)
        
        logger.info(f"✓ Compiled results saved to: {output_path}")


def load_data(use_cache: bool = True) -> Dict[str, Dict[str, Any]]:
    """
    Convenience function to load all data.
    
    Parameters
    ----------
    use_cache : bool
        If True, try to load from compiled pickle
    
    Returns
    -------
    all_results : dict
        Dictionary of all model results
    """
    
    loader = DataLoader()
    return loader.load_all(use_cache=use_cache)


def verify_data() -> pd.DataFrame:
    """
    Convenience function to verify data integrity.
    
    Returns
    -------
    integrity_df : pd.DataFrame
        Data integrity report
    """
    
    loader = DataLoader()
    return loader.verify_data_integrity()


if __name__ == "__main__":
    """Quick test of data loader."""
    
    logging.basicConfig(level=logging.INFO)
    
    print("="*80)
    print("DATA LOADER TEST")
    print("="*80)
    
    # Initialize loader
    loader = DataLoader()
    
    # Check data integrity
    print("\nChecking data integrity...")
    integrity_df = loader.verify_data_integrity()
    print(integrity_df[['model', 'complete']])
    
    # Load all data
    print("\nLoading all data...")
    all_results = loader.load_all(use_cache=False)
    print(f"✓ Loaded {len(all_results)} models")
    
    # Get summaries
    print("\nModel summaries:")
    summaries = loader.get_all_summaries()
    print(summaries[['model', 'rest_accuracy', 'task_accuracy', 'accuracy_drop']])
    
    print("\n✓ Data loader working correctly")