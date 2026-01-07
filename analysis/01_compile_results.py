"""
Compile results from all 9 models.

Usage:
    python analysis/01_compile_results.py
"""

import sys
from pathlib import Path
import logging

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from analysis.data_loader import load_all_results

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Compile all results."""
    
    logger.info("="*80)
    logger.info("COMPILING RESULTS FROM ALL MODELS")
    logger.info("="*80)
    
    # Load all results
    base_dir = project_root / 'data' / 'results'
    
    logger.info(f"\nLoading from: {base_dir}")
    
    all_results = load_all_results(base_dir)
    
    logger.info(f"\n✓ Successfully loaded {len(all_results)} models")
    
    # Save compiled results
    output_dir = project_root / 'outputs' / 'compiled'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    import pickle
    with open(output_dir / 'all_models_results.pkl', 'wb') as f:
        pickle.dump(all_results, f)
    
    logger.info(f"\n✓ Compiled results saved to: {output_dir}/all_models_results.pkl")
    
    # Quick summary
    logger.info("\n" + "="*80)
    logger.info("QUICK SUMMARY")
    logger.info("="*80)
    
    for model_name, results in all_results.items():
        summary = results['summary']
        logger.info(
            f"{model_name:20s} | "
            f"Rest: {summary['rest_train_accuracy']:.4f} | "
            f"Task: {summary['task_test_accuracy']:.4f} | "
            f"Drop: {summary['accuracy_drop']:.4f}"
        )
    
    logger.info("="*80)


if __name__ == "__main__":
    main()