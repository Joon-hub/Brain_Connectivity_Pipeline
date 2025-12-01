#!/usr/bin/env python3
"""
Aggregate hyperparameter search results and export best config.
"""
import json
import pandas as pd
import yaml
from pathlib import Path


def aggregate_results(search_dir='results/hyperparameter_search'):
    """Combine results from all iterations."""
    search_dir = Path(search_dir)
    
    if not search_dir.exists():
        print(f"ERROR: Search directory not found: {search_dir}")
        return None
    
    all_results = []
    
    # Look for iteration directories (iteration_000, iteration_001, etc.)
    iteration_dirs = sorted(search_dir.glob('iteration_*'))
    
    if not iteration_dirs:
        print(f"WARNING: No iteration directories found in {search_dir}")
        print(f"Looking for pattern: iteration_*")
        print(f"Available directories:")
        for item in search_dir.iterdir():
            if item.is_dir():
                print(f"  - {item.name}")
        return None
    
    print(f"Found {len(iteration_dirs)} iteration directories:")
    
    for iteration_dir in iteration_dirs:
        print(f"\nProcessing: {iteration_dir.name}")
        
        # Find best_params file (try multiple patterns)
        best_params_files = (
            list(iteration_dir.glob('best_params*.json')) +
            list(iteration_dir.glob('best_parameters*.json'))
        )
        
        if not best_params_files:
            print(f"  ✗ No best_params file found")
            print(f"    Available files:")
            for f in iteration_dir.glob('*'):
                print(f"      - {f.name}")
            continue
        
        best_params_file = best_params_files[0]
        print(f"  ✓ Found: {best_params_file.name}")
        
        try:
            with open(best_params_file, 'r') as f:
                data = json.load(f)
            
            # Extract iteration number from directory name
            try:
                # Handle both iteration_001 and iteration_1 formats
                iter_num_str = iteration_dir.name.split('_')[-1]
                iteration_num = int(iter_num_str)
            except (ValueError, IndexError):
                print(f"  ⚠ Could not parse iteration number from {iteration_dir.name}")
                iteration_num = len(all_results)  # Use position as fallback
            
            # Extract what we need
            result = {
                'iteration': iteration_num,
                'best_score': data.get('best_score', 0.0),
            }
            
            # Add individual parameters
            if 'best_params' in data:
                for param, value in data['best_params'].items():
                    clean_param = param.replace('classifier__', '')
                    result[clean_param] = value
            
            all_results.append(result)
            print(f"  ✓ Loaded: score={result['best_score']:.4f}")
            
        except Exception as e:
            print(f"  ✗ Error loading {best_params_file.name}: {e}")
            continue
    
    if not all_results:
        print("\nERROR: No valid results found!")
        print("\nExpected structure:")
        print("  results/hyperparameter_search/")
        print("  ├── iteration_000/")
        print("  │   ├── best_params_*.json")
        print("  │   └── cv_results_*.csv")
        print("  ├── iteration_001/")
        print("  └── ...")
        return None
    
    # Create summary
    df = pd.DataFrame(all_results).sort_values('best_score', ascending=False)
    summary_path = search_dir / 'summary_all_iterations.csv'
    df.to_csv(summary_path, index=False)
    
    print(f"\n{'='*70}")
    print(f"Successfully aggregated {len(df)} iterations")
    print(f"Saved summary to: {summary_path}")
    print(f"{'='*70}")
    
    print(f"\nTop 3 configurations:")
    print(df.head(3).to_string(index=False))
    
    return df


def export_best_params(summary_df, output_path='configs/models/best_from_search.yaml'):
    """Export best parameters as model config."""
    
    if summary_df is None or len(summary_df) == 0:
        print("ERROR: No results to export")
        return None
    
    best = summary_df.iloc[0]
    
    # Extract parameters
    params = {}
    for col in summary_df.columns:
        if col not in ['iteration', 'best_score']:
            value = best[col]
            if pd.notna(value):
                # Convert numpy types to Python types
                if hasattr(value, 'item'):
                    value = value.item()
                params[col] = value
    
    # Add required parameters for logistic regression
    if 'multi_class' not in params:
        params['multi_class'] = 'multinomial'
    
    # Create model config
    config = {
        'class_path': 'sklearn.linear_model.LogisticRegression',
        'params': params
    }
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"\n{'='*70}")
    print(f"✓ Exported best parameters to: {output_path}")
    print(f"{'='*70}")
    print(f"\nBest Configuration:")
    print(f"  Iteration: {int(best['iteration'])}")
    print(f"  Score:     {best['best_score']:.4f}")
    print(f"\n  Parameters:")
    for param, value in params.items():
        print(f"    {param}: {value}")
    print(f"{'='*70}")
    
    return config


def main():
    print("="*70)
    print("HYPERPARAMETER SEARCH AGGREGATION")
    print("="*70)
    print()
    
    # Aggregate
    df = aggregate_results()
    if df is None:
        return 1
    
    # Export
    export_best_params(df)
    
    print(f"\n{'='*70}")
    print("NEXT STEPS")
    print(f"{'='*70}")
    print("\nTo use optimized parameters:")
    print("  python run.py --model best_from_search --experiment-name OptimizedRun")
    print("\nOr in pipeline:")
    print("  ./sh_files/run_brain_pipeline.sh --experiment-name OptimizedRun --use-best-params")
    print(f"{'='*70}\n")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())