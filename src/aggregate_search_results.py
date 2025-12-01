#!/usr/bin/env python3
"""
Aggregate results from parallel hyperparameter search.
"""
import json
import pandas as pd
from pathlib import Path


def aggregate_results(search_dir='results/hyperparameter_search'):
    """Combine results from all parallel iterations."""
    search_dir = Path(search_dir)
    
    all_results = []
    
    for iteration_dir in sorted(search_dir.glob('iteration_*')):
        best_params_file = iteration_dir / 'best_params.json'
        results_file = iteration_dir / 'random_search_results.csv'
        
        if best_params_file.exists() and results_file.exists():
            with open(best_params_file, 'r') as f:
                best_params = json.load(f)
            
            results_df = pd.read_csv(results_file)
            best_score = results_df['mean_test_score'].max()
            
            all_results.append({
                'iteration': int(iteration_dir.name.split('_')[1]),
                'best_score': best_score,
                **best_params
            })
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(all_results).sort_values('best_score', ascending=False)
    
    # Save
    summary_df.to_csv(search_dir / 'summary_all_iterations.csv', index=False)
    
    print(f"\n{'='*60}")
    print("HYPERPARAMETER SEARCH SUMMARY")
    print(f"{'='*60}")
    print(f"Total iterations: {len(summary_df)}")
    print(f"\nTop 5 configurations:")
    print(summary_df.head(5).to_string(index=False))
    print(f"\nBest configuration:")
    best = summary_df.iloc[0]
    print(f"  Accuracy: {best['best_score']:.4f}")
    for col in summary_df.columns:
        if col not in ['iteration', 'best_score']:
            print(f"  {col}: {best[col]}")
    
    return summary_df


if __name__ == "__main__":
    aggregate_results()