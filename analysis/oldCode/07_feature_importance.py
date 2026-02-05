"""
Analyze feature importance and connection-level patterns.

For logistic regression models, analyzes:
- Coefficient magnitudes
- Top important connections per region
- Connection patterns that drive classification
- Changes from rest to task (if models available)

Usage:
    python analysis/07_feature_importance.py
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import logging

# Add project root
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from analysis.data_loader import DataLoader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def extract_model_coefficients(model, feature_names: list = None) -> pd.DataFrame:
    """
    Extract coefficients from logistic regression model.
    
    Parameters
    ----------
    model : LogisticRegression
        Trained model
    feature_names : list, optional
        Names of features
    
    Returns
    -------
    coef_df : pd.DataFrame
        DataFrame with coefficients
    """
    
    # Get coefficients
    # Shape: (n_classes, n_features) for multinomial
    # or (n_classes, n_features) for OvR
    coefs = model.coef_
    
    n_classes, n_features = coefs.shape
    
    if feature_names is None:
        feature_names = [f'feature_{i}' for i in range(n_features)]
    
    # Create DataFrame
    coef_data = []
    
    for class_idx in range(n_classes):
        for feat_idx in range(n_features):
            coef_data.append({
                'class': class_idx,
                'feature_idx': feat_idx,
                'feature_name': feature_names[feat_idx],
                'coefficient': coefs[class_idx, feat_idx],
                'abs_coefficient': abs(coefs[class_idx, feat_idx])
            })
    
    coef_df = pd.DataFrame(coef_data)
    
    return coef_df


def compute_feature_importance(
    model,
    X: np.ndarray,
    feature_names: list = None
) -> pd.DataFrame:
    """
    Compute feature importance based on coefficients and feature std.
    
    Importance = |coefficient| * std(feature)
    
    Parameters
    ----------
    model : LogisticRegression
        Trained model
    X : np.ndarray
        Feature matrix (to compute std)
    feature_names : list, optional
        Feature names
    
    Returns
    -------
    importance_df : pd.DataFrame
        Feature importance scores
    """
    
    coef_df = extract_model_coefficients(model, feature_names)
    
    # Compute feature standard deviations
    feature_stds = X.std(axis=0)
    
    # Add to DataFrame
    coef_df['feature_std'] = coef_df['feature_idx'].map(
        lambda idx: feature_stds[idx]
    )
    
    # Compute importance
    coef_df['importance'] = coef_df['abs_coefficient'] * coef_df['feature_std']
    
    return coef_df


def get_top_features_per_class(
    importance_df: pd.DataFrame,
    n_top: int = 10
) -> pd.DataFrame:
    """
    Get top N important features for each class.
    
    Parameters
    ----------
    importance_df : pd.DataFrame
        Feature importance DataFrame
    n_top : int
        Number of top features to return per class
    
    Returns
    -------
    top_features_df : pd.DataFrame
        Top features for each class
    """
    
    top_features = importance_df.groupby('class').apply(
        lambda x: x.nlargest(n_top, 'importance')
    ).reset_index(drop=True)
    
    return top_features


def analyze_connection_patterns(
    region_info: pd.DataFrame,
    importance_df: pd.DataFrame,
    top_n: int = 20
) -> pd.DataFrame:
    """
    Analyze patterns in important connections.
    
    Parameters
    ----------
    region_info : pd.DataFrame
        Region metadata
    importance_df : pd.DataFrame
        Feature importance
    top_n : int
        Number of top features to analyze
    
    Returns
    -------
    pattern_df : pd.DataFrame
        Connection pattern analysis
    """
    
    # Get top features overall
    top_features = importance_df.nlargest(top_n, 'importance')
    
    # For each feature (connection), identify source region
    # Feature indices correspond to brain regions
    
    patterns = []
    
    for _, row in top_features.iterrows():
        feat_idx = row['feature_idx']
        
        if feat_idx < len(region_info):
            region = region_info.iloc[feat_idx]
            
            patterns.append({
                'feature_idx': feat_idx,
                'feature_name': row['feature_name'],
                'region_name': region['region_name'],
                'network': region['network'],
                'hemisphere': region.get('hemisphere', 'unknown'),
                'importance': row['importance'],
                'coefficient': row['coefficient'],
                'class': row['class']
            })
    
    pattern_df = pd.DataFrame(patterns)
    
    return pattern_df


def compare_coefficient_distributions(
    all_results: dict,
    scope: str = 'full'
) -> pd.DataFrame:
    """
    Compare coefficient distributions across strategies.
    
    Parameters
    ----------
    all_results : dict
        All model results
    scope : str
        Scope to analyze
    
    Returns
    -------
    comparison_df : pd.DataFrame
        Coefficient distribution comparison
    """
    
    strategies = ['multinomial', 'ovr', 'ovo']
    
    comparison_data = []
    
    for strategy in strategies:
        model_name = f'{scope}_{strategy}'
        
        if model_name not in all_results:
            continue
        
        if 'final_model' not in all_results[model_name]:
            logger.warning(f"Model not available for {model_name}")
            continue
        
        model = all_results[model_name]['final_model']
        coefs = model.coef_.flatten()
        
        comparison_data.append({
            'strategy': strategy,
            'mean_abs_coef': np.abs(coefs).mean(),
            'median_abs_coef': np.median(np.abs(coefs)),
            'std_abs_coef': np.abs(coefs).std(),
            'max_abs_coef': np.abs(coefs).max(),
            'sparsity': (np.abs(coefs) < 0.01).sum() / len(coefs)
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    return comparison_df


def plot_coefficient_heatmap(
    coef_df: pd.DataFrame,
    region_info: pd.DataFrame,
    save_path: Path,
    n_regions: int = 50,
    n_classes: int = 50
):
    """
    Plot heatmap of coefficients for top regions/classes.
    
    Parameters
    ----------
    coef_df : pd.DataFrame
        Coefficient DataFrame
    region_info : pd.DataFrame
        Region metadata
    save_path : Path
        Where to save figure
    n_regions : int
        Number of top regions to show
    n_classes : int
        Number of top classes to show
    """
    
    # Pivot to matrix form
    coef_matrix = coef_df.pivot_table(
        index='class',
        columns='feature_idx',
        values='coefficient',
        fill_value=0
    )
    
    # Select top regions by variance
    region_variances = coef_matrix.var(axis=0)
    top_regions = region_variances.nlargest(n_regions).index
    
    # Select top classes by variance
    class_variances = coef_matrix.var(axis=1)
    top_classes = class_variances.nlargest(n_classes).index
    
    # Subset
    coef_subset = coef_matrix.loc[top_classes, top_regions]
    
    # Plot
    fig, ax = plt.subplots(figsize=(14, 10))
    
    sns.heatmap(
        coef_subset,
        cmap='RdBu_r',
        center=0,
        cbar_kws={'label': 'Coefficient'},
        ax=ax,
        vmin=-coef_subset.abs().max(),
        vmax=coef_subset.abs().max()
    )
    
    ax.set_title('Coefficient Heatmap (Top Regions & Classes)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Feature (Region) Index', fontsize=12)
    ax.set_ylabel('Class (Region) Index', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"✓ Saved: {save_path}")
    plt.close()


def plot_top_connections(
    pattern_df: pd.DataFrame,
    save_path: Path,
    n_top: int = 20
):
    """
    Plot top important connections.
    
    Parameters
    ----------
    pattern_df : pd.DataFrame
        Connection pattern analysis
    save_path : Path
        Where to save figure
    n_top : int
        Number of top connections to show
    """
    
    # Sort by importance
    top_df = pattern_df.nlargest(n_top, 'importance')
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Color by network
    networks = top_df['network'].unique()
    network_colors = {net: plt.cm.tab10(i) for i, net in enumerate(networks)}
    colors = [network_colors[net] for net in top_df['network']]
    
    y_pos = np.arange(len(top_df))
    
    ax.barh(y_pos, top_df['importance'], color=colors, alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_df['region_name'], fontsize=8)
    ax.set_xlabel('Feature Importance', fontsize=12)
    ax.set_title(f'Top {n_top} Important Connections', fontsize=14, fontweight='bold')
    
    # Legend
    handles = [plt.Rectangle((0,0),1,1, color=network_colors[net]) 
               for net in networks]
    ax.legend(handles, networks, loc='lower right', fontsize=9)
    
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"✓ Saved: {save_path}")
    plt.close()


def plot_network_importance_distribution(
    pattern_df: pd.DataFrame,
    save_path: Path
):
    """
    Plot distribution of importance across networks.
    
    Parameters
    ----------
    pattern_df : pd.DataFrame
        Connection pattern analysis
    save_path : Path
        Where to save figure
    """
    
    # Aggregate by network
    network_importance = pattern_df.groupby('network').agg({
        'importance': ['mean', 'sum', 'count']
    }).reset_index()
    
    network_importance.columns = ['network', 'mean_importance', 'total_importance', 'n_connections']
    network_importance = network_importance.sort_values('total_importance', ascending=False)
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # Panel A: Total importance
    ax = axes[0]
    ax.bar(range(len(network_importance)), network_importance['total_importance'], alpha=0.7, color='teal')
    ax.set_xticks(range(len(network_importance)))
    ax.set_xticklabels(network_importance['network'], rotation=45, ha='right')
    ax.set_ylabel('Total Importance', fontsize=11)
    ax.set_title('A) Total Importance by Network', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Panel B: Mean importance
    ax = axes[1]
    ax.bar(range(len(network_importance)), network_importance['mean_importance'], alpha=0.7, color='coral')
    ax.set_xticks(range(len(network_importance)))
    ax.set_xticklabels(network_importance['network'], rotation=45, ha='right')
    ax.set_ylabel('Mean Importance', fontsize=11)
    ax.set_title('B) Mean Importance by Network', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Panel C: Number of connections
    ax = axes[2]
    ax.bar(range(len(network_importance)), network_importance['n_connections'], alpha=0.7, color='steelblue')
    ax.set_xticks(range(len(network_importance)))
    ax.set_xticklabels(network_importance['network'], rotation=45, ha='right')
    ax.set_ylabel('Number of Connections', fontsize=11)
    ax.set_title('C) Important Connections by Network', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"✓ Saved: {save_path}")
    plt.close()


def plot_coefficient_distribution_comparison(
    all_results: dict,
    save_path: Path,
    scope: str = 'full'
):
    """
    Plot coefficient distributions across strategies.
    
    Parameters
    ----------
    all_results : dict
        All model results
    save_path : Path
        Where to save figure
    scope : str
        Scope to analyze
    """
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    strategies = ['multinomial', 'ovr', 'ovo']
    colors = ['steelblue', 'coral', 'teal']
    
    for i, (strategy, color) in enumerate(zip(strategies, colors)):
        model_name = f'{scope}_{strategy}'
        
        if model_name not in all_results or 'final_model' not in all_results[model_name]:
            continue
        
        model = all_results[model_name]['final_model']
        coefs = model.coef_.flatten()
        
        ax = axes[i]
        
        # Histogram
        ax.hist(coefs, bins=50, alpha=0.7, color=color, edgecolor='black')
        ax.axvline(0, color='red', linestyle='--', linewidth=1, label='Zero')
        ax.set_xlabel('Coefficient Value', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title(f'{strategy.capitalize()}', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Add statistics
        stats_text = f'Mean: {coefs.mean():.4f}\nStd: {coefs.std():.4f}\nMax: {coefs.max():.4f}\nMin: {coefs.min():.4f}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    plt.suptitle(f'Coefficient Distributions ({scope.capitalize()} Scope)', 
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"✓ Saved: {save_path}")
    plt.close()


def main():
    """Run feature importance analysis."""
    
    logger.info("="*80)
    logger.info("FEATURE IMPORTANCE ANALYSIS")
    logger.info("="*80)
    
    # Load data
    loader = DataLoader()
    all_results = loader.load_all()
    
    logger.info(f"Loaded {len(all_results)} models")
    
    # Create output directories
    output_dir = project_root / 'outputs' / 'compiled'
    figures_dir = project_root / 'outputs' / 'figures'
    
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # Analyze full_multinomial as primary example
    model_name = 'full_multinomial'
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Analyzing {model_name.upper()}")
    logger.info(f"{'='*60}")
    
    if model_name not in all_results:
        logger.error(f"Model {model_name} not found!")
        return
    
    results = all_results[model_name]
    
    # Check if model is available
    if 'final_model' not in results:
        logger.error("Final model not saved. Please re-run training with --save_models flag")
        logger.info("\nAlternative: Analyzing coefficient statistics from available data...")
        
        # Can still analyze some patterns from predictions/probabilities
        logger.info("Limited analysis mode - using prediction patterns instead")
        return
    
    model = results['final_model']
    region_info = results['per_region_metrics'][['region_name', 'network']].copy()
    region_info['region_idx'] = range(len(region_info))
    
    # Add hemisphere info
    hemispheres = []
    for region in region_info['region_name']:
        if region.startswith('LH_') or region.endswith('-lh'):
            hemispheres.append('left')
        elif region.startswith('RH_') or region.endswith('-rh'):
            hemispheres.append('right')
        else:
            hemispheres.append('unknown')
    region_info['hemisphere'] = hemispheres
    
    feature_names = region_info['region_name'].tolist()
    
    # Extract coefficients
    logger.info("\n1. Extracting coefficients...")
    coef_df = extract_model_coefficients(model, feature_names)
    coef_df.to_csv(output_dir / f'{model_name}_coefficients.csv', index=False)
    logger.info(f"✓ Saved coefficients ({len(coef_df)} entries)")
    
    # Compute feature importance (need X data)
    # For now, just use coefficient magnitudes
    logger.info("\n2. Computing feature importance...")
    
    # Get top features per class
    top_features = get_top_features_per_class(coef_df, n_top=10)
    top_features.to_csv(output_dir / f'{model_name}_top_features_per_class.csv', index=False)
    logger.info(f"✓ Saved top features per class")
    
    # Analyze connection patterns
    logger.info("\n3. Analyzing connection patterns...")
    pattern_df = analyze_connection_patterns(region_info, coef_df, top_n=50)
    pattern_df.to_csv(output_dir / f'{model_name}_connection_patterns.csv', index=False)
    logger.info(f"✓ Saved connection patterns")
    
    # Generate visualizations
    logger.info("\n4. Generating visualizations...")
    
    # Coefficient heatmap
    plot_coefficient_heatmap(
        coef_df,
        region_info,
        save_path=figures_dir / f'{model_name}_coefficient_heatmap.png',
        n_regions=50,
        n_classes=50
    )
    
    # Top connections
    plot_top_connections(
        pattern_df,
        save_path=figures_dir / f'{model_name}_top_connections.png',
        n_top=20
    )
    
    # Network importance
    plot_network_importance_distribution(
        pattern_df,
        save_path=figures_dir / f'{model_name}_network_importance.png'
    )
    
    # Compare strategies
    logger.info("\n5. Comparing coefficient distributions across strategies...")
    
    for scope in ['full', 'left', 'right']:
        comparison_df = compare_coefficient_distributions(all_results, scope=scope)
        comparison_df.to_csv(output_dir / f'coefficient_comparison_{scope}.csv', index=False)
        logger.info(f"✓ Saved {scope} scope comparison")
        
        # Plot
        plot_coefficient_distribution_comparison(
            all_results,
            save_path=figures_dir / f'coefficient_distributions_{scope}.png',
            scope=scope
        )
    
    # Summary statistics
    logger.info("\n" + "="*80)
    logger.info("FEATURE IMPORTANCE SUMMARY")
    logger.info("="*80)
    
    logger.info(f"\nModel: {model_name}")
    logger.info(f"  Total coefficients: {len(coef_df)}")
    logger.info(f"  Mean |coefficient|: {coef_df['abs_coefficient'].mean():.6f}")
    logger.info(f"  Max |coefficient|: {coef_df['abs_coefficient'].max():.6f}")
    logger.info(f"  Sparsity (|coef| < 0.01): {(coef_df['abs_coefficient'] < 0.01).sum() / len(coef_df) * 100:.2f}%")
    
    logger.info("\nTop 5 most important connections:")
    top_5 = pattern_df.nlargest(5, 'abs_coefficient')
    for _, row in top_5.iterrows():
        logger.info(f"  • {row['region_name']} ({row['network']}): coef={row['coefficient']:.6f}")
    
    logger.info("\n" + "="*80)
    logger.info("FEATURE IMPORTANCE ANALYSIS COMPLETE")
    logger.info("="*80)
    logger.info(f"\nResults saved to:")
    logger.info(f"  • Data: {output_dir}")
    logger.info(f"  • Figures: {figures_dir}")


if __name__ == "__main__":
    main()