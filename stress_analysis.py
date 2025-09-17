import os
import pandas as pd
import itertools
from scipy.stats import spearmanr, kendalltau
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle

def load_ranked_list(path):
    """Load ranked list from CSV by taking the first column in order."""
    df = pd.read_csv(path)
    first_col = df.columns[0]
    return df[first_col].tolist()

def topk_overlap(list1, list2, k=50):
    k_eff = min(k, len(list1), len(list2))
    if k_eff == 0:
        return {"precision": np.nan, "jaccard": np.nan}
    set1, set2 = set(list1[:k_eff]), set(list2[:k_eff])
    inter = len(set1 & set2)
    union = len(set1 | set2)
    return {
        "precision": inter / k_eff,
        "jaccard": inter / union if union > 0 else np.nan
    }

def rbo(list1, list2, p=0.9):
    """Rank-biased overlap (Webber et al. 2010)."""
    s, l = sorted([list1, list2], key=len)
    depth = min(len(s), len(l))
    if depth == 0:
        return np.nan
    rbo_ext = 0.0
    for d in range(1, depth + 1):
        overlap = len(set(s[:d]) & set(l[:d]))
        rbo_ext += overlap / d * (p ** (d - 1))
    return (1 - p) * rbo_ext

def compare_lists(list1, list2, k=50):
    rank1 = {f: i for i, f in enumerate(list1)}
    rank2 = {f: i for i, f in enumerate(list2)}
    common = list(set(rank1) & set(rank2))

    if len(common) > 1:
        r1 = [rank1[f] for f in common]
        r2 = [rank2[f] for f in common]
        spearman = spearmanr(r1, r2).correlation
        kendall = kendalltau(r1, r2).correlation
    else:
        spearman, kendall = np.nan, np.nan

    overlap = topk_overlap(list1, list2, k=k)
    rbo_score = rbo(list1, list2)

    return {
        "spearman": spearman,
        "kendall": kendall,
        "precision@k": overlap["precision"],
        "jaccard@k": overlap["jaccard"],
        "rbo": rbo_score
    }

def collect_files(base_dir, include_words=None, exclude_words=None, layers=None):
    csv_files = list(Path(base_dir).rglob("*.csv"))
    entries = []

    for f in csv_files:
        parts = f.parts
        if "identification" not in parts:
            continue

        idx = parts.index("identification")

        model = parts[idx + 1] if idx + 1 < len(parts) else "NA"
        method = parts[idx + 2] if idx + 2 < len(parts) else "NA"
        layer = parts[idx + 3] if idx + 3 < len(parts) else "NA"
        dataset = parts[idx + 4] if idx + 4 < len(parts) else "NA"
        split = parts[idx + 5] if idx + 5 < len(parts) else "NA"

        lang = f.stem
        config = f"{method}/{dataset}/{split}"
        
        # Apply layer filter if specified
        if layers:
            if layer not in layers:
                continue
        
        # Apply include filter if specified
        if include_words:
            if not any(word in config for word in include_words):
                continue
        
        # Apply exclude filter if specified
        if exclude_words:
            if any(word in config for word in exclude_words):
                continue
        
        entries.append((lang, layer, config, f))

    return entries

def create_language_heatmap(df_lang, layer_name, lang_name, output_dir, metric="spearman"):
    """Create a heatmap for a specific language and layer showing similarity between configs."""
    if df_lang.empty:
        return
    
    # Get unique configs
    configs = sorted(set(df_lang['config1'].unique()) | set(df_lang['config2'].unique()))
    
    if len(configs) < 2:
        return
    
    # Create matrix
    n = len(configs)
    matrix = np.ones((n, n))
    
    config_to_idx = {c: i for i, c in enumerate(configs)}
    
    for _, row in df_lang.iterrows():
        i = config_to_idx[row['config1']]
        j = config_to_idx[row['config2']]
        val = row[metric]
        if not np.isnan(val):
            matrix[i, j] = val
            matrix[j, i] = val
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Use mask for upper triangle
    mask = np.triu(np.ones_like(matrix, dtype=bool), k=1)
    
    sns.heatmap(matrix, 
                mask=mask,
                annot=True, 
                fmt='.2f',
                cmap='RdYlBu_r',
                vmin=0, 
                vmax=1,
                square=True,
                xticklabels=configs,
                yticklabels=configs,
                cbar_kws={'label': f'{metric.capitalize()} Correlation'})
    
    plt.title(f'Ranking Similarity - {layer_name} - {lang_name} ({metric})')
    plt.xlabel('Configuration')
    plt.ylabel('Configuration')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    plot_path = output_dir / f"{layer_name}_{lang_name}_{metric}_heatmap.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"    Saved heatmap to {plot_path}")

def create_language_boxplot(df_lang, layer_name, lang_name, output_dir):
    """Create boxplots for all metrics for a specific language and layer."""
    if df_lang.empty:
        return
    
    metrics = ['spearman', 'kendall', 'precision@k', 'jaccard@k', 'rbo']
    
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    
    for i, metric in enumerate(metrics):
        data = df_lang[metric].dropna()
        if len(data) > 0:
            axes[i].boxplot(data)
            axes[i].set_title(metric.capitalize())
            axes[i].set_ylabel('Score')
            axes[i].set_ylim([0, 1])
            axes[i].grid(True, alpha=0.3)
            
            # Add mean line
            mean_val = data.mean()
            axes[i].axhline(y=mean_val, color='r', linestyle='--', alpha=0.5, label=f'Mean: {mean_val:.3f}')
            axes[i].legend()
    
    plt.suptitle(f'Metric Distributions - {layer_name} - {lang_name}')
    plt.tight_layout()
    
    plot_path = output_dir / f"{layer_name}_{lang_name}_metrics_boxplot.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"    Saved boxplot to {plot_path}")

def create_language_pairwise_plot(df_lang, layer_name, lang_name, output_dir):
    """Create a scatter plot matrix showing pairwise metric relationships."""
    if df_lang.empty:
        return
    
    metrics = ['spearman', 'kendall', 'precision@k', 'jaccard@k', 'rbo']
    df_metrics = df_lang[metrics].dropna()
    
    if df_metrics.empty or len(df_metrics) < 2:
        return
    
    # Create scatter plot matrix
    fig, axes = plt.subplots(5, 5, figsize=(15, 15))
    
    for i, metric1 in enumerate(metrics):
        for j, metric2 in enumerate(metrics):
            ax = axes[i, j]
            if i == j:
                # Diagonal: histogram
                ax.hist(df_metrics[metric1], bins=20, edgecolor='black', alpha=0.7)
                if i == 0:
                    ax.set_title(metric1.replace('@k', ''))
            else:
                # Off-diagonal: scatter plot
                ax.scatter(df_metrics[metric2], df_metrics[metric1], alpha=0.6)
                if not df_metrics[metric2].isna().all() and not df_metrics[metric1].isna().all():
                    corr = df_metrics[metric2].corr(df_metrics[metric1])
                    ax.text(0.05, 0.95, f'r={corr:.2f}', transform=ax.transAxes, 
                           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            # Labels
            if j == 0:
                ax.set_ylabel(metric1.replace('@k', ''))
            if i == 4:
                ax.set_xlabel(metric2.replace('@k', ''))
            
            ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'Metric Relationships - {layer_name} - {lang_name}')
    plt.tight_layout()
    
    plot_path = output_dir / f"{layer_name}_{lang_name}_metric_relationships.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"    Saved metric relationships plot to {plot_path}")

def create_config_comparison_bar(df_lang, layer_name, lang_name, output_dir):
    """Create bar plot comparing average metrics across configs."""
    if df_lang.empty:
        return
    
    # Calculate average metrics for each config
    configs = list(set(df_lang['config1'].unique()) | set(df_lang['config2'].unique()))
    
    if len(configs) < 2:
        return
    
    config_metrics = []
    for config in configs:
        # Get all rows where this config appears
        config_rows = df_lang[(df_lang['config1'] == config) | (df_lang['config2'] == config)]
        if not config_rows.empty:
            avg_metrics = config_rows[['spearman', 'kendall', 'precision@k', 'jaccard@k', 'rbo']].mean()
            avg_metrics['config'] = config
            config_metrics.append(avg_metrics)
    
    if not config_metrics:
        return
    
    df_config_metrics = pd.DataFrame(config_metrics)
    df_config_metrics = df_config_metrics.set_index('config')
    
    # Create bar plot
    fig, ax = plt.subplots(figsize=(14, 6))
    df_config_metrics.plot(kind='bar', ax=ax)
    ax.set_title(f'Average Metrics by Configuration - {layer_name} - {lang_name}')
    ax.set_xlabel('Configuration')
    ax.set_ylabel('Score')
    ax.set_ylim([0, 1])
    ax.legend(title='Metrics', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    plot_path = output_dir / f"{layer_name}_{lang_name}_config_comparison.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"    Saved config comparison to {plot_path}")

def create_layer_summary_plot(layer_summaries, layer_name, output_dir):
    """Create a summary plot comparing all languages for a layer."""
    if not layer_summaries:
        return
    
    df_summary = pd.DataFrame(layer_summaries)
    
    if df_summary.empty:
        return
    
    # Create subplot for each metric
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    metrics = ['consistency_spearman', 'consistency_kendall', 'consistency_precision@k',
              'consistency_jaccard@k', 'consistency_rbo']
    
    for i, metric in enumerate(metrics):
        ax = axes[i // 3, i % 3]
        
        # Sort languages by metric value for better visualization
        sorted_df = df_summary.sort_values(metric)
        
        bars = ax.bar(range(len(sorted_df)), sorted_df[metric])
        ax.set_xticks(range(len(sorted_df)))
        ax.set_xticklabels(sorted_df['lang'], rotation=45, ha='right')
        ax.set_title(metric.replace('consistency_', '').replace('@k', '').capitalize())
        ax.set_ylabel('Score')
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3)
        
        # Color bars by value
        for j, bar in enumerate(bars):
            val = sorted_df[metric].iloc[j]
            if val >= 0.8:
                bar.set_color('green')
            elif val >= 0.6:
                bar.set_color('yellow')
            else:
                bar.set_color('red')
        
        # Add value labels on bars
        for j, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}', ha='center', va='bottom', fontsize=8)
    
    # Add info box in the 6th subplot
    ax = axes[1, 2]
    ax.axis('off')
    info_text = f"Layer: {layer_name}\n"
    info_text += f"Total languages: {len(df_summary)}\n"
    info_text += f"Total comparisons: {df_summary['n_comparisons'].sum()}\n"
    info_text += f"Avg configs per language: {df_summary['n_configs'].mean():.1f}"
    ax.text(0.5, 0.5, info_text, transform=ax.transAxes,
           fontsize=12, ha='center', va='center',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle(f'Language Consistency Summary - {layer_name}')
    plt.tight_layout()
    
    plot_path = output_dir / f"{layer_name}_all_languages_summary.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved layer summary plot to {plot_path}")

def main(base_dir, k=50, include_words=None, exclude_words=None, layers=None):
    entries = collect_files(base_dir, include_words, exclude_words, layers)
    
    if not entries:
        print("No files found matching the criteria.")
        return
    
    # Print filter info
    if layers:
        print(f"Processing only layers: {layers}")
    if include_words:
        print(f"Including configs containing any of: {include_words}")
    if exclude_words:
        print(f"Excluding configs containing any of: {exclude_words}")
    
    print(f"Found {len(entries)} files after filtering")
    
    # Group by layer
    layer_groups = {}
    for lang, layer, config, path in entries:
        layer_groups.setdefault(layer, []).append((lang, config, path))
    
    # Ensure output directories exist
    out_dir = Path("analysis")
    out_dir.mkdir(exist_ok=True)
    
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    layer_dir = out_dir / "by_layer"
    layer_dir.mkdir(exist_ok=True)
    
    # Process each layer separately
    all_summaries = []
    
    for layer, files in sorted(layer_groups.items()):
        print(f"\nProcessing {layer}...")
        
        # Group by language within this layer
        lang_groups = {}
        for lang, config, path in files:
            lang_groups.setdefault(lang, []).append((config, path))
        
        layer_results = []
        layer_summary = []
        
        # Create layer-specific plot directory
        layer_plots_dir = plots_dir / layer
        layer_plots_dir.mkdir(exist_ok=True)
        
        # Process each language
        for lang, lang_files in sorted(lang_groups.items()):
            print(f"  Processing language: {lang}")
            
            lists = {cfg: load_ranked_list(path) for cfg, path in lang_files}
            
            lang_results = []
            
            # Compare all pairs within this language and layer
            for (c1, p1), (c2, p2) in itertools.combinations(lang_files, 2):
                metrics = compare_lists(lists[c1], lists[c2], k=k)
                row = {
                    "lang": lang,
                    "layer": layer,
                    "config1": c1,
                    "config2": c2,
                    **metrics
                }
                layer_results.append(row)
                lang_results.append(row)
            
            # Create language-specific plots if there are comparisons
            if lang_results:
                df_lang = pd.DataFrame(lang_results)
                
                # Create language-specific subdirectory
                lang_plots_dir = layer_plots_dir / lang
                lang_plots_dir.mkdir(exist_ok=True)
                
                # Generate language-specific plots
                create_language_heatmap(df_lang, layer, lang, lang_plots_dir, 'spearman')
                create_language_heatmap(df_lang, layer, lang, lang_plots_dir, 'precision@k')
                create_language_heatmap(df_lang, layer, lang, lang_plots_dir, 'rbo')
                create_language_boxplot(df_lang, layer, lang, lang_plots_dir)
                create_language_pairwise_plot(df_lang, layer, lang, lang_plots_dir)
                create_config_comparison_bar(df_lang, layer, lang, lang_plots_dir)
                
                # Save language-specific comparison CSV
                lang_file = layer_dir / f"{layer}_{lang}_comparisons.csv"
                df_lang.to_csv(lang_file, index=False)
                print(f"    Saved {len(lang_results)} comparisons to {lang_file}")
            
            # Calculate summary for this language-layer combination
            if len(lang_files) > 1 and lang_results:
                df_ll = pd.DataFrame(lang_results)
                summary_row = {
                    "lang": lang,
                    "layer": layer,
                    "n_configs": len(lang_files),
                    "n_comparisons": len(lang_results),
                    "consistency_spearman": df_ll["spearman"].mean(),
                    "consistency_kendall": df_ll["kendall"].mean(),
                    "consistency_precision@k": df_ll["precision@k"].mean(),
                    "consistency_jaccard@k": df_ll["jaccard@k"].mean(),
                    "consistency_rbo": df_ll["rbo"].mean(),
                    "std_spearman": df_ll["spearman"].std(),
                    "std_kendall": df_ll["kendall"].std(),
                    "std_precision@k": df_ll["precision@k"].std(),
                    "std_jaccard@k": df_ll["jaccard@k"].std(),
                    "std_rbo": df_ll["rbo"].std()
                }
                layer_summary.append(summary_row)
                all_summaries.append(summary_row)
        
        # Save layer-specific results (all languages combined)
        if layer_results:
            df_layer = pd.DataFrame(layer_results)
            layer_file = layer_dir / f"{layer}_all_comparisons.csv"
            df_layer.to_csv(layer_file, index=False)
            print(f"  Saved total {len(layer_results)} comparisons to {layer_file}")
        
        # Create layer summary plot comparing all languages
        if layer_summary:
            create_layer_summary_plot(layer_summary, layer, layer_plots_dir)
            
            # Save layer summary
            df_layer_summary = pd.DataFrame(layer_summary)
            summary_file = layer_dir / f"{layer}_summary.csv"
            df_layer_summary.to_csv(summary_file, index=False)
            print(f"  Saved summary to {summary_file}")
    
    # Save overall summary
    if all_summaries:
        df_all_summary = pd.DataFrame(all_summaries)
        df_all_summary.to_csv(out_dir / "all_layers_summary.csv", index=False)
        print(f"\nSaved overall summary to {out_dir / 'all_layers_summary.csv'}")
        
        # Create cross-layer comparison plot
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        metrics = ['consistency_spearman', 'consistency_kendall', 'consistency_precision@k', 
                  'consistency_jaccard@k', 'consistency_rbo']
        
        for i, metric in enumerate(metrics):
            ax = axes[i // 3, i % 3]
            pivot = df_all_summary.pivot_table(index='layer', columns='lang', values=metric)
            pivot.plot(kind='bar', ax=ax)
            ax.set_title(metric.replace('consistency_', '').replace('@k', '').capitalize())
            ax.set_xlabel('Layer')
            ax.set_ylabel('Score')
            ax.legend(title='Language')
            ax.grid(True, alpha=0.3)
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Remove the extra subplot
        fig.delaxes(axes[1, 2])
        
        plt.suptitle('Consistency Metrics Across Layers and Languages')
        plt.tight_layout()
        
        cross_layer_plot = plots_dir / "cross_layer_comparison.png"
        plt.savefig(cross_layer_plot, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved cross-layer comparison plot to {cross_layer_plot}")
        
        # Create a heatmap showing all layers and languages
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for i, metric in enumerate(['consistency_spearman', 'consistency_precision@k', 'consistency_rbo']):
            ax = axes[i]
            pivot = df_all_summary.pivot_table(index='layer', columns='lang', values=metric)
            sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdYlBu_r', 
                       vmin=0, vmax=1, ax=ax, cbar_kws={'label': 'Score'})
            ax.set_title(metric.replace('consistency_', '').replace('@k', '').capitalize())
            ax.set_xlabel('Language')
            ax.set_ylabel('Layer')
        
        plt.suptitle('Consistency Heatmap Across Layers and Languages')
        plt.tight_layout()
        
        heatmap_plot = plots_dir / "layer_language_heatmap.png"
        plt.savefig(heatmap_plot, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved layer-language heatmap to {heatmap_plot}")
    
    print("\nAnalysis complete!")
    
    # Print summary statistics
    if all_summaries:
        df = pd.DataFrame(all_summaries)
        print("\nOverall Statistics:")
        print(f"  Total layers analyzed: {df['layer'].nunique()}")
        print(f"  Total languages analyzed: {df['lang'].nunique()}")
        print(f"  Total comparisons: {df['n_comparisons'].sum()}")
        print(f"  Average consistency (Spearman): {df['consistency_spearman'].mean():.3f}")
        print(f"  Average consistency (Precision@k): {df['consistency_precision@k'].mean():.3f}")
        print(f"  Average consistency (RBO): {df['consistency_rbo'].mean():.3f}")
    
    return df_all_summary if all_summaries else None

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, default="./identification", 
                       help="Base directory containing identification/*/*/layer_x/<dataset>/<split>/*.csv files")
    parser.add_argument("--k", type=int, default=50, 
                       help="Top-k overlap size")
    parser.add_argument("--layers", nargs='+', default=None,
                       help="Specific layers to process (e.g., --layers layer_0 layer_5 layer_10)")
    parser.add_argument("--include", type=str, nargs='+', default=["jw300", "europarl", "flores_plus"],
                       help="Include configs containing any of these words (e.g., --include probe mlp)")
    parser.add_argument("--exclude", type=str, nargs='+', default=["scratch"],
                       help="Exclude configs containing any of these words (e.g., --exclude test debug)")
    args = parser.parse_args()

    main(args.base_dir, k=args.k, include_words=args.include, exclude_words=args.exclude, layers=args.layers)