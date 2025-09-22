import os
import pandas as pd
import itertools
from scipy.stats import spearmanr, kendalltau
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import re
from collections import defaultdict

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
        "rbo": rbo_score,
        "common_neurons": common,
        "n_common": len(common)
    }

def parse_config_path(path):
    """Parse configuration details from file path."""
    parts = path.parts
    config = {}
    
    if "identification" not in parts:
        return None
        
    idx = parts.index("identification")
    
    # Parse model
    config['model'] = parts[idx + 1] if idx + 1 < len(parts) else "NA"
    
    # Parse method
    config['method'] = parts[idx + 2] if idx + 2 < len(parts) else "NA"
    
    # Parse layer
    config['layer'] = parts[idx + 3] if idx + 3 < len(parts) else "NA"
    
    # Parse dataset configuration (contains dataset, threshold, shuffle info)
    if idx + 4 < len(parts):
        dataset_config = parts[idx + 4]
        config['dataset_config'] = dataset_config
        
        # Parse dataset name
        if 'europarl' in dataset_config:
            config['dataset'] = 'europarl'
        elif 'flores_plus' in dataset_config:
            config['dataset'] = 'flores_plus'
        elif 'jw300' in dataset_config:
            config['dataset'] = 'jw300'
        else:
            config['dataset'] = dataset_config
        
        # Parse threshold
        thresh_match = re.search(r'thresh-(\d+)', dataset_config)
        config['threshold'] = thresh_match.group(1) if thresh_match else 'NA'
        
        # Parse shuffle status
        config['shuffle'] = 'shuffled' if 'shuffle' in dataset_config else 'original'
        
        # Parse if it's scratch
        config['is_scratch'] = 'scratch' in dataset_config
        
        # Parse special configurations (e.g., en-de-ja-ko)
        if 'en-de-ja-ko' in dataset_config:
            config['lang_subset'] = 'en-de-ja-ko'
        else:
            config['lang_subset'] = 'all'
    
    # Parse split
    config['split'] = parts[idx + 5] if idx + 5 < len(parts) else "NA"
    
    # Parse language
    config['lang'] = path.stem
    
    return config

def collect_files(base_dir, include_words=None, exclude_words=None, layers=None):
    csv_files = list(Path(base_dir).rglob("*.csv"))
    entries = []

    for f in csv_files:
        config = parse_config_path(f)
        if not config:
            continue
        
        # Apply layer filter
        if layers and config['layer'] not in layers:
            continue
        
        # Apply include/exclude filters
        config_str = f"{config['method']}/{config['dataset_config']}/{config['split']}"
        
        if include_words and not any(word in config_str for word in include_words):
            continue
        
        if exclude_words and any(word in config_str for word in exclude_words):
            continue
        
        entries.append((config, f))
    
    return entries

def create_language_heatmap(df_lang, layer_name, lang_name, output_dir, metric="spearman"):
    """Create a heatmap for a specific language and layer showing similarity between configs."""
    if df_lang.empty:
        return
    
    configs = sorted(set(df_lang['config1'].unique()) | set(df_lang['config2'].unique()))
    
    if len(configs) < 2:
        return
    
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
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
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
    
    print(f"    Saved {metric} heatmap to {plot_path}")

def analyze_dimension_overlap(entries, dimension, output_dir, k=50):
    """Analyze neuron overlap across a specific dimension (method, dataset, threshold, etc.)"""
    print(f"\n=== Analyzing dimension: {dimension} ===")
    
    # Group entries by the dimension
    dim_groups = defaultdict(list)
    for config, path in entries:
        if dimension in config:
            dim_value = config[dimension]
            dim_groups[dim_value].append((config, path))
    
    if len(dim_groups) < 2:
        print(f"  Not enough values for dimension {dimension} to compare")
        return None
    
    # Prepare results storage
    overlap_results = []
    
    # For each layer and language, compare across dimension values
    layer_lang_groups = defaultdict(lambda: defaultdict(list))
    for dim_value, dim_entries in dim_groups.items():
        for config, path in dim_entries:
            key = (config['layer'], config['lang'])
            layer_lang_groups[key][dim_value].append((config, path))
    
    # Analyze each layer-language combination
    for (layer, lang), dim_data in layer_lang_groups.items():
        if len(dim_data) < 2:
            continue
            
        print(f"\n  Layer: {layer}, Language: {lang}")
        
        # Load neuron lists for each dimension value
        dim_lists = {}
        for dim_value, entries in dim_data.items():
            if entries:  # Take first entry if multiple
                config, path = entries[0]
                neuron_list = load_ranked_list(path)
                dim_lists[dim_value] = neuron_list
                print(f"    {dim_value}: {len(neuron_list)} neurons")
        
        # Compare all pairs
        for (val1, list1), (val2, list2) in itertools.combinations(dim_lists.items(), 2):
            comparison = compare_lists(list1, list2, k=k)
            
            result = {
                'layer': layer,
                'lang': lang,
                'dimension': dimension,
                f'{dimension}_1': val1,
                f'{dimension}_2': val2,
                **comparison
            }
            overlap_results.append(result)
            
            # Print key findings
            print(f"    {val1} vs {val2}:")
            print(f"      Jaccard@{k}: {comparison['jaccard@k']:.3f}")
            print(f"      Precision@{k}: {comparison['precision@k']:.3f}")
            print(f"      Common neurons: {comparison['n_common']}")
    
    if not overlap_results:
        print(f"  No comparisons possible for dimension {dimension}")
        return None
    
    # Create DataFrame and save results
    df_results = pd.DataFrame(overlap_results)
    
    # Save to CSV
    csv_path = output_dir / f"dimension_analysis_{dimension}.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"\n  Saved dimension analysis to {csv_path}")
    
    # Create visualizations
    create_dimension_plots(df_results, dimension, output_dir)
    
    return df_results

def create_dimension_plots(df_results, dimension, output_dir):
    """Create plots for dimension analysis."""
    
    # 1. Average overlap by dimension value
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    metrics = ['jaccard@k', 'precision@k', 'rbo']
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        # Calculate average metric for each dimension value
        avg_by_val = []
        dim_values = set(df_results[f'{dimension}_1'].unique()) | set(df_results[f'{dimension}_2'].unique())
        
        for val in dim_values:
            mask = (df_results[f'{dimension}_1'] == val) | (df_results[f'{dimension}_2'] == val)
            if mask.any():
                avg_val = df_results.loc[mask, metric].mean()
                avg_by_val.append({'value': val, 'avg': avg_val})
        
        if avg_by_val:
            df_avg = pd.DataFrame(avg_by_val).sort_values('avg')
            bars = ax.bar(range(len(df_avg)), df_avg['avg'])
            ax.set_xticks(range(len(df_avg)))
            ax.set_xticklabels(df_avg['value'], rotation=45, ha='right')
            ax.set_title(f'Average {metric} by {dimension}')
            ax.set_ylabel('Score')
            ax.set_ylim([0, 1])
            ax.grid(True, alpha=0.3)
            
            # Add value labels
            for j, bar in enumerate(bars):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}', ha='center', va='bottom', fontsize=8)
    
    plt.suptitle(f'Overlap Analysis by {dimension}')
    plt.tight_layout()
    
    plot_path = output_dir / f"dimension_{dimension}_overview.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved dimension overview plot to {plot_path}")
    
    # 2. Heatmap of pairwise overlaps
    if len(df_results) > 0:
        # Group by layer and create heatmaps
        for layer in df_results['layer'].unique():
            df_layer = df_results[df_results['layer'] == layer]
            
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            
            for i, metric in enumerate(['jaccard@k', 'precision@k']):
                ax = axes[i]
                
                # Create pivot table for heatmap
                dim_vals = sorted(set(df_layer[f'{dimension}_1'].unique()) | 
                                 set(df_layer[f'{dimension}_2'].unique()))
                
                matrix = np.zeros((len(dim_vals), len(dim_vals)))
                val_to_idx = {v: i for i, v in enumerate(dim_vals)}
                
                for _, row in df_layer.iterrows():
                    i = val_to_idx[row[f'{dimension}_1']]
                    j = val_to_idx[row[f'{dimension}_2']]
                    matrix[i, j] = row[metric]
                    matrix[j, i] = row[metric]
                
                np.fill_diagonal(matrix, 1.0)
                
                sns.heatmap(matrix, 
                           annot=True, 
                           fmt='.2f',
                           cmap='RdYlBu_r',
                           vmin=0, 
                           vmax=1,
                           xticklabels=dim_vals,
                           yticklabels=dim_vals,
                           ax=ax,
                           cbar_kws={'label': metric})
                
                ax.set_title(f'{metric} - {layer}')
                ax.set_xlabel(dimension)
                ax.set_ylabel(dimension)
            
            plt.suptitle(f'Pairwise {dimension} Overlap - {layer}')
            plt.tight_layout()
            
            plot_path = output_dir / f"dimension_{dimension}_{layer}_heatmap.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  Saved {layer} heatmap to {plot_path}")

def analyze_neuron_intersection(entries, output_dir, k=50):
    """Analyze which specific neurons appear across multiple configurations."""
    print("\n=== Analyzing Neuron Intersections ===")
    
    # Group by layer and language
    layer_lang_groups = defaultdict(list)
    for config, path in entries:
        key = (config['layer'], config['lang'])
        layer_lang_groups[key].append((config, path))
    
    intersection_results = []
    
    for (layer, lang), group_entries in layer_lang_groups.items():
        if len(group_entries) < 2:
            continue
            
        print(f"\n  Layer: {layer}, Language: {lang}")
        
        # Load all neuron lists
        config_neurons = {}
        for config, path in group_entries:
            neuron_list = load_ranked_list(path)[:k]  # Top-k only
            config_str = f"{config['method']}-{config['dataset']}-{config['threshold']}-{config['shuffle']}"
            config_neurons[config_str] = set(neuron_list)
        
        if len(config_neurons) < 2:
            continue
        
        # Find core neurons (appear in all configs)
        all_configs = list(config_neurons.keys())
        core_neurons = set.intersection(*config_neurons.values())
        
        # Find neurons that appear in at least half of configs
        neuron_counts = defaultdict(int)
        for neurons in config_neurons.values():
            for n in neurons:
                neuron_counts[n] += 1
        
        half_threshold = len(config_neurons) / 2
        frequent_neurons = {n for n, count in neuron_counts.items() if count >= half_threshold}
        
        result = {
            'layer': layer,
            'lang': lang,
            'n_configs': len(config_neurons),
            'n_core_neurons': len(core_neurons),
            'n_frequent_neurons': len(frequent_neurons),
            'core_neurons': list(core_neurons)[:20],  # Store first 20 for reference
            'core_ratio': len(core_neurons) / k if k > 0 else 0,
            'frequent_ratio': len(frequent_neurons) / k if k > 0 else 0
        }
        intersection_results.append(result)
        
        print(f"    Configs analyzed: {len(config_neurons)}")
        print(f"    Core neurons (in all): {len(core_neurons)} ({result['core_ratio']:.1%})")
        print(f"    Frequent neurons (≥50%): {len(frequent_neurons)} ({result['frequent_ratio']:.1%})")
        
        if core_neurons:
            print(f"    Sample core neurons: {list(core_neurons)[:10]}")
    
    if intersection_results:
        df_intersections = pd.DataFrame(intersection_results)
        csv_path = output_dir / "neuron_intersections.csv"
        df_intersections.to_csv(csv_path, index=False)
        print(f"\n  Saved intersection analysis to {csv_path}")
        
        # Create visualization
        create_intersection_plots(df_intersections, output_dir)
        
        return df_intersections
    
    return None

def create_intersection_plots(df_intersections, output_dir):
    """Create plots for neuron intersection analysis."""
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Core vs Frequent neurons by layer
    ax = axes[0]
    layers = df_intersections.groupby('layer').agg({
        'core_ratio': 'mean',
        'frequent_ratio': 'mean'
    }).reset_index()
    
    x = np.arange(len(layers))
    width = 0.35
    
    ax.bar(x - width/2, layers['core_ratio'], width, label='Core (100%)', color='darkblue')
    ax.bar(x + width/2, layers['frequent_ratio'], width, label='Frequent (≥50%)', color='lightblue')
    
    ax.set_xlabel('Layer')
    ax.set_ylabel('Ratio of Top-k Neurons')
    ax.set_title('Neuron Consistency Across Configurations')
    ax.set_xticks(x)
    ax.set_xticklabels(layers['layer'], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Distribution by language
    ax = axes[1]
    langs = df_intersections.groupby('lang').agg({
        'core_ratio': 'mean',
        'frequent_ratio': 'mean',
        'n_configs': 'mean'
    }).reset_index().sort_values('core_ratio', ascending=False)
    
    # Only plot top 10 languages for clarity
    langs = langs.head(10)
    
    x = np.arange(len(langs))
    ax.bar(x - width/2, langs['core_ratio'], width, label='Core', color='darkgreen')
    ax.bar(x + width/2, langs['frequent_ratio'], width, label='Frequent', color='lightgreen')
    
    ax.set_xlabel('Language')
    ax.set_ylabel('Ratio of Top-k Neurons')
    ax.set_title('Top 10 Languages by Neuron Consistency')
    ax.set_xticks(x)
    ax.set_xticklabels(langs['lang'], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add config count as text
    for i, (idx, row) in enumerate(langs.iterrows()):
        ax.text(i, row['frequent_ratio'] + 0.02, f"n={row['n_configs']:.0f}", 
               ha='center', fontsize=8)
    
    plt.suptitle('Neuron Intersection Analysis')
    plt.tight_layout()
    
    plot_path = output_dir / "neuron_intersections.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved intersection plots to {plot_path}")

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
    
    # Ensure output directories exist
    out_dir = Path("analysis")
    out_dir.mkdir(exist_ok=True)
    
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    layer_dir = out_dir / "by_layer"
    layer_dir.mkdir(exist_ok=True)
    
    dimension_dir = out_dir / "by_dimension"
    dimension_dir.mkdir(exist_ok=True)
    
    # Group by layer
    layer_groups = {}
    for config, path in entries:
        layer = config['layer']
        layer_groups.setdefault(layer, []).append((config, path))
    
    # Process each layer
    all_summaries = []
    
    for layer, files in sorted(layer_groups.items()):
        print(f"\nProcessing {layer}...")
        
        # Group by language within this layer
        lang_groups = {}
        for config, path in files:
            lang = config['lang']
            config_str = f"{config['method']}/{config['dataset_config']}/{config['split']}"
            lang_groups.setdefault(lang, []).append((config_str, path))
        
        layer_results = []
        
        # Create layer-specific plot directory
        layer_plots_dir = plots_dir / layer
        layer_plots_dir.mkdir(exist_ok=True)
        
        # Process each language
        for lang, lang_files in sorted(lang_groups.items()):
            print(f"  Processing language: {lang}")
            
            lists = {}
            
            # Load lists
            for cfg, path in lang_files:
                neuron_list = load_ranked_list(path)
                lists[cfg] = neuron_list
            
            lang_results = []
            
            # Compare all pairs within this language and layer
            for (c1, p1), (c2, p2) in itertools.combinations(lang_files, 2):
                metrics = compare_lists(lists[c1], lists[c2], k=k)
                row = {
                    "lang": lang,
                    "layer": layer,
                    "config1": c1,
                    "config2": c2,
                    **{k: v for k, v in metrics.items() if k not in ['common_neurons']}
                }
                layer_results.append(row)
                lang_results.append(row)
            
            # Create language-specific plots if there are comparisons
            if lang_results:
                df_lang = pd.DataFrame(lang_results)
                
                # Create language-specific subdirectory
                lang_plots_dir = layer_plots_dir / lang
                lang_plots_dir.mkdir(exist_ok=True)
                
                # Generate heatmaps including Jaccard
                create_language_heatmap(df_lang, layer, lang, lang_plots_dir, 'spearman')
                create_language_heatmap(df_lang, layer, lang, lang_plots_dir, 'jaccard@k')  # Added Jaccard heatmap
                create_language_heatmap(df_lang, layer, lang, lang_plots_dir, 'precision@k')
                create_language_heatmap(df_lang, layer, lang, lang_plots_dir, 'rbo')
                
                # Save language-specific comparison CSV
                lang_file = layer_dir / f"{layer}_{lang}_comparisons.csv"
                df_lang.to_csv(lang_file, index=False)
                print(f"    Saved {len(lang_results)} comparisons to {lang_file}")
    
    # Analyze different dimensions
    dimensions_to_analyze = ['method', 'dataset', 'threshold', 'shuffle', 'lang_subset']
    
    for dimension in dimensions_to_analyze:
        df_dim = analyze_dimension_overlap(entries, dimension, dimension_dir, k=k)
    
    # Analyze neuron intersections
    analyze_neuron_intersection(entries, out_dir, k=k)
    
    print("\n=== Analysis Summary ===")
    print(f"Completed analysis for:")
    print(f"  - {len(layer_groups)} layers")
    print(f"  - {len(set(config['lang'] for config, _ in entries))} languages")
    print(f"  - Dimension analyses: {', '.join(dimensions_to_analyze)}")
    print(f"\nResults saved in {out_dir}")
    
    return True

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, default="./identification", 
                       help="Base directory containing identification files")
    parser.add_argument("--k", type=int, default=50, 
                       help="Top-k overlap size")
    parser.add_argument("--layers", nargs='+', default=None,
                       help="Specific layers to process")
    parser.add_argument("--include", type=str, nargs='+', default=["jw300", "europarl", "flores_plus"],
                       help="Include configs containing any of these words")
    parser.add_argument("--exclude", type=str, nargs='+', default=["scratch"],
                       help="Exclude configs containing any of these words")
    args = parser.parse_args()

    main(args.base_dir, k=args.k, include_words=args.include, exclude_words=args.exclude, layers=args.layers)