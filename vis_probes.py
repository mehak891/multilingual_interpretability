#!/usr/bin/env python3
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

def get_args():
    parser = argparse.ArgumentParser(description="Create heatmaps from probing results")
    parser.add_argument("--results_dir", type=str, required=True,
                        help="Directory name x (inside results/) containing *_sources.csv files")
    parser.add_argument("--output_root", type=str, default="lang2vec_probing/visualizations",
                        help="Root directory for visualizations")
    parser.add_argument("--r2_threshold", type=float, default=0.5,
                        help="Minimum R² score to consider significant (default: 0.5)")
    parser.add_argument("--figsize", nargs=2, type=int, default=[24, 12],
                        help="Figure size (width height) in inches")
    parser.add_argument("--top_k_features", type=int, default=5,
                        help="Number of top features to show per neuron (default: 5)")
    return parser.parse_args()

def aggregate_features(df, layer):
    """Aggregate max and avg R² scores for each neuron-language pair per feature_set."""
    df = df[df["layer"] == layer]
    if df.empty:
        return None, None

    records = []
    for (neuron, langs, feat_set), group in df.groupby(["neuron_idx", "source_languages", "feature_set"]):
        for lang in langs.split(","):
            max_r2 = group["r2_score"].max()
            avg_r2 = group["r2_score"].mean()
            records.append({
                "pair": f"{neuron}_{lang}",
                "lang": lang,
                "neuron": neuron,
                "feature_set": feat_set,
                "max_r2": max_r2,
                "avg_r2": avg_r2
            })

    df_agg = pd.DataFrame(records)
    # Pivot into matrices
    max_matrix = df_agg.pivot(index="pair", columns="feature_set", values="max_r2")
    avg_matrix = df_agg.pivot(index="pair", columns="feature_set", values="avg_r2")

    # Sort rows grouped by language, then neuron id
    def sort_key(idx):
        neuron, lang = idx.split("_")
        return (lang, int(neuron))
    max_matrix = max_matrix.reindex(sorted(max_matrix.index, key=sort_key))
    avg_matrix = avg_matrix.reindex(sorted(avg_matrix.index, key=sort_key))

    return max_matrix, avg_matrix

def get_neuron_top_features(max_matrix, avg_matrix, top_k=5):
    """For each neuron, pool across languages to find top feature sets."""
    neuron_features = {}
    
    # Extract unique neurons from the index
    neurons = set()
    for idx in max_matrix.index:
        neuron, _ = idx.split("_")
        neurons.add(int(neuron))
    
    for neuron in sorted(neurons):
        # Pool R² scores across all languages for this neuron
        neuron_str = str(neuron)
        neuron_rows = [idx for idx in max_matrix.index if idx.split("_")[0] == neuron_str]
        
        if not neuron_rows:
            continue
            
        # Calculate pooled statistics per feature set
        feature_stats = {}
        for feature_set in max_matrix.columns:
            max_vals = max_matrix.loc[neuron_rows, feature_set].dropna()
            avg_vals = avg_matrix.loc[neuron_rows, feature_set].dropna()
            
            if len(max_vals) > 0:
                # Get the index with max value (e.g., "42_fr") and extract language
                best_idx = max_vals.idxmax() if not max_vals.empty else None
                best_lang = best_idx.split("_")[1] if best_idx else None
                
                feature_stats[feature_set] = {
                    'pooled_max': max_vals.max(),
                    'pooled_avg': avg_vals.mean(),
                    'n_langs': len(max_vals),
                    'best_lang': best_lang
                }
        
        # Sort by pooled_max and get top k
        sorted_features = sorted(feature_stats.items(), 
                               key=lambda x: x[1]['pooled_max'], 
                               reverse=True)[:top_k]
        
        neuron_features[neuron] = sorted_features
    
    return neuron_features

def create_r2_histograms(max_matrix, avg_matrix, output_path, bins=20):
    """Create histograms showing R² score distributions for each feature set."""
    n_features = len(max_matrix.columns)
    n_cols = 4
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4*n_rows))
    axes = axes.flatten() if n_features > 1 else [axes]
    
    for idx, feature_set in enumerate(max_matrix.columns):
        ax = axes[idx]
        
        # Get all R² scores for this feature set (across all neuron-language pairs)
        max_scores = max_matrix[feature_set].dropna()
        avg_scores = avg_matrix[feature_set].dropna()
        
        # Create histogram
        bins_edges = np.linspace(0, 1, bins + 1)
        
        # Plot both max and avg distributions
        ax.hist(max_scores, bins=bins_edges, alpha=0.5, label='Max R²', color='blue', edgecolor='black')
        ax.hist(avg_scores, bins=bins_edges, alpha=0.5, label='Avg R²', color='orange', edgecolor='black')
        
        ax.set_xlabel('R² Score')
        ax.set_ylabel('Number of Neurons')
        ax.set_title(f'{feature_set}', fontsize=10, fontweight='bold')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Add vertical line at R²=0.5
        ax.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='R²=0.5')
        
        # Add statistics text
        stats_text = f"Max: μ={max_scores.mean():.2f}, σ={max_scores.std():.2f}\n"
        stats_text += f"Avg: μ={avg_scores.mean():.2f}, σ={avg_scores.std():.2f}"
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                fontsize=7, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Hide empty subplots
    for idx in range(n_features, len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('R² Score Distributions by Feature Set', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved R² histograms to {output_path}")

def create_neuron_std_histogram(max_matrix, output_path, bins=20):
    """Create histogram showing standard deviation of max R² scores per neuron across feature sets."""
    
    # Get unique neurons and calculate std dev for each
    neuron_stds = {}
    neurons = {}
    
    # Group indices by neuron
    for idx in max_matrix.index:
        neuron_str, lang = idx.split("_")
        neuron = int(neuron_str)
        if neuron not in neurons:
            neurons[neuron] = []
        neurons[neuron].append(idx)
    
    # Calculate std dev of max scores across feature sets for each neuron
    for neuron, indices in neurons.items():
        neuron_data = max_matrix.loc[indices]
        # Get max R² for each feature set (across languages)
        feature_maxes = neuron_data.max(axis=0)  # Max across languages for each feature
        # Calculate std dev across feature sets
        std_dev = feature_maxes.std()
        neuron_stds[neuron] = std_dev
    
    # Create histogram
    fig, ax = plt.subplots(figsize=(10, 6))
    
    std_values = list(neuron_stds.values())
    
    # Remove NaN values if any
    std_values = [s for s in std_values if not np.isnan(s)]
    
    # Create bins
    bins_edges = np.linspace(0, max(std_values) if std_values else 1, bins + 1)
    
    ax.hist(std_values, bins=bins_edges, color='steelblue', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Standard Deviation of Max R² Scores Across Feature Sets')
    ax.set_ylabel('Number of Neurons')
    ax.set_title('Distribution of Neuron Selectivity\n(Std Dev of Max R² Across Feature Sets)', 
                 fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add statistics
    mean_std = np.mean(std_values)
    median_std = np.median(std_values)
    ax.axvline(mean_std, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_std:.3f}')
    ax.axvline(median_std, color='green', linestyle='--', linewidth=2, label=f'Median: {median_std:.3f}')
    ax.legend()
    
    # Add text box with statistics
    stats_text = f'Total neurons: {len(std_values)}\n'
    stats_text += f'Min std: {min(std_values):.3f}\n'
    stats_text += f'Max std: {max(std_values):.3f}\n'
    stats_text += f'Mean std: {mean_std:.3f}\n'
    stats_text += f'Median std: {median_std:.3f}'
    
    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, 
            fontsize=9, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved neuron std histogram to {output_path}")
    
    return neuron_stds

def analyze_low_r2_neurons(max_matrix, threshold=0.5):
    """Analyze neurons whose maximum R² score (across all features and languages) is below threshold."""
    
    # Get unique neurons
    neurons = {}
    for idx in max_matrix.index:
        neuron_str, lang = idx.split("_")
        neuron = int(neuron_str)
        if neuron not in neurons:
            neurons[neuron] = []
        neurons[neuron].append(idx)
    
    # Find neurons with max R² below threshold
    low_r2_neurons = []
    neuron_max_scores = {}
    
    for neuron, indices in neurons.items():
        # Get maximum R² across all feature sets and languages for this neuron
        neuron_data = max_matrix.loc[indices]
        max_r2 = neuron_data.max().max()  # Max across all features and languages
        neuron_max_scores[neuron] = max_r2
        
        if max_r2 < threshold:
            low_r2_neurons.append(neuron)
            
    return low_r2_neurons, neuron_max_scores


def save_low_r2_analysis(max_matrix, out_file, threshold=0.5):
    """Save analysis of neurons with low R² scores."""
    low_r2_neurons, neuron_max_scores = analyze_low_r2_neurons(max_matrix, threshold)
    
    with open(out_file, 'w') as f:
        f.write(f"Low R² Neuron Analysis (threshold = {threshold})\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Total unique neurons analyzed: {len(neuron_max_scores)}\n")
        f.write(f"Neurons with max R² < {threshold}: {len(low_r2_neurons)} "
                f"({100*len(low_r2_neurons)/len(neuron_max_scores):.1f}%)\n\n")
        
        # Distribution statistics
        all_max_scores = list(neuron_max_scores.values())
        f.write("Overall R² Distribution (max per neuron):\n")
        f.write(f"  Mean: {np.mean(all_max_scores):.3f}\n")
        f.write(f"  Std:  {np.std(all_max_scores):.3f}\n")
        f.write(f"  Min:  {np.min(all_max_scores):.3f}\n")
        f.write(f"  25%:  {np.percentile(all_max_scores, 25):.3f}\n")
        f.write(f"  50%:  {np.percentile(all_max_scores, 50):.3f}\n")
        f.write(f"  75%:  {np.percentile(all_max_scores, 75):.3f}\n")
        f.write(f"  Max:  {np.max(all_max_scores):.3f}\n\n")
        
        # List low R² neurons sorted by their max score
        f.write(f"Neurons with max R² < {threshold} (sorted by max R²):\n")
        f.write("-"*50 + "\n")
        
        low_r2_with_scores = [(n, neuron_max_scores[n]) for n in low_r2_neurons]
        low_r2_with_scores.sort(key=lambda x: x[1])
        
        for neuron, max_score in low_r2_with_scores[:50]:  # Show top 50
            f.write(f"  Neuron {neuron:4d}: max R² = {max_score:.3f}\n")
        
        if len(low_r2_with_scores) > 50:
            f.write(f"  ... and {len(low_r2_with_scores) - 50} more neurons\n")
        
        # Histogram of max R² scores
        f.write("\n\nR² Score Distribution (text histogram):\n")
        f.write("-"*50 + "\n")
        
        bins = np.linspace(0, 1, 11)
        hist, bin_edges = np.histogram(all_max_scores, bins=bins)
        
        for i in range(len(hist)):
            bar_length = int(40 * hist[i] / max(hist))
            bar = '█' * bar_length
            f.write(f"  [{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}]: {bar} {hist[i]} neurons\n")
    
    print(f"Saved low R² analysis to {out_file}")

def save_neuron_std_analysis(neuron_stds, out_file):
    """Save analysis of neuron standard deviations."""
    
    with open(out_file, 'w') as f:
        f.write("Neuron Feature Selectivity Analysis (Std Dev of Max R² Across Features)\n")
        f.write("="*70 + "\n\n")
        
        std_values = [s for s in neuron_stds.values() if not np.isnan(s)]
        
        f.write(f"Total neurons analyzed: {len(std_values)}\n\n")
        
        f.write("Distribution Statistics:\n")
        f.write(f"  Mean std:   {np.mean(std_values):.4f}\n")
        f.write(f"  Median std: {np.median(std_values):.4f}\n")
        f.write(f"  Min std:    {np.min(std_values):.4f}\n")
        f.write(f"  Max std:    {np.max(std_values):.4f}\n")
        f.write(f"  25th %ile:  {np.percentile(std_values, 25):.4f}\n")
        f.write(f"  75th %ile:  {np.percentile(std_values, 75):.4f}\n\n")
        
        # Interpretation
        f.write("Interpretation:\n")
        f.write("  - Low std: Neuron responds similarly to all feature sets (generalist)\n")
        f.write("  - High std: Neuron is selective for specific feature sets (specialist)\n\n")
        
        # Top selective neurons (high std)
        f.write("Top 20 Most Selective Neurons (Highest Std Dev):\n")
        f.write("-"*50 + "\n")
        sorted_neurons = sorted(neuron_stds.items(), key=lambda x: x[1], reverse=True)
        for neuron, std in sorted_neurons[:20]:
            if not np.isnan(std):
                f.write(f"  Neuron {neuron:4d}: std = {std:.4f}\n")
        
        f.write("\n")
        
        # Least selective neurons (low std)
        f.write("Top 20 Least Selective Neurons (Lowest Std Dev):\n")
        f.write("-"*50 + "\n")
        sorted_neurons_asc = sorted([(n, s) for n, s in neuron_stds.items() if not np.isnan(s)], 
                                   key=lambda x: x[1])
        for neuron, std in sorted_neurons_asc[:20]:
            f.write(f"  Neuron {neuron:4d}: std = {std:.4f}\n")
    
    print(f"Saved neuron std analysis to {out_file}")

def create_side_by_side_heatmap(max_matrix, avg_matrix, title, output_path, figsize=(24, 12)):
    """Plot two heatmaps side by side: Max R² and Avg R²."""
    if max_matrix is None or avg_matrix is None:
        print(f"No data to plot for {title}")
        return

    # Align indices/columns
    common_pairs = max_matrix.index.union(avg_matrix.index)
    common_feats = max_matrix.columns.union(avg_matrix.columns)
    max_matrix = max_matrix.reindex(index=common_pairs, columns=common_feats)
    avg_matrix = avg_matrix.reindex(index=common_pairs, columns=common_feats)

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)
    cmap = sns.color_palette("YlOrRd", as_cmap=True)

    sns.heatmap(max_matrix, cmap=cmap, vmin=0, vmax=1,
                cbar_kws={'label': 'R² Score'}, ax=axes[0])
    axes[0].set_title("Max R²", fontsize=14, fontweight='bold')
    axes[0].set_xlabel("Feature Sets")
    axes[0].set_ylabel("Neuron_Language")

    sns.heatmap(avg_matrix, cmap=cmap, vmin=0, vmax=1,
                cbar_kws={'label': 'R² Score'}, ax=axes[1])
    axes[1].set_title("Average R²", fontsize=14, fontweight='bold')
    axes[1].set_xlabel("Feature Sets")
    axes[1].set_ylabel("")

    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {output_path}")

def save_top_neurons(max_matrix, avg_matrix, out_file, r2_threshold, top_k_features=5):
    """Save neurons above threshold per feature + language summaries + top features per neuron."""

    with open(out_file, "w") as f:
        f.write("Significant Neuron-Language Pairs per Feature (R² > threshold)\n")
        f.write("="*70 + "\n\n")

        valid_features = []

        # ---- List all neurons above threshold per feature ----
        for col in max_matrix.columns:
            col_max = max_matrix[col].dropna()
            col_avg = avg_matrix[col].dropna()
            if col_max.empty or col_avg.empty:
                continue

            mask = (col_max > r2_threshold) | (col_avg > r2_threshold)
            if not mask.any():
                continue

            valid_features.append(col)
            f.write(f"Feature: {col}\n")
            print(col)
            # Collect neurons above threshold
            rows = []
            for idx in col_max.index[mask]:
                rows.append((idx, col_max[idx], col_avg[idx]))
            # Sort by max R² descending
            rows.sort(key=lambda x: -x[1])
            for idx, max_val, avg_val in rows:
                if (idx[-2:] == "de"): print(max_val)
                f.write(f"  {idx:<12}  Max R²={max_val:.3f}   Avg R²={avg_val:.3f}\n")
            f.write("\n")

        # ---- NEW: Top feature sets per neuron (pooled across languages) ----
        f.write("\nTop Feature Sets per Neuron (Pooled Across Languages)\n")
        f.write("="*70 + "\n\n")
        
        neuron_features = get_neuron_top_features(max_matrix, avg_matrix, top_k_features)
        
        for neuron in sorted(neuron_features.keys()):
            f.write(f"Neuron {neuron}:\n")
            for rank, (feature_set, stats) in enumerate(neuron_features[neuron], 1):
                f.write(f"  {rank}. {feature_set:<30} "
                       f"Pooled Max R²={stats['pooled_max']:.3f}  "
                       f"Pooled Avg R²={stats['pooled_avg']:.3f}  "
                       f"(n_langs={stats['n_langs']}, best_lang={stats['best_lang']})\n")
            f.write("\n")

        # ---- Feature-level averages per language ----
        if valid_features:
            f.write("\nLanguage Averages Per Feature (thresholded neurons)\n")
            f.write("="*70 + "\n\n")
            for col in valid_features:
                col_max = max_matrix[col].dropna()
                col_avg = avg_matrix[col].dropna()
                mask = (col_max > r2_threshold) | (col_avg > r2_threshold)
                if not mask.any():
                    continue

                f.write(f"Feature: {col}\n")
                lang_scores = {}
                for idx in col_max.index[mask]:
                    _, lang = idx.split("_")
                    lang_scores.setdefault(lang, []).append(col_avg[idx])
                for lang, vals in sorted(lang_scores.items(), key=lambda x: -np.mean(x[1])):
                    f.write(f"  {lang:<5} mean Avg R²={np.mean(vals):.3f}\n")
                f.write("\n")

        # ---- Overall language averages ----
        f.write("\nOverall Language-Level Analysis\n")
        f.write("="*70 + "\n\n")

        lang_scores_all = {}
        lang_scores_valid = {}

        for pair in max_matrix.index:
            _, lang = pair.split("_")
            # across all features
            lang_scores_all.setdefault(lang, []).extend(
                avg_matrix.loc[pair].dropna().tolist()
            )
            # across valid features (using avg)
            if valid_features:
                lang_scores_valid.setdefault(lang, []).extend(
                    avg_matrix.loc[pair, valid_features].dropna().tolist()
                )

        # Average per language
        avg_all = {lang: np.mean(vals) for lang, vals in lang_scores_all.items() if vals}
        avg_valid = {lang: np.mean(vals) for lang, vals in lang_scores_valid.items() if vals}

        f.write("Top Languages Across Thresholded Features:\n")
        for lang, score in sorted(avg_valid.items(), key=lambda x: -x[1]):
            f.write(f"  {lang}: {score:.3f}\n")

        f.write("\nTop Languages Across All Features:\n")
        for lang, score in sorted(avg_all.items(), key=lambda x: -x[1]):
            f.write(f"  {lang}: {score:.3f}\n")

def save_neuron_feature_csv(max_matrix, avg_matrix, out_file, top_k=10):
    """Save a CSV with top feature sets per neuron for easier analysis."""
    neuron_features = get_neuron_top_features(max_matrix, avg_matrix, top_k)
    
    records = []
    for neuron, feature_sets in neuron_features.items():
        for rank, (feature_set, stats) in enumerate(feature_sets, 1):
            records.append({
                'neuron': neuron,
                'rank': rank,
                'feature_set': feature_set,
                'pooled_max_r2': stats['pooled_max'],
                'pooled_avg_r2': stats['pooled_avg'],
                'n_languages': stats['n_langs'],
                'best_language': stats['best_lang']
            })
    
    df_out = pd.DataFrame(records)
    df_out.to_csv(out_file, index=False)
    print(f"Saved neuron-feature set analysis to {out_file}")

def main():
    args = get_args()
    base_dir = f"/home/aastha/multilingual_interpretability/lang2vec_probing/results/{args.results_dir}"
    out_dir = os.path.join(args.output_root, args.results_dir)
    os.makedirs(out_dir, exist_ok=True)

    # Load all *_sources.csv files, skip empties
    dfs = []
    for f in os.listdir(base_dir):
        if not f.endswith("_sources.csv"):
            continue
        path = os.path.join(base_dir, f)
        if os.path.getsize(path) == 0:
            print(f"⚠️ Skipping empty file: {f}")
            continue
        try:
            df = pd.read_csv(path)
            if df.empty:
                print(f"⚠️ Skipping file with no rows: {f}")
                continue
            dfs.append(df)
        except Exception as e:
            print(f"⚠️ Skipping {f} due to error: {e}")
            continue

    if not dfs:
        print("No valid CSV files found!")
        return

    df_all = pd.concat(dfs, ignore_index=True)
    layers = sorted(df_all["layer"].unique())

    for layer in layers:
        max_matrix, avg_matrix = aggregate_features(df_all, layer)
        if max_matrix is None:
            continue

        # Threshold filtering
        keep_rows = ((max_matrix > args.r2_threshold) | (avg_matrix > args.r2_threshold)).any(axis=1)
        max_matrix_filtered = max_matrix.loc[keep_rows]
        avg_matrix_filtered = avg_matrix.loc[keep_rows]

        if max_matrix_filtered.empty:
            continue

        output_path = os.path.join(out_dir, f"layer_{layer}_heatmaps.png")
        title = f"Layer {layer} – Neuron/Language vs Feature Sets"
        create_side_by_side_heatmap(max_matrix_filtered, avg_matrix_filtered, title, output_path, args.figsize)

        # Create R² distribution histograms
        hist_path = os.path.join(out_dir, f"layer_{layer}_r2_distributions.png")
        create_r2_histograms(max_matrix, avg_matrix, hist_path)

        
        
        # Analyze neurons with low R² scores
        low_r2_file = os.path.join(out_dir, f"layer_{layer}_low_r2_analysis.txt")
        save_low_r2_analysis(max_matrix, low_r2_file, threshold=args.r2_threshold)

        # Save top neurons per feature (with new neuron analysis)
        top_file = os.path.join(out_dir, f"layer_{layer}_top_neurons.txt")
        save_top_neurons(max_matrix, avg_matrix, top_file, args.r2_threshold, args.top_k_features)
        
        # Save neuron-feature analysis as CSV (using unfiltered matrices for complete view)
        csv_file = os.path.join(out_dir, f"layer_{layer}_neuron_features.csv")
        save_neuron_feature_csv(max_matrix, avg_matrix, csv_file, top_k=10)

    print("\nAll visualizations + summaries saved under:", out_dir)

if __name__ == "__main__":
    main()