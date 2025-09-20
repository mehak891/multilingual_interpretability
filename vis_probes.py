#!/usr/bin/env python3
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
    axes[0].set_title("Max R²", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Feature Sets")
    axes[0].set_ylabel("Neuron_Language")

    sns.heatmap(avg_matrix, cmap=cmap, vmin=0, vmax=1,
                cbar_kws={'label': 'R² Score'}, ax=axes[1])
    axes[1].set_title("Average R²", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Feature Sets")
    axes[1].set_ylabel("")

    plt.suptitle(title, fontsize=16, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {output_path}")

def save_top_neurons(max_matrix, avg_matrix, out_file, r2_threshold):
    """Save neurons above threshold per feature + language summaries."""

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
        max_matrix = max_matrix.loc[keep_rows]
        avg_matrix = avg_matrix.loc[keep_rows]

        if max_matrix.empty:
            continue

        output_path = os.path.join(out_dir, f"layer_{layer}_heatmaps.png")
        title = f"Layer {layer} – Neuron/Language vs Feature Sets"
        create_side_by_side_heatmap(max_matrix, avg_matrix, title, output_path, args.figsize)

        # Save top neurons per feature
        top_file = os.path.join(out_dir, f"layer_{layer}_top_neurons.txt")
        save_top_neurons(max_matrix, avg_matrix, top_file, args.r2_threshold)

    print("\nAll visualizations + summaries saved under:", out_dir)

if __name__ == "__main__":
    main()
