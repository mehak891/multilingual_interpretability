import os
import pandas as pd
import itertools
from scipy.stats import spearmanr, kendalltau
import numpy as np
from pathlib import Path

def load_ranked_list(path):
    df = pd.read_csv(path)
    return df.sort_values("rank")["feature_idx"].tolist()

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
    # build rank dicts
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

def collect_files(base_dir):
    csv_files = list(Path(base_dir).rglob("*.csv"))
    entries = []

    for f in csv_files:
        parts = f.parts
        if "identification" not in parts:
            continue
        idx = parts.index("identification")
        # dataset/split are right after layer_x
        dataset = parts[idx + 4]
        split = parts[idx + 5]
        config = f"{dataset}/{split}"
        lang = f.stem
        entries.append((lang, config, f))

    return entries

def main(base_dir, k=50, out_file="ranking_similarity.csv"):
    entries = collect_files(base_dir)
    # group by lang
    lang_groups = {}
    for lang, config, path in entries:
        lang_groups.setdefault(lang, []).append((config, path))

    results = []
    for lang, files in lang_groups.items():
        lists = {cfg: load_ranked_list(path) for cfg, path in files}
        for (c1, p1), (c2, p2) in itertools.combinations(files, 2):
            metrics = compare_lists(lists[c1], lists[c2], k=k)
            results.append({
                "lang": lang,
                "config1": c1,
                "config2": c2,
                **metrics
            })

    df = pd.DataFrame(results)
    df.to_csv(out_file, index=False)
    print(f"Saved results to {out_file}")
    return df

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("base_dir", help="Base directory containing identification/*/*/layer_x/<dataset>/<split>/*.csv files")
    parser.add_argument("--k", type=int, default=50, help="Top-k overlap size")
    parser.add_argument("--out", type=str, default="ranking_similarity.csv", help="Output CSV file")
    args = parser.parse_args()

    main(args.base_dir, k=args.k, out_file=args.out)
