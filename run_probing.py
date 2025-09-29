import os
import argparse
import torch
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import defaultdict

# project imports
from data.multiloader import MultilingualDatasetManager
from models import loader as m_loader
from utils import config
from tqdm import tqdm

# -----------------------------
# CLI
# -----------------------------
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True,
                        help="Base model (e.g. HuggingFace ID or local path)")
    parser.add_argument('--model_name', type=str, default='Llama-3.2-1B')
    parser.add_argument("--sae-model", type=str, required=True,
                        help="Trained SAE path")
    parser.add_argument("--method", type=str, default="sae_lape")
    parser.add_argument("--layers", nargs="+", required=True,
                        help="Layers to process, e.g. 0 1 2")
    parser.add_argument("--langs", nargs="+", required=True,
                        help="Languages, e.g. en de fr it")
    parser.add_argument("--exp", type=str, required=True,
                        help="Experiment descriptor, e.g. jw300-thresh-50")
    parser.add_argument("--features", type=str, nargs="+", required=True,
                        help="Feature CSV names without prefix, e.g. geo fam")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save-acts", action="store_true",
                        help="If set, save activations before probing")
    parser.add_argument("--all-neurons", action="store_true",
                        help="If set, run probing for all SAE neurons instead of just loaded ones")
    return parser.parse_args()

# -----------------------------
# Load identified neuron indices with source tracking
# -----------------------------
def load_neuron_indices_with_sources(exp, model_name, method, layers, langs, split):
    indices = {int(l): {} for l in layers}
    # Track which languages contribute each neuron
    neuron_sources = {int(l): defaultdict(list) for l in layers}
    # print(langs)
    
    base_dir = f"identification/{os.path.basename(model_name)}/{method}"
    for layer in layers:
        layer_int = int(layer)
        for lang in langs:
            csv_path = os.path.join(base_dir, f"layer_{layer}", exp, split, f"{lang}.csv")
            if os.path.exists(csv_path):
                print(lang)
                df = pd.read_csv(csv_path)
                neuron_list = df["feature_idx"].tolist()
                indices[layer_int][lang] = neuron_list
                # print(len(neuron_list))
                # Track source languages for each neuron
                for neuron_idx in neuron_list:
                    neuron_sources[layer_int][neuron_idx].append(lang)
            else:
                indices[layer_int][lang] = []
    
    return indices, neuron_sources

# -----------------------------
# Load Lang2Vec features from CSV
# -----------------------------
def load_feature_csvs(langs, feature_names):
    feats = {}
    for fset in feature_names:
        path = os.path.join("lang2vec_probing/features", f"lang2vec_{fset}.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Feature file not found: {path}")
        df = pd.read_csv(path, index_col=0)
        df = df.loc[df.index.intersection(langs)]
        feats[fset] = df
    return feats

# -----------------------------
# Collect activations for selected neurons
# -----------------------------
def collect_activations(model, saes, dataset_manager, langs, layers, neuron_indices, 
                        neuron_sources, split, batch_size, device, all_neurons=False):
    model.to(device)
    model.eval()
    print(langs)

    lang_to_acts = {lang: {} for lang in langs}
    
    # Compute the union of neuron indices and track sources
    union_indices = {}
    union_sources = {}
    
    for layer in layers:
        l = int(layer)
        
        if all_neurons:
            # Get the total number of neurons from the SAE for this layer
            layer_name = f"layers.{l}.mlp"
            if layer_name in saes:
                sae = saes[layer_name]
                total_neurons = sae.num_latents  # Assuming d_sae contains the total number of neurons
                union_indices[l] = list(range(total_neurons))
                # For all neurons, we don't track source languages (set to empty)
                union_sources[l] = {i: [] for i in range(total_neurons)}
                print(f"Layer {l}: Using all {total_neurons} SAE neurons")
            else:
                print(f"Warning: SAE not found for layer {l}, skipping")
                union_indices[l] = []
                union_sources[l] = {}
        else:
            # Original logic for selective neurons
            all_indices = set()  # Using set ensures uniqueness
            for lang in langs:
                if neuron_indices[l][lang]:
                    all_indices.update(neuron_indices[l][lang])
            
            # Convert to sorted list for consistent ordering
            union_indices[l] = sorted(list(all_indices))
            
            # Create mapping of neuron index to source languages for union
            union_sources[l] = {}
            for neuron_idx in union_indices[l]:
                union_sources[l][neuron_idx] = neuron_sources[l][neuron_idx]
            
            # Verify uniqueness
            assert len(union_indices[l]) == len(set(union_indices[l])), \
                f"Duplicate neurons found in union for layer {l}"
            
            print(f"Layer {l}: Union of {len(union_indices[l])} unique neurons across languages")
            
            # Print distribution of neurons by source language count
            source_counts = {}
            for neuron_idx in union_indices[l]:
                num_sources = len(union_sources[l][neuron_idx])
                source_counts[num_sources] = source_counts.get(num_sources, 0) + 1
            
            print(f"  Neuron distribution by number of source languages:")
            for num_sources in sorted(source_counts.keys()):
                print(f"    {num_sources} language(s): {source_counts[num_sources]} neurons")

    for lang in langs:
        dl = dataset_manager.create_dataloader(
            "jw300", lang, split,
            batch_size=batch_size, shuffle=False
        )
        if dl is None:
            print(f"Dataloader for {lang} is None")
            continue
        
        for i, (layer_name, sae) in enumerate(saes.items()):
            l = int(layer_name.split(".")[1])
            
            if not union_indices[l]:
                print(f"No neurons in union for layer {l}")
                continue
                
            sae = sae.to(device)
            collected = []

            print(f"Collecting activations for {lang} | Layer {l} | Union indices: {len(union_indices[l])}")

            for batch in tqdm(dl, desc=f"{lang} | Layer {l}"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)

                with torch.no_grad():
                    outputs = model(input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    output_hidden_states=True)
                    hidden = outputs.hidden_states[int(l) + 1]  # (B,T,H)
                    sae_out = sae.encode(hidden)
                    latents = sae_out.pre_acts  # (B,T,N)

                    selected = latents[:, :, union_indices[l]]  # (B,T,K)
                    collected.append(selected.mean(dim=(0, 1)).cpu())

            if collected:
                lang_to_acts[lang][l] = torch.stack(collected).mean(dim=0).numpy()
            sae.to("cpu")
            torch.cuda.empty_cache()

    return lang_to_acts, union_indices, union_sources

# -----------------------------
# Run probes with source language tracking
# -----------------------------
def run_probes_with_sources(lang_to_acts, union_indices, union_sources, features, 
                            langs, layers, out_dir, all_neurons=False):
    os.makedirs(out_dir, exist_ok=True)

    for fset, feat_df in features.items():
        results = []
        processed_neurons = set()  # Track processed neuron-feature pairs

        for l in layers:
            l = int(l)
            lang_subset = [lang for lang in langs if l in lang_to_acts[lang].keys()]
            if len(lang_subset) < 2:
                continue
            
            print(f"Processing layer {l} with languages: {lang_subset}")
            
            # Build matrix: langs × neurons (using union indices)
            X = np.stack([lang_to_acts[lang][l] for lang in lang_subset], axis=0)  # (L, K)
            neuron_ids = union_indices[l]  # These are already unique
            feat_subdf = feat_df.loc[lang_subset]
            
            # Verify uniqueness of neuron IDs
            assert len(neuron_ids) == len(set(neuron_ids)), \
                f"Duplicate neurons found in union_indices for layer {l}"
            
            print(f"Activation matrix shape: {np.shape(X)}")
            print(f"Number of unique neurons: {len(neuron_ids)}")
            
            for n_idx in tqdm(range(X.shape[1])):
                x = X[:, n_idx].reshape(-1, 1)
                neuron_id = neuron_ids[n_idx]
                source_langs = union_sources[l].get(neuron_id, [])  # Get source languages, default to empty list
                
                for f_idx in range(feat_subdf.shape[1]):
                    # Create unique key for this neuron-feature pair
                    pair_key = (l, neuron_id, fset, f_idx)
                    
                    # Skip if already processed (shouldn't happen with proper union)
                    if pair_key in processed_neurons:
                        print(f"Warning: Skipping duplicate neuron-feature pair: {pair_key}")
                        continue
                    
                    processed_neurons.add(pair_key)
                    
                    y = feat_subdf.iloc[:, f_idx].values
                    if np.allclose(y, y[0]):
                        continue
                    try:
                        clf = LinearRegression()
                        clf.fit(x, y)
                        y_pred = clf.predict(x)
                        score = r2_score(y, y_pred)
                    except Exception:
                        score = float("nan")

                    results.append({
                        "layer": l,
                        "neuron_idx": neuron_id,
                        "source_languages": ",".join(sorted(source_langs)) if source_langs else "",  # Store as comma-separated string, empty if no sources
                        "num_source_langs": len(source_langs),
                        "feature_set": fset,
                        "feature_name": feat_subdf.columns[f_idx],  # Include feature name
                        "feature_idx": f_idx,
                        "r2_score": score
                    })
            
            print(f"  Processed {len(set(neuron_ids))} unique neurons for {feat_subdf.shape[1]} features")

        # Check for duplicates in results
        df = pd.DataFrame(results)
        duplicates = df.duplicated(subset=['layer', 'neuron_idx', 'feature_idx'], keep=False)
        if duplicates.any():
            print(f"Warning: Found {duplicates.sum()} duplicate entries in results!")
            # Remove duplicates, keeping the first occurrence
            df = df.drop_duplicates(subset=['layer', 'neuron_idx', 'feature_idx'], keep='first')
        
        # Use different filename when using all neurons
        filename = f"{fset}_probes_all_neurons.csv" if all_neurons else f"{fset}_probes_with_sources.csv"
        df.to_csv(os.path.join(out_dir, filename), index=False)
        print(f"[INFO] Saved {len(df)} unique probe results with source info → {out_dir}")

# -----------------------------
# Main function
# -----------------------------
def main():
    args = get_args()
    print(args.features)
    args.layer_names = args.layers
    args.layers = [layer.split(".")[1] for layer in args.layers]
    logger = config.get_logger()

    # Load model + SAE
    model_loader = m_loader.HFModelLoader(args.model_path, "llm", args.device, logger)
    model = model_loader.model
    sae_loader = m_loader.SAELoader(args.sae_model, args.layer_names, args.device, logger)
    saes = sae_loader.sae_model

    dataset_manager = MultilingualDatasetManager(model_name=args.model_path)

    # Load neuron indices with source tracking (only needed if not using all neurons)
    if args.all_neurons:
        print("[INFO] Using all SAE neurons for probing")
        neuron_indices, neuron_sources = {}, {}
    else:
        print("[INFO] Using selective neurons from identification results")
        neuron_indices, neuron_sources = load_neuron_indices_with_sources(
            args.exp, args.model_name, args.method, args.layers, args.langs, args.split
        )

    # Collect activations for union of neurons
    lang_to_acts, union_indices, union_sources = collect_activations(
        model, saes, dataset_manager,
        args.langs, args.layers, neuron_indices, neuron_sources,
        args.split, args.batch_size, args.device, args.all_neurons
    )

    # Optional save
    if args.save_acts:
        out_act_dir = f"lang2vec_probing/results/{args.exp}/activations"
        os.makedirs(out_act_dir, exist_ok=True)
        torch.save(lang_to_acts, os.path.join(out_act_dir, "activations.pt"))
        torch.save(union_indices, os.path.join(out_act_dir, "union_indices.pt"))
        torch.save(union_sources, os.path.join(out_act_dir, "union_sources.pt"))
        print(f"[INFO] Saved activations, union indices, and sources → {out_act_dir}")

    # Load lang2vec features
    print(args.features)
    features = load_feature_csvs(args.langs, args.features)

    # Run probes with source tracking
    out_dir = f"lang2vec_probing/results/{args.exp}"
    run_probes_with_sources(lang_to_acts, union_indices, union_sources, features, 
                            args.langs, args.layers, out_dir, args.all_neurons)

if __name__ == "__main__":
    main()