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
    parser.add_argument("--use-shared", action="store_true",
                    help="If set, probe only neurons shared across multiple languages")
    parser.add_argument("--raw-model", action="store_true",
                    help="If set, probe raw hidden states instead of SAE latents")

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

def load_shared_neuron_indices(exp, model_name, method, layers, split):
    indices = {int(l): [] for l in layers}
    neuron_sources = {int(l): {} for l in layers}

    base_dir = f"identification/{os.path.basename(model_name)}/{method}"
    for layer in layers:
        l = int(layer)
        csv_path = os.path.join(base_dir, f"layer_{layer}", exp, split, "shared_neurons.csv")
        if not os.path.exists(csv_path):
            print(f"[WARN] Missing shared neuron file: {csv_path}")
            continue

        df = pd.read_csv(csv_path)
        neuron_list = df["feature_idx"].tolist()
        indices[l] = neuron_list

        for row in df.itertuples():
            neuron_sources[l][row.feature_idx] = row.languages.split(",")

        print(f"Layer {l}: Loaded {len(neuron_list)} shared neurons "
              f"(avg num_languages={df['num_languages'].mean():.2f})")

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
                        neuron_sources, split, batch_size, device, 
                        all_neurons=False, use_shared=False, raw_model=False):
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
            # Use all neurons from SAE
            layer_name = f"layers.{l}.mlp"
            if layer_name in saes:
                sae = saes[layer_name]
                if hasattr(sae,"num_latents"):
                    total_neurons = sae.num_latents  # Assuming d_sae contains the total number of neurons
                else:
                    total_neurons = sae.cfg.d_sae
                union_indices[l] = list(range(total_neurons))
                union_sources[l] = {i: [] for i in range(total_neurons)}  # no sources tracked
                print(f"Layer {l}: Using all {total_neurons} SAE neurons")
            else:
                print(f"[WARN] SAE not found for layer {l}, skipping")
                union_indices[l] = []
                union_sources[l] = {}

        elif use_shared:
            # Shared neurons already come as a flat list
            all_indices = set(neuron_indices[l])
            union_indices[l] = sorted(list(all_indices))
            
            # Source langs come directly from shared CSV
            union_sources[l] = {}
            for neuron_idx in union_indices[l]:
                union_sources[l][neuron_idx] = neuron_sources[l].get(neuron_idx, [])
            
            print(f"Layer {l}: Loaded {len(union_indices[l])} shared neurons")
            print(f"  Example: neuron {union_indices[l][0]} → {union_sources[l][union_indices[l][0]]}")

        else:
            # Selective neurons from per-language identification
            all_indices = set()
            for lang in langs:
                if neuron_indices[l][lang]:
                    all_indices.update(neuron_indices[l][lang])
            
            union_indices[l] = sorted(list(all_indices))
            
            union_sources[l] = {}
            for neuron_idx in union_indices[l]:
                union_sources[l][neuron_idx] = neuron_sources[l][neuron_idx]
            
            print(f"Layer {l}: Union of {len(union_indices[l])} unique neurons across languages")
            
            # Print distribution
            source_counts = {}
            for neuron_idx in union_indices[l]:
                num_sources = len(union_sources[l][neuron_idx])
                source_counts[num_sources] = source_counts.get(num_sources, 0) + 1
            
            print(f"  Neuron distribution by number of source languages:")
            for num_sources in sorted(source_counts.keys()):
                print(f"    {num_sources} language(s): {source_counts[num_sources]} neurons")

    # ---- Collect activations ----
    for lang in langs:
        dl = dataset_manager.create_dataloader(
            "flores_plus", lang, split,
            batch_size=batch_size, shuffle=False
        )
        if dl is None:
            print(f"[WARN] Dataloader for {lang} is None")
            continue
        
        for layer_name, sae in saes.items():
            l = int(layer_name.split(".")[1])
            if not union_indices.get(l):
                print(f"No neurons in union for layer {l}")
                continue

            if not raw_model:
                sae = sae.to(device)
            collected = []

            print(f"Collecting activations for {lang} | Layer {l} | Union neurons: {len(union_indices[l])}")

            for batch in tqdm(dl, desc=f"{lang} | Layer {l}"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)

                with torch.no_grad():
                    outputs = model(input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    output_hidden_states=True)
                    hidden = outputs.hidden_states[int(l) + 1]  # (B,T,H)
                    if not raw_model:
                        sae_out = sae.encode(hidden)
                        if hasattr(sae_out,'pre_acts'):
                            latents = sae_out.pre_acts  # (B,T,N)
                        else:
                            latents = sae_out
                    else:
                        latents = hidden
                
                    selected = latents[:, :, union_indices[l]]  # (B,T,K)
                    collected.append(selected.mean(dim=(0, 1)).cpu())
                    # else:
                    #     collected.append(hidden[:, :, union_indices[l]].mean(dim=(0, 1)).cpu())
            if collected:
                lang_to_acts[lang][l] = torch.stack(collected).mean(dim=0).numpy()
            sae.to("cpu")
            torch.cuda.empty_cache()

    return lang_to_acts, union_indices, union_sources

# -----------------------------
# Run probes with source language tracking
# -----------------------------
def run_probes_with_sources(lang_to_acts, union_indices, union_sources, features, 
                            langs, layers, out_dir, all_neurons=False, use_shared=False):
    import torch
    os.makedirs(out_dir, exist_ok=True)

    device = "cuda"  # hard-using GPU now

    for fset, feat_df in features.items():
        print(fset)
        results = []

        for l in layers:
            l = int(l)

            # Only use languages that have activations for this layer
            lang_subset = [lang for lang in langs if l in lang_to_acts[lang]]
            if len(lang_subset) < 2:
                continue

            print(f"\n[Layer {l}] Languages: {lang_subset}")

            # Activation matrix: (L, K)
            X = np.stack([lang_to_acts[lang][l] for lang in lang_subset], axis=0)
            neuron_ids = union_indices[l]
            feat_subdf = feat_df.loc[lang_subset]  # (L, F)

            print(f" - Activations: {X.shape}  (langs × neurons)")
            print(f" - Features:    {feat_subdf.shape}  (langs × feat_dims)")

            # Move to GPU tensors
            print("Moving to GPU tensors")
            X_t = torch.tensor(X, dtype=torch.float32, device=device)               # (L,K)
            Y_t = torch.tensor(feat_subdf.values, dtype=torch.float32, device=device)  # (L,F)

            # Center (matches sklearn fit_intercept=True)
            print("Centering")
            X_c = X_t - X_t.mean(dim=0, keepdim=True)  # (L,K)
            Y_c = Y_t - Y_t.mean(dim=0, keepdim=True)  # (L,F)

            # β: (K,F)
            print("Computing beta")
            denom = (X_c ** 2).sum(dim=0).unsqueeze(1).clamp(min=1e-9)  # (K,1)
            beta = (X_c.T @ Y_c) / denom  # (K,F)

            # print("Computing R²")
            # # R² per neuron-feature pair
            # # predictions = X_c @ β → (L,F)
            # # but we need per-neuron pair -> broadcast
            # # r2_matrix: (K,F)
            # # Y_pred = X_c @ beta  # (L,F)
            # # ss_res = ((Y_c - Y_pred) ** 2).sum(dim=0)  # (F,)
            # # ss_tot = (Y_c ** 2).sum(dim=0).clamp(min=1e-9)  # (F,)
            # # r2_global = 1 - ss_res / ss_tot  # but this is feature-wise

            # # # To get per-neuron-feature score:
            # # # contribution of each neuron = β[k,f] * X_c[:,k]
            # # # Compute per-neuron predictions directly
            # # # shape: (L,K,F)
            # # Y_pred_full = X_c.unsqueeze(2) * beta.unsqueeze(0)  # (*broadcast*)
            # # # sum across tokens (languages):
            # # # shape: (K,F)
            # # ss_res_nf = (Y_c.unsqueeze(1) - Y_pred_full).pow(2).sum(dim=0)
            # # r2_matrix = 1 - ss_res_nf / ss_tot  # (K,F)
            # # r2_matrix = r2_matrix.detach().cpu().numpy()

            # # --- OOM-SAFE CHUNKED R² COMPUTATION ---
            # import math

            # F = Y_c.shape[1]   # number of features
            # K = X_c.shape[1]   # number of neurons

            # chunk_size = 256   # change to 128 if memory is tight

            # r2_matrix = torch.empty((K, F), device="cpu")  # final result on CPU

            # for start in tqdm(range(0, F, chunk_size)):
            #     end = min(start + chunk_size, F)

            #     # slice feature subset
            #     Y_c_chunk = Y_c[:, start:end]              # (L, f)
            #     beta_chunk = beta[:, start:end]            # (K, f)

            #     # (L,K,f) prediction contributions
            #     Y_pred_chunk = X_c.unsqueeze(2) * beta_chunk.unsqueeze(0)

            #     # (K,f): sum across languages
            #     ss_res_chunk = (Y_c_chunk.unsqueeze(1) - Y_pred_chunk).pow(2).sum(dim=0)

            #     # (f,)
            #     ss_tot_chunk = (Y_c_chunk ** 2).sum(dim=0).clamp(min=1e-9)

            #     r2_chunk = 1 - ss_res_chunk / ss_tot_chunk  # (K,f)

            #     # move result to CPU (safe for large layers)
            #     r2_matrix[:, start:end] = r2_chunk.detach().cpu()

            #     torch.cuda.empty_cache()

            # # now r2_matrix matches previous code output exactly
            # print("Converting to numpy")
            # r2_matrix = r2_matrix.numpy()
            # print("Sample r2_matrix: ", r2_matrix[:5, :5])

            # # Store results
            # print("Storing results")
            # print(len(neuron_ids))
            # for ni, neuron_id in tqdm(enumerate(neuron_ids)):
            #     source_langs = union_sources[l].get(neuron_id, [])
            #     # print(f"{len(feat_subdf.columns)} features")
            #     for fi, feat_name in (enumerate(feat_subdf.columns)):
            #         score = float(r2_matrix[ni, fi])
            #         if np.isnan(score):
            #             continue
            #         results.append({
            #             "layer": l,
            #             "neuron_idx": neuron_id,
            #             "source_languages": ",".join(sorted(source_langs)),
            #             "num_source_langs": len(source_langs),
            #             "feature_set": fset,
            #             "feature_name": feat_name,
            #             "feature_idx": fi,
            #             "r2_score": score
            #         })

            # print(f" → Completed layer {l}: stored {len(neuron_ids)*feat_subdf.shape[1]} probe scores")

        # df = pd.DataFrame(results)

        # if all_neurons:
        #     filename = f"{fset}_probes_all_neurons.csv"
        # elif use_shared:
        #     filename = f"{fset}_probes_shared.csv"
        # else:
        #     filename = f"{fset}_probes_with_sources.csv"

        # df.to_csv(os.path.join(out_dir, filename), index=False)
        # print(f"[SAVED] {len(df)} rows → {os.path.join(out_dir, filename)}")

            print("Computing R² (GPU, low memory)")

            # --- GPU R² WITHOUT BROADCAST (VERY IMPORTANT) ---
            # Precompute sums
            x2 = (X_c ** 2).sum(dim=0, keepdim=True)         # (1,K)
            y2 = (Y_c ** 2).sum(dim=0, keepdim=True)         # (1,F)
            xy = X_c.T @ Y_c                                 # (K,F)

            # ss_res: (K,F)
            ss_res = y2 - 2 * beta * xy + (beta * beta) * x2.T
            r2_matrix = 1 - ss_res / y2.clamp(min=1e-9)      # (K,F)

            # Keep on GPU, move only what we stream
            r2_matrix = r2_matrix

            print("Streaming write to CSV (no in-memory results)")

            # Setup output file
            if all_neurons:
                filename = f"{fset}_probes_all_neurons.csv"
            elif use_shared:
                filename = f"{fset}_probes_shared.csv"
            else:
                filename = f"{fset}_probes_with_sources.csv"

            out_path = os.path.join(out_dir, filename)

            # Write header once
            if not os.path.exists(out_path):
                with open(out_path, "w") as f:
                    f.write("layer,neuron_idx,source_languages,num_source_langs,feature_set,feature_name,feature_idx,r2_score\n")

            # STREAM rows in chunks of 100,000
            CHUNK = 2000000
            buffer = []
            written = 0

            feat_names = list(feat_subdf.columns)

            for ni, neuron_id in tqdm(enumerate(neuron_ids), total=len(neuron_ids), desc="Writing results"):
                source_langs = union_sources[l].get(neuron_id, [])
                num_src = len(source_langs)
                src_str = ",".join(sorted(source_langs))

                # pull one neuron row to CPU when needed
                row = r2_matrix[ni].detach().cpu().numpy()

                for fi, feat_name in enumerate(feat_names):
                    score = float(row[fi])
                    if np.isnan(score):
                        continue

                    buffer.append(f"{l},{neuron_id},{src_str},{num_src},{fset},{feat_name},{fi},{score}\n")

                    # if buffer is large → flush to disk
                    if len(buffer) >= CHUNK:
                        with open(out_path, "a") as f:
                            f.writelines(buffer)
                        written += len(buffer)
                        buffer = []
                        print(f"  [FLUSH] wrote {written:,} rows so far")

            # final flush
            if buffer:
                with open(out_path, "a") as f:
                    f.writelines(buffer)
                written += len(buffer)

        print(f" → Finished layer {l}, total rows written: {written:,}")

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

    if not args.raw_model:
        sae_loader = m_loader.SAELoader(args.sae_model, args.layer_names, args.device, logger)
        saes = sae_loader.sae_model

    dataset_manager = MultilingualDatasetManager(model_name=args.model_path)

    # Load neuron indices with source tracking (only needed if not using all neurons)
    if args.raw_model:
        print("[INFO] Running probing on raw model activations (no SAE)")
        saes = {}              # do not load SAE
        neuron_indices = {int(l): {lang: list(range(model.config.hidden_size))
                                for lang in args.langs}
                        for l in args.layers}
        neuron_sources = {int(l): {i: [] for i in range(model.config.hidden_size)}
                        for l in args.layers}
    elif args.all_neurons:
        print("[INFO] Using all SAE neurons for probing")
        neuron_indices, neuron_sources = {}, {}
    elif args.use_shared:
        print("[INFO] Using only shared neurons across languages")
        neuron_indices, neuron_sources = load_shared_neuron_indices(
            args.exp, args.model_name, args.method, args.layers, args.split
        )
    else:
        print("[INFO] Using selective neurons from identification results")
        neuron_indices, neuron_sources = load_neuron_indices_with_sources(
            args.exp, args.model_name, args.method, args.layers, args.langs, args.split
        )

    # Collect activations for union of neurons
    lang_to_acts, union_indices, union_sources = collect_activations(
        model, saes, dataset_manager,
        args.langs, args.layers, neuron_indices, neuron_sources,
        args.split, args.batch_size, args.device, args.all_neurons, args.use_shared,
        args.raw_model
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
    if "llama" in args.model_path.lower():
        if args.raw_model:
            out_dir = f"lang2vec_probing/results_raw/{args.layers[0]}/{args.exp}"
        else:
            out_dir = f"lang2vec_probing/results/{args.layers[0]}/{args.exp}"
    else:
        if args.raw_model:
            out_dir = f"lang2vec_probing/gemma_results_new_raw/{args.layers[0]}/{args.exp}"
        else:
            out_dir = f"lang2vec_probing/gemma_results_new/{args.layers[0]}/{args.exp}"
    
    run_probes_with_sources(lang_to_acts, union_indices, union_sources, features, 
                            args.langs, args.layers, out_dir, args.all_neurons, args.use_shared)

if __name__ == "__main__":
    main()