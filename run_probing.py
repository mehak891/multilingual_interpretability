import os
import argparse
import torch
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader
from tqdm import tqdm

# project imports
from data.multiloader import MultilingualDatasetManager
from models import loader as m_loader
from utils import config

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
    parser.add_argument("--features", nargs="+", required=True,
                        help="Feature CSV names without prefix, e.g. geo fam")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save-acts", action="store_true",
                        help="If set, save activations before probing")
    return parser.parse_args()

# -----------------------------
# Load identified neuron indices
# -----------------------------
def load_neuron_indices(exp, model_name, method, layers, langs, split):
    indices = {int(l): {} for l in layers}
    base_dir = f"identification/{os.path.basename(model_name)}/{method}"
    for layer in layers:
        print("hi", layer)
        for lang in langs:
            csv_path = os.path.join(base_dir, f"layer_{layer}", exp, split, f"{lang}.csv")
            print(csv_path)
            if os.path.exists(csv_path):
                
                df = pd.read_csv(csv_path)
                print(df)
                indices[int(layer)][lang] = df["feature_idx"].tolist()
            else:
                indices[int(layer)][lang] = []
    return indices

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
def collect_activations(model, saes, dataset_manager, langs, layers, neuron_indices, split, batch_size, device):
    model.to(device)
    model.eval()

    lang_to_acts = {lang: {} for lang in langs}
    
    # First, compute the union of neuron indices for each layer across all languages
    union_indices = {}
    for layer in layers:
        l = int(layer)
        all_indices = set()
        for lang in langs:
            if neuron_indices[l][lang]:
                all_indices.update(neuron_indices[l][lang])
        union_indices[l] = sorted(list(all_indices))
        print(f"Layer {l}: Union of {len(union_indices[l])} neurons across languages")

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
            
            # Use union of indices instead of language-specific indices
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

                    # Select neurons from the union of all languages
                    selected = latents[:, :, union_indices[l]]  # (B,T,K)
                    collected.append(selected.mean(dim=(0, 1)).cpu())

            if collected:
                lang_to_acts[lang][l] = torch.stack(collected).mean(dim=0).numpy()
            sae.to("cpu")
            torch.cuda.empty_cache()

    return lang_to_acts, union_indices
# -----------------------------
# Run probes
# -----------------------------
# -----------------------------
# Run probes with union indices
# -----------------------------
def run_probes(lang_to_acts, union_indices, features, langs, layers, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    for fset, feat_df in features.items():
        results = []

        for l in layers:
            l = int(l)
            lang_subset = [lang for lang in langs if l in lang_to_acts[lang].keys()]
            if len(lang_subset) < 2:
                continue
            
            print(f"Processing layer {l} with languages: {lang_subset}")
            
            # build matrix: langs × neurons (using union indices)
            X = np.stack([lang_to_acts[lang][l] for lang in lang_subset], axis=0)  # (L, K)
            neuron_ids = union_indices[l]  # Use union indices instead of language-specific
            feat_subdf = feat_df.loc[lang_subset]
            
            print(f"Activation matrix shape: {np.shape(X)}")
            print(f"Number of union neurons: {len(neuron_ids)}")
            
            for n_idx in range(X.shape[1]):
                x = X[:, n_idx].reshape(-1, 1)
                for f_idx in range(feat_subdf.shape[1]):
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
                        "neuron_idx": neuron_ids[n_idx],
                        "feature_set": fset,
                        "feature_idx": f_idx,
                        "r2_score": score
                    })

        df = pd.DataFrame(results)
        df.to_csv(os.path.join(out_dir, f"{fset}_probes.csv"), index=False)
        print(f"[INFO] Saved {fset} probes → {out_dir}")

# -----------------------------
# Updated Main function
# -----------------------------
def main():
    args = get_args()
    args.layer_names = args.layers
    args.layers = [layer.split(".")[1] for layer in args.layers]
    logger = config.get_logger()

    # load model + SAE
    model_loader = m_loader.HFModelLoader(args.model_path, "llm", args.device, logger)
    model = model_loader.model
    sae_loader = m_loader.SAELoader(args.sae_model, args.layer_names, args.device, logger)
    saes = sae_loader.sae_model

    dataset_manager = MultilingualDatasetManager(model_name=args.model_path)

    # load neuron indices
    neuron_indices = load_neuron_indices(args.exp, args.model_name, args.method, args.layers, args.langs, args.split)

    # collect activations for union of neurons
    lang_to_acts, union_indices = collect_activations(
        model, saes, dataset_manager,
        args.langs, args.layers, neuron_indices,
        args.split, args.batch_size, args.device
    )

    # optional save
    if args.save_acts:
        out_act_dir = f"lang2vec_probing/results/{args.exp}/activations"
        os.makedirs(out_act_dir, exist_ok=True)
        torch.save(lang_to_acts, os.path.join(out_act_dir, "activations.pt"))
        torch.save(union_indices, os.path.join(out_act_dir, "union_indices.pt"))
        print(f"[INFO] Saved activations and union indices → {out_act_dir}")

    # load lang2vec features
    features = load_feature_csvs(args.langs, args.features)

    # run probes using union indices
    out_dir = f"lang2vec_probing/results/{args.exp}"
    run_probes(lang_to_acts, union_indices, features, args.langs, args.layers, out_dir)

if __name__ == "__main__":
    main()
