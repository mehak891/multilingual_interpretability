import torch
import pandas as pd
from pathlib import Path


def save_sae_lape_features(
    final_indices,
    features_info,
    sorted_langs,
    model_name,
    layer_names,
    dataset,
    split,
    method="sae_lape",
    base_dir="identification",
    top_k=100,
    experiment_tag=""
):
    """Save SAE-LAPE results to CSV files with proper layer naming."""
    saved_files = []
    
    for lang_idx, lang in enumerate(sorted_langs):
        if lang_idx >= len(final_indices) or lang not in features_info:
            continue
            
        lang_indices = final_indices[lang_idx]
        lang_info = features_info[lang]
        
        # Create mapping from (layer, feature) to entropy/prob
        if method == "sae_lape":
            coord_to_data = {}
            for i, (layer_idx, feat_idx) in enumerate(lang_info['indices']):
                coord_to_data[(layer_idx, feat_idx)] = (
                    lang_info['entropies'][i].item(),
                    lang_info['selected_probs'][i].item()
                )
        elif method == "magnitude":
            coord_to_data = {}
            for i, (layer_idx, feat_idx) in enumerate(lang_info['indices']):
                coord_to_data[(layer_idx, feat_idx)] = (
                    lang_info['avg_activations'][i].item()
                )
        
        # Save features for each layer using actual layer names
        for layer_idx, layer_features in enumerate(lang_indices):
            if len(layer_features) == 0:
                continue
            
            # Get the actual layer name from the layer_names list
            if layer_idx < len(layer_names):
                layer_name = layer_names[layer_idx]
                # Extract layer number from name like "layers.0.mlp" -> "0"
                layer_num = layer_name.split('.')[1] if '.' in layer_name else str(layer_idx)
            else:
                layer_num = str(layer_idx)
                
            # Extract data
            data = []
            for feat_idx in layer_features:
                coord = (layer_idx, feat_idx.item())
                if coord in coord_to_data:
                    if method == "sae_lape":
                        entropy, prob = coord_to_data[coord]
                        data.append({
                            'feature_idx': feat_idx.item(),
                            'entropy': entropy,
                            'activation_prob': prob,
                            'rank': len(data) + 1
                        })
                    elif method == "magnitude":
                        avg_act = coord_to_data[coord]
                        data.append({
                            'feature_idx': feat_idx.item(),
                            'avg_activation': avg_act
                        })
            
            if data:
                # Create directory and save using actual layer number
                # Include experiment_tag as suffix to dataset folder if provided
                dataset_folder = f"{dataset}"
                if experiment_tag:
                    dataset_folder += f"-{experiment_tag}"
                dir_path = Path(base_dir) / model_name / method / f"layer_{layer_num}" / dataset_folder / split
                dir_path.mkdir(parents=True, exist_ok=True)
                
                df = pd.DataFrame(data[:top_k])
                csv_path = dir_path / f"{lang}.csv"
                df.to_csv(csv_path, index=False)
                saved_files.append(csv_path)
    
    print(f"Saved {len(saved_files)} files to {base_dir}/{model_name}/{method}/")