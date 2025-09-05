import torch
import math
import bisect
from collections import Counter

def stack_activations_count(lang_to_stats, sorted_lang):
    """Stack activations count for each language."""
    num_examples = []
    num_tokens = []
    over_zero_token = []
    over_zero_example = []
    global_max_active_over_zero = None
    global_min_active_over_zero = None
    global_over_zero_total = 0
    global_over_zero_token = 0

    # Get feature dimension from first available tensor
    H = None
    for lang in sorted_lang:
        for layer in lang_to_stats[lang]:
            if layer["over_zero_token"] is not None:
                H = layer["over_zero_token"].shape[0]
                break
        if H is not None:
            break
    
    if H is None:
        H = 16384  # Default SAE feature dimension

    for lang in sorted_lang:
        # Language totals
        lang_num_examples = sum(layer["num_examples"] for layer in lang_to_stats[lang])
        lang_num_tokens = sum(layer["num_tokens"] for layer in lang_to_stats[lang])
        
        num_examples.append(lang_num_examples)
        num_tokens.append(lang_num_tokens)
        
        # Stack layers for this language
        lang_over_zero_token = []
        lang_over_zero_example = []
        lang_max_active = []
        lang_min_active = []
        
        for layer in lang_to_stats[lang]:
            # Handle potential None values
            if layer["over_zero_token"] is not None:
                layer_token_count = layer["over_zero_token"].cpu()
                lang_over_zero_token.append(layer_token_count)
                # FIXED: Add to global_over_zero_token (matching Implementation 2)
                global_over_zero_token += layer_token_count
            else:
                lang_over_zero_token.append(torch.zeros(H, dtype=torch.long))
                
            if layer["over_zero_example"] is not None:
                lang_over_zero_example.append(layer["over_zero_example"].cpu())
            else:
                lang_over_zero_example.append(torch.zeros(H, dtype=torch.long))
                
            if layer["max_active_over_zero"] is not None:
                lang_max_active.append(layer["max_active_over_zero"].cpu())
            else:
                lang_max_active.append(torch.zeros(H, dtype=torch.float))
                
            if layer["min_active_over_zero"] is not None:
                lang_min_active.append(layer["min_active_over_zero"].cpu())
            else:
                lang_min_active.append(torch.full((H,), float('inf'), dtype=torch.float))
            
            # Accumulate global totals
            if layer["over_zero_total"] is not None:
                global_over_zero_total += layer["over_zero_total"].cpu()
        
        # Stack tensors
        lang_over_zero_token = torch.stack(lang_over_zero_token)
        lang_over_zero_example = torch.stack(lang_over_zero_example)
        lang_max_active = torch.stack(lang_max_active)
        lang_min_active = torch.stack(lang_min_active)
        
        over_zero_token.append(lang_over_zero_token)
        over_zero_example.append(lang_over_zero_example)
        
        # Update global max/min
        if global_max_active_over_zero is None:
            global_max_active_over_zero = lang_max_active
        else:
            global_max_active_over_zero = torch.maximum(global_max_active_over_zero, lang_max_active)

        if global_min_active_over_zero is None:
            global_min_active_over_zero = lang_min_active
        else:
            global_min_active_over_zero = torch.minimum(global_min_active_over_zero, lang_min_active)

    # Final stacking across languages
    num_examples = torch.tensor(num_examples, dtype=torch.long)
    num_tokens = torch.tensor(num_tokens, dtype=torch.long)
    over_zero_token = torch.stack(over_zero_token, dim=-1)  # (layers, hidden_dim, langs)
    over_zero_example = torch.stack(over_zero_example, dim=-1)

    global_avg_active_over_zero = global_over_zero_total / (global_over_zero_token + 1e-10)

    return (
        num_examples,
        num_tokens,
        over_zero_token,
        over_zero_example,
        global_max_active_over_zero,
        global_min_active_over_zero,
        global_avg_active_over_zero,
    )

def sae_lape(
    num_examples,
    num_tokens,
    over_zero_token,
    over_zero_example,
    global_max_active_over_zero,
    global_min_active_over_zero,
    global_avg_active_over_zero,
    sorted_lang,
    topk_threshold_ratio=0.8,
    example_rate=0.98,
    top=None,
    top_per_layer=False,
    entropy_threshold=None,
    lang_specific=True,
    lang_shared=False,
    shared_count=2,
    top_by_frequency=False
):
    """Original SAE-LAPE implementation."""
    num_layers, hidden_dim, num_langs = over_zero_token.size()

    # Calculate activation probabilities
    activation_probs = over_zero_token.float() / num_tokens.float()

    # L1 normalization
    normed_activation_probs = activation_probs / (activation_probs.sum(dim=-1, keepdim=True) + 1e-10)
    normed_activation_probs[torch.isnan(normed_activation_probs)] = 0

    # Entropy calculation
    log_probs = torch.where(normed_activation_probs > 0, normed_activation_probs.log(), 0)
    entropy = -torch.sum(normed_activation_probs * log_probs, dim=-1)

    # Feature filtering
    largest = False  # We want smallest entropy (most language-specific)

    # Example rate filtering
    num_examples_thresh = (num_examples.float() * example_rate).long()
    over_zero_example_filter = (over_zero_example >= num_examples_thresh.unsqueeze(0).unsqueeze(0)).any(dim=-1)

    # Token rate filtering
    hfl_rate = 0.1
    num_tokens_thresh = (num_tokens.float() * hfl_rate).long()
    over_zero_token_filter = (over_zero_token > num_tokens_thresh.unsqueeze(0).unsqueeze(0)).any(dim=-1)

    # Apply filters
    dismissed_neurons = over_zero_example_filter & over_zero_token_filter
    patched_val = torch.inf if not largest else -torch.inf
    entropy[~dismissed_neurons] = patched_val

    # Select features by entropy
    flattened_entropy = entropy.flatten()
    valid_mask = flattened_entropy != patched_val
    
    if valid_mask.sum() == 0:
        return [], {}

    valid_entropies = flattened_entropy[valid_mask]
    valid_indices = torch.where(valid_mask)[0]
    
    # Sort by entropy
    sorted_entropies, sort_order = valid_entropies.sort()
    sorted_indices = valid_indices[sort_order]

    # Convert to layer/feature coordinates
    layer_indices = sorted_indices // hidden_dim
    feature_indices = sorted_indices % hidden_dim
    
    # Get probabilities for selected features
    selected_probs = activation_probs[layer_indices, feature_indices]
    selected_probs = selected_probs.transpose(0, 1)  # (langs, features)
    
    # Language assignment
    max_probs = selected_probs.max(dim=0, keepdim=True)[0]
    lang_mask = selected_probs >= (max_probs * topk_threshold_ratio)
    
    # Handle sharing preferences
    feature_counts = lang_mask.sum(dim=0)
    
    if lang_specific:
        # Remove shared features
        shared_mask = feature_counts > 1
        lang_mask[:, shared_mask] = False
    elif lang_shared:
        # Keep only features shared across exactly shared_count languages
        wrong_count_mask = feature_counts != shared_count
        lang_mask[:, wrong_count_mask] = False
    
    # Extract results per language
    final_indices = []
    features_info = {}
    merged_coords = torch.stack([layer_indices, feature_indices], dim=1)
    
    for lang_idx, lang in enumerate(sorted_lang):
        lang_feature_mask = lang_mask[lang_idx]
        selected_coords = merged_coords[lang_feature_mask]
        
        if len(selected_coords) == 0:
            final_indices.append([torch.tensor([], dtype=torch.long) for _ in range(num_layers)])
            features_info[lang] = {"indices": [], "selected_probs": torch.tensor([]), "entropies": torch.tensor([])}
            continue
            
        # Convert to coordinate tuples
        lang_coords = [tuple(coord.tolist()) for coord in selected_coords]
        
        # Apply top-k if specified
        if top and len(lang_coords) > 0:
            if top_per_layer:
                layer_counts = [0] * num_layers
                filtered_coords = []
                for layer_idx, feat_idx in lang_coords:
                    if layer_counts[layer_idx] < top:
                        filtered_coords.append((layer_idx, feat_idx))
                        layer_counts[layer_idx] += 1
                lang_coords = filtered_coords
            else:
                lang_coords = lang_coords[:top]
        
        # Organize by layer
        layer_features = [[] for _ in range(num_layers)]
        for layer_idx, feat_idx in lang_coords:
            layer_features[layer_idx].append(feat_idx)
        
        # Convert to tensors
        for layer_idx in range(num_layers):
            layer_features[layer_idx] = torch.tensor(layer_features[layer_idx], dtype=torch.long)
        
        final_indices.append(layer_features)
        
        # Store feature info
        if lang_coords:
            coord_indices = []
            for coord in lang_coords:
                coord_tensor = torch.tensor(coord)
                matches = (merged_coords == coord_tensor).all(dim=1)
                coord_indices.extend(torch.where(matches)[0].tolist())
            
            features_info[lang] = {
                "indices": lang_coords,
                "selected_probs": selected_probs[lang_idx, coord_indices] if coord_indices else torch.tensor([]),
                "entropies": sorted_entropies[coord_indices] if coord_indices else torch.tensor([])
            }
        else:
            features_info[lang] = {"indices": [], "selected_probs": torch.tensor([]), "entropies": torch.tensor([])}

    return final_indices, features_info