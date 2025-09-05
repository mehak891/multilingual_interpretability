"""
Feature Storage Utilities for SAE-LAPE Analysis
Minimal utility functions to store detailed feature rankings by layer and language.
"""

import torch
import pandas as pd
from pathlib import Path
from typing import Dict, List


def save_layer_language_features(
    base_dir: str,
    model_name: str,
    method: str,
    layer_idx: int,
    language: str,
    feature_indices: List[int],
    entropies: List[float],
    activation_probs: List[float],
    top_k: int = 100
) -> Path:
    """
    Save top-k features for a specific layer and language.
    
    Creates directory structure: identification/modelname/method/layer_X/language.csv
    
    Args:
        base_dir: Base directory (e.g., "identification")
        model_name: Name of the model
        method: Analysis method (e.g., "sae_lape")
        layer_idx: Layer index
        language: Language code (e.g., "en", "es")
        feature_indices: List of feature indices sorted by relevance
        entropies: Corresponding entropy values
        activation_probs: Corresponding activation probabilities
        top_k: Number of top features to save
        
    Returns:
        Path to saved CSV file
    """
    # Create directory structure
    dir_path = Path(base_dir) / model_name / method / f"layer_{layer_idx}"
    dir_path.mkdir(parents=True, exist_ok=True)
    
    # Prepare feature data (take only top_k)
    num_features = min(top_k, len(feature_indices))
    features_df = pd.DataFrame({
        'feature_idx': feature_indices[:num_features],
        'entropy': entropies[:num_features],
        'activation_prob': activation_probs[:num_features],
        'rank': list(range(1, num_features + 1))
    })
    
    # Save as CSV
    csv_path = dir_path / f"{language}.csv"
    features_df.to_csv(csv_path, index=False)
    
    return csv_path


def extract_detailed_features_from_stats(lang_to_stats: Dict, sorted_langs: List[str]) -> Dict:
    """
    Extract detailed feature analysis from accumulated statistics.
    
    Args:
        lang_to_stats: Statistics dictionary from StreamingExtractor
        sorted_langs: List of language codes in sorted order
        
    Returns:
        Dictionary organized as: {language: {layer_idx: {feature_data}}}
    """
    # Import here to avoid circular dependencies
    from utils.metrics import stack_activations_count
    
    # Get stacked data
    (
        num_examples,
        num_tokens, 
        over_zero_token,
        over_zero_example,
        global_max_active_over_zero,
        global_min_active_over_zero,
        global_avg_active_over_zero,
    ) = stack_activations_count(lang_to_stats, sorted_langs)
    
    # Calculate activation probabilities and entropy
    num_layers, hidden_dim, num_langs = over_zero_token.size()
    activation_probs = over_zero_token.float() / num_tokens.float()
    
    # L1 normalization with stability constant
    normed_activation_probs = activation_probs / (activation_probs.sum(dim=-1, keepdim=True) + 1e-10)
    normed_activation_probs[torch.isnan(normed_activation_probs)] = 0
    
    # Entropy calculation
    log_probs = torch.where(normed_activation_probs > 0, normed_activation_probs.log(), 0)
    entropy = -torch.sum(normed_activation_probs * log_probs, dim=-1)
    
    # Organize results by language and layer
    results = {}
    
    for lang_idx, lang in enumerate(sorted_langs):
        results[lang] = {}
        for layer_idx in range(num_layers):
            # Get entropy and probabilities for this layer
            layer_entropy = entropy[layer_idx, :]
            layer_probs = activation_probs[layer_idx, :, lang_idx]
            
            # Sort by entropy (most language-specific first - lowest entropy)
            sorted_indices = torch.argsort(layer_entropy)
            
            results[lang][layer_idx] = {
                'feature_indices': sorted_indices.cpu().tolist(),
                'entropies': layer_entropy[sorted_indices].cpu().tolist(),
                'activation_probs': layer_probs[sorted_indices].cpu().tolist(),
            }
    
    return results


def save_all_detailed_features(
    lang_to_stats: Dict,
    model_name: str,
    method: str = "sae_lape",
    base_dir: str = "identification",
    top_k: int = 100
) -> None:
    """
    Save detailed feature rankings for all layers and languages.
    
    Args:
        lang_to_stats: Statistics dictionary from StreamingExtractor
        model_name: Name of the model
        method: Analysis method name
        base_dir: Base directory for saving
        top_k: Number of top features to save per layer-language combination
    """
    sorted_langs = sorted(lang_to_stats.keys())
    detailed_features = extract_detailed_features_from_stats(lang_to_stats, sorted_langs)
    
    saved_files = []
    for lang in sorted_langs:
        for layer_idx in detailed_features[lang]:
            layer_data = detailed_features[lang][layer_idx]
            
            csv_path = save_layer_language_features(
                base_dir=base_dir,
                model_name=model_name,
                method=method,
                layer_idx=layer_idx,
                language=lang,
                feature_indices=layer_data['feature_indices'],
                entropies=layer_data['entropies'],
                activation_probs=layer_data['activation_probs'],
                top_k=top_k
            )
            saved_files.append(csv_path)
    
    print(f"Saved {len(saved_files)} feature ranking files to {base_dir}/{model_name}/{method}/")


def save_detailed_features_from_extractor(
    extractor,
    model_name: str,
    method: str = "sae_lape", 
    base_dir: str = "identification",
    top_k: int = 100
) -> None:
    """
    Convenience function to save features directly from a StreamingExtractor instance.
    
    Args:
        extractor: StreamingExtractor instance after running analysis
        model_name: Name of the model
        method: Analysis method name
        base_dir: Base directory for saving
        top_k: Number of top features to save per layer-language combination
    """
    save_all_detailed_features(
        lang_to_stats=extractor.lang_to_stats,
        model_name=model_name,
        method=method,
        base_dir=base_dir,
        top_k=top_k
    )