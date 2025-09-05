import torch
import math
from tqdm import tqdm
from collections import defaultdict
from itertools import islice

class StreamingExtractor:
    def __init__(self, model, saes, device):
        self.model = model
        self.saes = saes
        self.device = device
        self.lang_to_stats = defaultdict(list)

    def update_stats(self, lang, sae_latents, layer_idx):
        """Update running counts - simplified version."""
        # Ensure enough layers allocated
        while len(self.lang_to_stats[lang]) <= layer_idx:
            self.lang_to_stats[lang].append({
                "num_examples": 0,
                "num_tokens": 0,
                "over_zero_token": None,
                "over_zero_example": None,
                "over_zero_total": None,
                "max_active_over_zero": None,
                "min_active_over_zero": None,
            })

        stats = self.lang_to_stats[lang][layer_idx]
        B, T, H = sae_latents.shape

        # Initialize on first batch
        if stats["over_zero_token"] is None:
            stats["over_zero_token"] = torch.zeros(H, dtype=torch.long, device=self.device)
            stats["over_zero_example"] = torch.zeros(H, dtype=torch.long, device=self.device)
            stats["over_zero_total"] = torch.zeros(H, dtype=torch.long, device=self.device)
            stats["max_active_over_zero"] = torch.zeros(H, dtype=torch.float, device=self.device)
            stats["min_active_over_zero"] = torch.full((H,), float('inf'), dtype=torch.float, device=self.device)

        # Compute statistics for this batch
        over_zero_mask = sae_latents > 0
        
        # Token-level counts
        token_counts = over_zero_mask.sum(dim=(0, 1))  # Sum over batch and sequence
        
        # Example-level counts
        example_mask = over_zero_mask.sum(dim=1) > 0  # Any activation in sequence
        example_counts = example_mask.sum(dim=0)  # Sum over batch
        
        # Max values
        batch_max = sae_latents.max()
        if batch_max > 0:  # Only update if there are activations
            feature_max = sae_latents.view(-1, H).max(dim=0)[0]
        else:
            feature_max = torch.zeros(H, dtype=torch.float, device=self.device)
        
        # Min values (only for non-zero activations)
        feature_min = torch.full((H,), float('inf'), dtype=torch.float, device=self.device)
        sae_flat = sae_latents.view(-1, H)
        
        for h in range(H):
            nonzero_values = sae_flat[:, h][sae_flat[:, h] > 0]
            if len(nonzero_values) > 0:
                feature_min[h] = nonzero_values.min().item()
        
        # Update accumulated statistics
        stats["num_examples"] += B
        stats["num_tokens"] += B * T
        stats["over_zero_token"] += token_counts
        stats["over_zero_example"] += example_counts
        stats["over_zero_total"] += token_counts
        stats["max_active_over_zero"] = torch.maximum(stats["max_active_over_zero"], feature_max)
        stats["min_active_over_zero"] = torch.minimum(stats["min_active_over_zero"], feature_min)

    def run(self, data_loader, lang):
        """Stream through dataset and aggregate SAE stats."""
        for layer_index, (layer_name, sae_model) in enumerate(self.saes.items()):
            sae_model.to(self.device)
            
            for batch in tqdm(islice(data_loader, 3), desc=f"{lang} | Layer {layer_name}"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                with torch.no_grad():
                    outputs = self.model(
                        input_ids=input_ids, 
                        attention_mask=attention_mask, 
                        output_hidden_states=True
                    )
                    hidden_states = outputs.hidden_states
                    hidden = hidden_states[layer_index + 1]
                    sae_latents = sae_model.encode(hidden).pre_acts

                self.update_stats(lang, sae_latents, layer_index)

            sae_model.to("cpu")
            torch.cuda.empty_cache()

    def get_stacked_data(self):
        """Convert to original format for sae_lape function."""
        from utils.metrics import stack_activations_count
        return stack_activations_count(self.lang_to_stats, sorted(self.lang_to_stats.keys()))

    def compute_sae_lape(self, **kwargs):
        """Call original sae_lape function with stacked data."""
        from utils.metrics import sae_lape
        
        (
            num_examples,
            num_tokens, 
            over_zero_token,
            over_zero_example,
            global_max_active_over_zero,
            global_min_active_over_zero,
            global_avg_active_over_zero,
        ) = self.get_stacked_data()
        
        sorted_lang = sorted(self.lang_to_stats.keys())
        
        return sae_lape(
            num_examples=num_examples,
            num_tokens=num_tokens,
            over_zero_token=over_zero_token,
            over_zero_example=over_zero_example,
            global_max_active_over_zero=global_max_active_over_zero,
            global_min_active_over_zero=global_min_active_over_zero,
            global_avg_active_over_zero=global_avg_active_over_zero,
            sorted_lang=sorted_lang,
            **kwargs
        )