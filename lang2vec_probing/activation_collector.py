# lang2vec_probing/activation_collector.py
import os
import torch
import gc
from tqdm import tqdm
from data.multiloader import MultilingualDatasetManager
from models import loader as m_loader
from utils.config import Config, get_logger
import pandas as pd

def load_identified_neurons(csv_dir):
    """Union of neuron indices from CSVs."""
    all_neurons = set()
    for f in os.listdir(csv_dir):
        if f.endswith(".csv"):
            df = pd.read_csv(os.path.join(csv_dir, f))
            all_neurons.update(df["feature_idx"].tolist())
    return sorted(all_neurons)

class ActivationCollector:
    def __init__(self, model, layers, neuron_indices, device):
        self.model = model
        self.layers = layers
        self.device = device
        self.neuron_indices = neuron_indices
        self.storage = {}

    def collect(self, dataloader, lang):
        self.storage[lang] = {layer: [] for layer in self.layers}

        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Collecting {lang}"):
                input_ids = batch["input_ids"].to(self.device)
                attn_mask = batch["attention_mask"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attn_mask,
                    output_hidden_states=True
                )
                hidden_states = outputs.hidden_states

                for layer in self.layers:
                    acts = hidden_states[layer].detach().cpu()  # (B,T,H)
                    acts = acts[:, :, self.neuron_indices]       # (B,T,N)
                    self.storage[lang][layer].append(acts)

        # concatenate per layer
        for layer in self.layers:
            self.storage[lang][layer] = torch.cat(self.storage[lang][layer], dim=0)

    def save(self, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        for lang, layers in self.storage.items():
            path = os.path.join(out_dir, f"{lang}_activations.pt")
            torch.save(layers, path)
            print(f"[INFO] Saved activations for {lang} → {path}")


def main():
    args = Config()
    logger = get_logger()

    # load identified neurons
    identified_csv_dir = "identification/Llama-3.2-1B/layer_1/jw300-thresh-50/train"
    neuron_indices = load_identified_neurons(identified_csv_dir)
    print(f"[INFO] Collecting only {len(neuron_indices)} identified neurons")

    # dataloader manager
    dataset_manager = MultilingualDatasetManager(
        model_name=args.model_path, max_length=args.max_length, verbose=True
    )
    model_loader = m_loader.HFModelLoader(args.model_path, args.model_type, args.device, logger)
    model = model_loader.model

    collector = ActivationCollector(model, args.layers, neuron_indices, args.device)

    for lang in args.languages:
        dataset_manager.download_and_cache_dataset(args.dataset_name, [lang], [args.split])
        dataloader = dataset_manager.create_dataloader(
            args.dataset_name, lang, args.split,
            batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
        )
        collector.collect(dataloader, lang)
        del dataloader
        gc.collect()
        torch.cuda.empty_cache()

    collector.save("lang2vec_probing/collected_activations")


if __name__ == "__main__":
    main()
