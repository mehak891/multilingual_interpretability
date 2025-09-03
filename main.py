import os
import sys
from tqdm import tqdm
from torch.profiler import profile, record_function, ProfilerActivity
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import utils.config as config
from data import loader as d_loader
from models import loader as m_loader
import time
import torch
import gc

sys.path.append(os.path.abspath('.'))
logger = config.get_logger()
args = config.Config()

def save_results(results, layer_name, output_type: str = ""):
    try:
        path = os.path.join(args.save_dir,os.path.join(f"sae_features_{output_type}_{args.dataset_name}_{args.split}_{args.language}"))
        #path = os.path.join(args.save_dir,os.path.join(f"sae_features_{output_type}_{args.language}"))
        os.makedirs(path, exist_ok=True)
        layer_tensor = torch.cat(results, dim=0)
        save_path = os.path.join(path, f"{layer_name.replace('.', '_')}.pt")
        torch.save(layer_tensor, save_path)
        logger.info(f" Saved: {save_path} | shape = {layer_tensor.shape}")
    except Exception as e:
        logger.error(f"Error in saving {layer_name} at {output_type}: {e}")
        


def main():
    dataset_loader = d_loader.HFDatasetLoader(args.model_name,
                    args.dataset_name, args.text_field, 
                    args.split, args.language, args.batch_size, 
                    args.max_length, args.num_workers, logger)
    data_loader = dataset_loader.dataloader
    model_loader = m_loader.HFModelLoader(args.model_name,args.model_type,args.device,logger)
    model = model_loader.model
    sae_loader = m_loader.SAELoader(args.sae_model,args.layers,args.device,logger)
    saes = sae_loader.sae_model
    device = args.device
    for layer_index, (layer_name, sae_model) in enumerate(saes.items()):
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True) as prof:
            start = time.time()
            logger.info(f"\n[Layer {layer_index}] Processing {layer_name}...")
            sae_model.to(args.device)
            layer_outputs, layer_indices, layer_preacts = [], [], []
            for batch in tqdm(data_loader, desc=f"Layer {layer_name}"):
                input_ids, attention_mask = batch["input_ids"].to(device), batch["attention_mask"].to(device)
                with torch.no_grad():
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    hidden_states = outputs.hidden_states  # list of hidden states

                # SAE expects input from a specific layer (e.g., hidden_states[1] for block.0)
                # Layer index +1 because hidden_states[0] is input embedding
                hidden = hidden_states[layer_index + 1]
                logger.info(f"Hidden state size {hidden.shape}")
                #flat_hidden = hidden.view(-1, hidden.shape[-1])  # (B*T, dim)
                logger.info(f"Flattened hidden state size {hidden.shape}")
                sae_latents = sae_model.encode(hidden)  # (B*T, latent)
                sae_latents_activations, sae_latents_indices, sae_latents_preacts = sae_latents.top_acts.cpu(), sae_latents.top_indices.cpu(), sae_latents.pre_acts.cpu()
                logger.info(f"Sae Latents size {sae_latents_activations.shape} and {sae_latents_indices.shape} and {sae_latents_preacts.shape}")
                layer_outputs.append(sae_latents_activations)
                layer_indices.append(sae_latents_indices)
                layer_preacts.append(sae_latents_preacts)
            # After all batches for this layer are done → save to disk
            save_results(layer_outputs, layer_name, "activations")
            save_results(layer_indices, layer_name, "indices")
            #save_results(layer_preacts, layer_name, "preacts")
            print(f"Allocated memory: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")
            print(f"Reserved memory: {torch.cuda.memory_reserved() / 1024 ** 2:.2f} MB")
            del sae_latents, sae_model, hidden, hidden_states, outputs, input_ids, attention_mask
            del sae_latents_activations, sae_latents_indices, sae_latents_preacts
            del layer_outputs, layer_indices, layer_preacts
            torch.cuda.empty_cache()
            gc.collect()
            print(f"After free, Allocated memory: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")
            print(f"After free, Reserved memory: {torch.cuda.memory_reserved() / 1024 ** 2:.2f} MB")
            end = time.time()
            logger.info(f" Time taken to run: {end-start}")
        print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))

if __name__ == "__main__":
    main()



