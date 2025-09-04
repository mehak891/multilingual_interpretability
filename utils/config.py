# utils/config.py
import argparse
import logging
import os
import json

def get_args():
    parser = argparse.ArgumentParser(description="Mechanistic Interpretability on Multilingual LLMs")

    # General
    parser.add_argument('--experiment_name', type=str, default='default_exp')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda:0')

    # Model
    parser.add_argument('--model_name', type=str, default='meta-llama/Llama-3.2-1B')
    parser.add_argument('--model_type', type=str, default='llm')
    
    # Data
    parser.add_argument('--dataset_name', type=str, default = 'openlanguagedata/flores_plus')
    parser.add_argument('--language', type=str, default='eng_Latn')
    parser.add_argument('--split', type=str, default='devtest')
    parser.add_argument('--subset', type=str, default='')
    parser.add_argument('--text_field', type=str, default='text')
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--shuffle', type=bool, default=False)


    # Interpretability
    parser.add_argument('--method', type=str, default='SAE',
                        choices=['SAE', 'probing', 'activation patching'])
    parser.add_argument('--sae_model', type=str, default="EleutherAI/sae-Llama-3.2-1B-131k")
    parser.add_argument('--layers', nargs="+", default=["layers.0.mlp","layers.1.mlp","layers.2.mlp","layers.3.mlp","layers.4.mlp",
                            "layers.5.mlp","layers.6.mlp","layers.7.mlp","layers.8.mlp","layers.9.mlp","layers.10.mlp","layers.11.mlp",
                            "layers.12.mlp","layers.13.mlp","layers.14.mlp","layers.15.mlp"])

    # Output
    parser.add_argument('--save_dir', type=str, default='./outputs')
    return parser.parse_args()


def get_logger(name='getInterLogger', log_file='Interpretability.log', level=logging.INFO):
    formatter = logging.Formatter('%(asctime)s | %(levelname)s |  %(name)s | %(message)s')

    handler = logging.FileHandler(log_file)        
    handler.setFormatter(formatter)

    console = logging.StreamHandler()
    console.setFormatter(formatter)

    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        logger.addHandler(handler)
        logger.addHandler(console)
    return logger

class Config:
    def __init__(self):
        args = get_args()
        print(args)
        self.experiment_name = args.experiment_name
        self.seed = args.seed
        self.device = args.device

        # Model
        self.model_name = args.model_name
        self.model_type = args.model_type
        self.layers = args.layers

        # Data
        self.dataset_name = args.dataset_name
        self.split = args.split
        self.language = args.language
        self.batch_size = args.batch_size
        self.text_field = args.text_field
        self.max_length = args.max_length
        self.subset = args.subset
        # Interpretability
        self.method = args.method
        self.sae_model = args.sae_model

        # Output
        self.save_dir = args.save_dir
        self.num_workers = args.num_workers

        # Optionally, save config
        self._save_config()

    def _save_config(self):
        path = f"{self.experiment_name}_config.json"
        try:
            with open(path, "w") as f:
                json.dump(self.__dict__, f, indent=4)
        except Exception as e:
            print(f"Warning: Could not save config: {e}")