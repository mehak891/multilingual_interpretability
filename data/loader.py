# models/loader.py

import torch
from transformers import (
    AutoModel,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    AutoModelForCausalLM
)
from sparsify import Sae
from sae_lens import SAE

class HFModelLoader:
    def __init__(self, 
        model_name: str,
        model_type: str = "llm",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        logger = None):
        self.model_name = model_name
        self.model_type = model_type
        self.device = device
        self.logger = logger
        self.model = None
        self.tokenizer = None
        self.load_model_and_tokenizer()
    
    def load_model_and_tokenizer(self):
        self.logger.info(f"Loading model '{self.model_name}' of type '{self.model_type}'")
        try:
            # Load the correct model type
            if self.model_type == "mlm":
                self.model = AutoModelForMaskedLM.from_pretrained(self.model_name)
            elif self.model_type == "classification":
                self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            elif self.model_type == "seq2seq":
                self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            else:  # Default: base model (no head)
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name, output_hidden_states=True)

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            self.model.to(self.device)
            self.model.eval()

            self.logger.info(f"Successfully loaded model on '{self.device}'")

        except Exception as e:
            self.logger.error(f"Error loading model '{self.model_name}': {e}")
            raise

    def get_model(self):
        return self.model

    def get_tokenizer(self):
        return self.tokenizer

    def summary(self):
        self.logger.info(f"Model: {self.model_name}")
        self.logger.info(f"Type: {self.model_type}")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Parameters: {sum(p.numel() for p in self.model.parameters()):,}")

class SAELoader:
    def __init__(self, model_name: str, layers: list[str], device: str = "cuda" if torch.cuda.is_available() else "cpu",logger = None):
        self.model_name = model_name
        self.device = device
        self.sae_model = None
        self.logger = logger
        self.layers = layers
        self.load_sae()

    def load_sae(self):
        self.logger.info(f"Loading Sae model '{self.model_name}' for layers {self.layers}")
        if 'llama' in self.model_name:
            self.sae_model = Sae.load_many(self.model_name, layers=self.layers, local=(self.model_name.startswith("/home/models/")))
            #self.sae_model = Sae.load_from_hub(self.model_name, hookpoint="layers.10")
        else:
            for layer in self.layers:
                layer = "layer_"+layer.split('.')[-2]
                root_dir = f"{layer}/width_65k/canonical"
                sae, cfg_dict, sparsity = SAE.from_pretrained(
                    release=self.model_name,  # see other options in sae_lens/pretrained_saes.yaml
                    sae_id=root_dir,  # won't always be a hook point
                    device=self.device
                    )
                self.sae_model = {self.layers[0]: sae}
        self.logger.info(f"Successfully loaded Sae model on '{self.device}'")

    def get_sae(self):
        return self.sae_model