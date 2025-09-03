# data/loader.py

from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Optional, Union, List
from transformers import DataCollatorWithPadding

class HFDataset(Dataset):
    def __init__(self,dataset_name: str,
                 model_name: str,
                 text_field: str,
                 split: str,
                 language: str,
                 max_length: int,logger):
        self.dataset_name = dataset_name
        self.split =split
        self.logger = logger
        self.language = language
        try:
            if self.language:
                self.dataset = load_dataset(self.dataset_name,language=self.language,split=self.split)
            else:
                self.dataset = load_dataset(self.dataset_name,split=self.split)
        except Exception as e:
            self.logger.error(f"Failed to load dataset: {e}")
            raise e
        self.logger.info(f"Loading dataset '{self.dataset_name}' for language '{self.language}' split='{self.split}'")
        self.text_field = text_field
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenized_dataset = self.dataset.map(self.tokenize_fn, remove_columns=self.dataset.column_names)
    
    def __len__(self):
        return len(self.dataset)

    def tokenize_fn(self,item):
        return self.tokenizer(
            item[f"{self.text_field}"],
            truncation=True,
            padding="max_length",
            max_length=128
        )

    def __get_item__(self,idx):
        text = self.dataset[idx][self.text_field]
        encoded = self.tokenizer(text=text,truncation=True,padding='max_length',
                    max_length=self.max_length,return_tensors='pt')
        return {k:v.squeeze(0) for k,v in encoded.items()}


class HFDatasetLoader:
    def __init__(self,model_name: str,
                 dataset_name: str,
                 text_field: str,
                 split: str,
                 language: str,
                 batch_size: int,
                 max_length: int,
                 num_workers: int, logger):
        self.dataset_name = dataset_name
        self.split = split
        self.language = language
        self.text_field = text_field
        self.max_length = max_length
        self.model_name = model_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.logger = logger
        self.dataset = None
        self.dataset_obj = None
        self.collator = None
        self.get_dataloader()
        

    def get_tokenizer(self,):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        def tokenize_fn(example):
            return tokenizer(
                example["text"],
                truncation=True,
                padding="max_length",
                max_length=128
            )
        tokenizer.pad_token = tokenizer.eos_token
        # Tokenize and clean up dataset
        tokenized_dataset = dataset.map(tokenize_fn, remove_columns=dataset.column_names)

    def get_dataloader(self, shuffle:bool=False):
        self.logger.info(f"Loading dataset {self.dataset_name} for {self.language} for split {self.split}.")
        self.dataset_obj = HFDataset(
            dataset_name=self.dataset_name,
            model_name = self.model_name,
            text_field=self.text_field,
            split=self.split,
            language=self.language,
            max_length=self.max_length,
            logger=self.logger
        )
        self.collator = DataCollatorWithPadding(self.dataset_obj.tokenizer)
        self.dataset = self.dataset_obj.tokenized_dataset
        self.logger.info("Creating DataLoader...")
        self.dataloader = DataLoader(self.dataset,
                          batch_size=self.batch_size,
                          shuffle=shuffle,
                          num_workers=self.num_workers,collate_fn=self.collator)



