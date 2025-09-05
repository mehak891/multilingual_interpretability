# data/loader.py

import os
import pickle
import json
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, Dataset as HFDatasetType
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorWithPadding
import torch

class MultilingualDatasetManager:
    """Manages multiple parallel multilingual datasets with unified language encoding."""
    
    def __init__(self, 
                 storage_dir: str = "./data/multilingual_datasets",
                 config_path: str = "./data/multilingual_datasets/data_config.json",
                 model_name: str = "bert-base-multilingual-cased",
                 max_length: int = 128,
                 verbose: bool = False):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self.max_length = max_length
        self.verbose = verbose
        
        # Load configurations from JSON
        with open(config_path, 'r') as f:
            config_data = json.load(f)
            self.language_mappings = config_data['language_mappings']
            self.dataset_configs = config_data['datasets']
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.cached_datasets = {}
    
    def _log(self, message: str):
        """Helper method for conditional logging"""
        if self.verbose:
            print(f"[INFO] {message}")
    
    def _get_cache_path(self, dataset_name: str, language: str, split: str) -> Path:
        """Return cache path in format: storage_dir/dataset_name/language-split.pkl"""
        dataset_dir = self.storage_dir / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        return dataset_dir / f"{language}-{split}.pkl"

    def _get_dataset_language_code(self, dataset_name: str, common_lang: str) -> str:
        """Convert common language code to dataset-specific code"""
        if common_lang not in self.language_mappings:
            raise ValueError(f"Language '{common_lang}' not supported. Available: {list(self.language_mappings.keys())}")
        if dataset_name not in self.language_mappings[common_lang]:
            raise ValueError(f"Language '{common_lang}' not supported for dataset '{dataset_name}'")
        return self.language_mappings[common_lang][dataset_name]
    
    def _load_dataset_split(self, dataset_name: str, common_lang: str, split: str) -> Optional[HFDatasetType]:
        """Load a single dataset split with proper language handling"""
        config = self.dataset_configs[dataset_name]
        dataset_lang_code = self._get_dataset_language_code(dataset_name, common_lang)
        
        try:
            if config['language_param_type'] == 'subset':
                # Language as subset parameter (like flores)
                dataset = load_dataset(config['name'], dataset_lang_code, split=split)

            elif config['language_param_type'] == 'flores_plus_subset':
                dataset = load_dataset(config['name'], dataset_lang_code, split=split)

            elif config['language_param_type'] == 'filter':
                # Load full dataset and filter by language
                dataset = load_dataset(config['name'], split=split)
                if 'language' in dataset.column_names:
                    dataset = dataset.filter(lambda x: x['language'] == dataset_lang_code)
                elif 'lang' in dataset.column_names:
                    dataset = dataset.filter(lambda x: x['lang'] == dataset_lang_code)
                    
            elif config['language_param_type'] == 'pair':
                # Translation pairs (like wmt, opus100)
                if common_lang == 'en':
                    # For English, we need to find which pairs it's part of
                    available_pairs = config.get('language_pairs', [])
                    en_pairs = [pair for pair in available_pairs if pair.startswith('en-')]
                    if en_pairs:
                        # Load first available English pair
                        dataset = load_dataset(config['name'], en_pairs[0], split=split)
                    else:
                        return None
                else:
                    # For non-English, look for en-{lang} pair
                    pair = f"en-{common_lang}"
                    if pair in config.get('language_pairs', []):
                        dataset = load_dataset(config['name'], pair, split=split)
                    else:
                        return None
            else:
                # Default: load without language parameter
                dataset = load_dataset(config['name'], split=split)
                
            return dataset
            
        except Exception as e:
            if self.verbose:
                print(f"[ERROR] Failed to load {dataset_name}-{common_lang}-{split}: {e}")
            return None

    def download_and_cache_dataset(self, 
                                 dataset_name: str, 
                                 languages: Optional[List[str]] = None,
                                 splits: Optional[List[str]] = None,
                                 force_redownload: bool = False) -> Dict[str, Dict[str, HFDatasetType]]:
        """Download and cache a dataset for multiple languages using common language codes"""
        if dataset_name not in self.dataset_configs:
            raise ValueError(f"Dataset {dataset_name} not supported. Available: {list(self.dataset_configs.keys())}")
        
        config = self.dataset_configs[dataset_name]
        languages = languages or config['supported_languages']
        splits = splits or config['split_configs']
        
        if dataset_name not in self.cached_datasets:
            self.cached_datasets[dataset_name] = {}
        
        downloaded_data = {}
        
        for common_lang in languages:
            if common_lang not in config['supported_languages']:
                self._log(f"Language {common_lang} not supported for {dataset_name}, skipping")
                continue
                
            lang_data = {}
            for split in splits:
                cache_path = self._get_cache_path(dataset_name, common_lang, split)
                
                # Load from cache if exists and not forcing redownload
                if cache_path.exists() and not force_redownload:
                    self._log(f"Loading cached {dataset_name}-{common_lang}-{split}")
                    with open(cache_path, 'rb') as f:
                        dataset = pickle.load(f)
                        lang_data[split] = dataset
                    continue
                
                # Download fresh data
                self._log(f"Downloading {dataset_name}-{common_lang}-{split}")
                dataset = self._load_dataset_split(dataset_name, common_lang, split)
                
                if dataset is not None:
                    lang_data[split] = dataset
                    self._log(f"Successfully loaded {len(dataset)} examples for {common_lang}-{split}")
                    
                    for ex in dataset.select(range(min(3, len(dataset)))):
                        self._log(f"Example: {ex}")
                    # Cache the split
                    with open(cache_path, 'wb') as f:
                        pickle.dump(dataset, f)
            
            if lang_data:
                self.cached_datasets[dataset_name][common_lang] = lang_data
                downloaded_data[common_lang] = lang_data
        
        return downloaded_data

    def download_all_datasets(self, 
                            languages: Optional[List[str]] = None,
                            force_redownload: bool = False) -> Dict[str, Dict[str, Dict[str, HFDatasetType]]]:
        """Download all configured datasets using common language codes"""
        all_data = {}
        for dataset_name in self.dataset_configs.keys():
            self._log(f"Processing dataset: {dataset_name}")
            all_data[dataset_name] = self.download_and_cache_dataset(
                dataset_name, languages, force_redownload=force_redownload
            )
        return all_data
    
    def get_dataset(self, dataset_name: str, language: str, split: str) -> Optional[HFDatasetType]:
        """Get a specific cached dataset using common language code"""
        try:
            return self.cached_datasets[dataset_name][language][split]
        except KeyError:
            cache_path = self._get_cache_path(dataset_name, language, split)
            if cache_path.exists():
                with open(cache_path, 'rb') as f:
                    dataset = pickle.load(f)
                    if dataset_name not in self.cached_datasets:
                        self.cached_datasets[dataset_name] = {}
                    if language not in self.cached_datasets[dataset_name]:
                        self.cached_datasets[dataset_name][language] = {}
                    self.cached_datasets[dataset_name][language][split] = dataset
                    return dataset
            return None
    
    def create_dataloader(self, 
                         dataset_name: str, 
                         language: str, 
                         split: str,
                         batch_size: int = 32,
                         shuffle: bool = False,
                         num_workers: int = 0) -> Optional[DataLoader]:
        """Create a PyTorch DataLoader using common language code"""
        hf_dataset = self.get_dataset(dataset_name, language, split)
        if hf_dataset is None:
            return None
        
        text_field = self.dataset_configs[dataset_name]['text_fields'][0]
        pytorch_dataset = TokenizedDataset(hf_dataset, self.tokenizer, text_field, self.max_length)
        collator = DataCollatorWithPadding(self.tokenizer)
        
        return DataLoader(
            pytorch_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collator
        )
    
    def get_available_languages(self, dataset_name: Optional[str] = None) -> Union[List[str], Dict[str, List[str]]]:
        """Get available languages using common language codes"""
        if dataset_name:
            return self.dataset_configs[dataset_name]['supported_languages']
        else:
            return {name: config['supported_languages'] for name, config in self.dataset_configs.items()}
    
    def get_available_data(self) -> Dict[str, Dict[str, List[str]]]:
        """Get summary of available cached data using common language codes"""
        summary = {}
        for dataset_name in self.cached_datasets:
            summary[dataset_name] = {}
            for language in self.cached_datasets[dataset_name]:
                summary[dataset_name][language] = list(self.cached_datasets[dataset_name][language].keys())
        return summary
    
    def cleanup_cache(self, dataset_name: Optional[str] = None):
        """Clean up cached data"""
        if dataset_name:
            pattern = f"{dataset_name}_*.pkl"
        else:
            pattern = "*.pkl"
        
        for path in self.storage_dir.glob(pattern):
            path.unlink()
            
    def get_parallel_dataloaders(self, 
                               dataset_name: str,
                               languages: List[str],
                               split: str,
                               batch_size: int = 32,
                               num_workers: int = 0) -> Dict[str, DataLoader]:
        """Create parallel dataloaders for multiple languages - useful for activation analysis"""
        dataloaders = {}
        for lang in languages:
            dl = self.create_dataloader(dataset_name, lang, split, batch_size, False, num_workers)
            if dl is not None:
                dataloaders[lang] = dl
        return dataloaders


class TokenizedDataset(Dataset):
    """PyTorch Dataset for tokenized text data"""
    
    def __init__(self, hf_dataset: HFDatasetType, tokenizer, text_field: str, max_length: int):
        self.hf_dataset = hf_dataset
        self.tokenizer = tokenizer
        self.text_field = text_field
        self.max_length = max_length
    
    def __len__(self):
        return len(self.hf_dataset)
    
    def __getitem__(self, idx):
        text = self.hf_dataset[idx][self.text_field]
        encoded = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {k: v.squeeze(0) for k, v in encoded.items()}


# # Legacy Dataset Classes (integrated from original file)
# class HFDataset(Dataset):
#     """Legacy HF Dataset class for backward compatibility"""
#     def __init__(self, dataset_name: str, model_name: str, text_field: str, 
#                  split: str, language: str, max_length: int, logger):
#         self.dataset_name = dataset_name
#         self.split = split
#         self.logger = logger
#         self.language = language
        
#         try:
#             if self.language:
#                 self.dataset = load_dataset(self.dataset_name, language=self.language, split=self.split)
#             else:
#                 self.dataset = load_dataset(self.dataset_name, split=self.split)
#         except Exception as e:
#             self.logger.error(f"Failed to load dataset: {e}")
#             raise e
            
#         self.text_field = text_field
#         self.max_length = max_length
#         self.tokenizer = AutoTokenizer.from_pretrained(model_name)
#         self.tokenizer.pad_token = self.tokenizer.eos_token
#         self.tokenized_dataset = self.dataset.map(self.tokenize_fn, remove_columns=self.dataset.column_names)
    
#     def __len__(self):
#         return len(self.dataset)

#     def tokenize_fn(self, item):
#         return self.tokenizer(
#             item[f"{self.text_field}"],
#             truncation=True,
#             padding="max_length",
#             max_length=128
#         )

#     def __getitem__(self, idx):  # Fixed the method name
#         text = self.dataset[idx][self.text_field]
#         encoded = self.tokenizer(text=text, truncation=True, padding='max_length',
#                     max_length=self.max_length, return_tensors='pt')
#         return {k: v.squeeze(0) for k, v in encoded.items()}


# class TextFileDataset(Dataset):
#     """Dataset for loading text from local files"""
#     def __init__(self, file_path: str, tokenizer_name: str, max_length: int):
#         if not os.path.isfile(file_path):
#             raise FileNotFoundError(file_path)
#         self.file_path = file_path
#         self.max_length = max_length
#         with open(file_path, 'r') as f:
#             self.lines = [ln.rstrip('\n') for ln in f]
#         self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
#         self.tokenizer.pad_token = self.tokenizer.eos_token

#     def __len__(self):
#         return len(self.lines)

#     def __getitem__(self, idx):
#         text = self.lines[idx]
#         encoded = self.tokenizer(
#             text,
#             truncation=True,
#             padding='max_length',
#             max_length=self.max_length,
#             return_tensors='pt'
#         )
#         return {k: v.squeeze(0) for k, v in encoded.items()}


# # Convenience functions
# def create_manager(storage_dir: str = "./multilingual_datasets",
#                   config_path: str = "dataset_configs.json",
#                   verbose: bool = False) -> MultilingualDatasetManager:
#     """Create a dataset manager"""
#     return MultilingualDatasetManager(storage_dir=storage_dir, config_path=config_path, verbose=verbose)


# def quick_setup(languages: List[str] = None, 
#                datasets: List[str] = None,
#                verbose: bool = False) -> MultilingualDatasetManager:
#     """Quick setup with minimal configuration using common language codes"""
#     manager = create_manager(verbose=verbose)
    
#     if datasets:
#         for dataset_name in datasets:
#             if dataset_name in manager.dataset_configs:
#                 manager.download_and_cache_dataset(dataset_name, languages=languages)
#     else:
#         manager.download_all_datasets(languages=languages)
    
#     return manager


# def build_textfile_dataloader(file_path: str, tokenizer_name: str, batch_size: int, 
#                             max_length: int, num_workers: int = 0) -> DataLoader:
#     """Build dataloader from text file"""
#     dataset = TextFileDataset(file_path=file_path, tokenizer_name=tokenizer_name, max_length=max_length)
#     collator = DataCollatorWithPadding(dataset.tokenizer)
#     return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collator)