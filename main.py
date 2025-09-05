import os
import sys
import torch
import gc
import utils.config as config
from data.multiloader import MultilingualDatasetManager
from models import loader as m_loader
from utils.streaming_activation_extractor import StreamingExtractor
from utils.feature_storage_utils import save_detailed_features_from_extractor


sys.path.append(os.path.abspath('.'))
logger = config.get_logger()
args = config.Config()

def main():
    languages = ["en", "es"]
    
    # Initialize components
    dataset_manager = MultilingualDatasetManager(
        model_name=args.model_path,
        max_length=args.max_length,
        verbose=True
    )
    
    model_loader = m_loader.HFModelLoader(args.model_path, args.model_type, args.device, logger)
    model = model_loader.model
    sae_loader = m_loader.SAELoader(args.sae_model, args.layers, args.device, logger)
    saes = sae_loader.sae_model
    
    # Initialize streaming extractor
    extractor = StreamingExtractor(model=model, saes=saes, device=args.device)
    
    # Process each language
    for lang in languages:
        logger.info(f"Processing language: {lang}")
        
        try:
            dataset_manager.download_and_cache_dataset(
                args.dataset_name, languages=[lang], splits=[args.split]
            )
            
            data_loader = dataset_manager.create_dataloader(
                args.dataset_name, lang, args.split,
                batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
            )
            
            extractor.run(data_loader, lang)
            
            del data_loader
            gc.collect()
            torch.cuda.empty_cache()
            
        except Exception as e:
            logger.error(f"Error processing language {lang}: {e}")
            continue
    
    # Run SAE-LAPE analysis
    # if args.method.lower()=="sae_lape":
    try:
        final_indices, features_info = extractor.compute_sae_lape(
            topk_threshold_ratio=0.8,
            example_rate=0.98,
            top=100,
            lang_specific=True
        )
        
        # Check if results are empty
        if not final_indices or len(final_indices) == 0:
            logger.warning("No language-specific features found. This may indicate:")
            logger.warning("  - Insufficient data collected")
            logger.warning("  - Features are too shared across languages")
            logger.warning("  - Filtering criteria are too strict")
            return
        
        # Save detailed features by layer and language
        save_detailed_features_from_extractor(
            extractor=extractor,
            model_name=args.model_name,
            method="sae_lape_streaming",
            top_k=100
        )

    except Exception as e:
        logger.error(f"SAE-LAPE computation failed: {e}")
        return

    # Save results  
    results = {
        "final_indices": final_indices,
        "features_info": features_info,
        "sorted_lang": sorted(extractor.lang_to_stats.keys())
    }
    
    # output_path = os.path.join(args.save_dir, f"sae_lape_{args.dataset_name}_{args.split}.pt")
    # os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # torch.save(results, output_path)
    # logger.info(f"Saved results: {output_path}")
    
    # Print summary - with bounds checking
    sorted_langs = results["sorted_lang"]
    for i, lang in enumerate(sorted_langs):
        if i < len(final_indices):
            num_features = sum(len(layer_features) for layer_features in final_indices[i])
            logger.info(f"{lang}: {num_features} language-specific features")
        else:
            logger.info(f"{lang}: 0 language-specific features (no data collected)")
            
    # Additional debugging info
    logger.info("Data collection summary:")
    for lang, layers in extractor.lang_to_stats.items():
        total_examples = sum(layer.get("num_examples", 0) for layer in layers)
        total_tokens = sum(layer.get("num_tokens", 0) for layer in layers)
        logger.info(f"  {lang}: {total_examples} examples, {total_tokens} tokens")

if __name__ == "__main__":
    main()