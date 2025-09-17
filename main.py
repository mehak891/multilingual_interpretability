import os
import sys
import torch
import gc
import utils.config as config
from data.multiloader import MultilingualDatasetManager
from models import loader as m_loader
from utils.streaming_activation_extractor import StreamingExtractor

sys.path.append(os.path.abspath('.'))
logger = config.get_logger()
args = config.Config()
debug_logger = config.debug_logger

def main():
    debug_logger(f"Starting main with languages: {args.languages}", args.debug)
    debug_logger(f"Device: {args.device}", args.debug)
    debug_logger(f"Model path: {args.model_path}", args.debug)
    debug_logger(f"SAE model: {args.sae_model}", args.debug)
    debug_logger(f"Layers: {args.layers}", args.debug)
    debug_logger(f"Batch size: {args.batch_size}", args.debug)
    debug_logger(f"Max length: {args.max_length}", args.debug)
    debug_logger(f"Ranking method: {getattr(args, 'ranking_method', 'sae_lape')}", args.debug)
    
    if args.languages:
        languages = args.languages
    else:
        languages = ["en", "es"]
    
    # Check minimum language requirement
    if len(languages) < 2:
        logger.error("Analysis requires at least 2 languages for meaningful comparison!")
        logger.error("With only 1 language, all features will have entropy ≈ 0")
        debug_logger(f"ERROR: Only {len(languages)} language(s) specified. Need at least 2.", args.debug)
        return
    
    # Initialize components
    debug_logger("\nInitializing dataset manager...", args.debug)
    dataset_manager = MultilingualDatasetManager(
        model_name=args.model_path,
        max_length=args.max_length,
        verbose=True
    )
    
    debug_logger("Loading model...", args.debug)
    model_loader = m_loader.HFModelLoader(args.model_path, args.model_type, args.device, logger)
    model = model_loader.model
    debug_logger(f"Model loaded: {type(model)}", args.debug)
    debug_logger(f"Model device: {next(model.parameters()).device}", args.debug)
    
    debug_logger("Loading SAE...", args.debug)
    sae_loader = m_loader.SAELoader(args.sae_model, args.layers, args.device, logger)
    saes = sae_loader.sae_model
    debug_logger(f"SAE loaded: {type(saes)}", args.debug)
    debug_logger(f"SAE keys: {list(saes.keys())}", args.debug)
    
    debug_logger("Initializing StreamingExtractor...", args.debug)
    extractor = StreamingExtractor(model=model, saes=saes, device=args.device)
    
    # Process each language
    for lang in languages:
        logger.info(f"Processing language: {lang}")
        debug_logger(f"\n===== Processing {lang} =====", args.debug)
        
        try:
            debug_logger(f"Downloading and caching dataset for {lang}...", args.debug)
            dataset_manager.download_and_cache_dataset(
                args.dataset_name, languages=[lang], splits=[args.split]
            )
            
            debug_logger(f"Creating dataloader for {lang}...", args.debug)
            data_loader = dataset_manager.create_dataloader(
                args.dataset_name, lang, args.split,
                batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                shuffle_words=getattr(args, 'shuffle_words', False), debug=args.debug
            )
            
            debug_logger(f"Dataloader created, type: {type(data_loader)}", args.debug)
            debug_logger("Starting extractor.run()...", args.debug)
            
            extractor.run(data_loader, lang)
                        
            del data_loader
            gc.collect()
            torch.cuda.empty_cache()
            
        except Exception as e:
            logger.error(f"Error processing language {lang}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Check what data was collected
    debug_logger("\n===== Data Collection Summary =====", args.debug)
    debug_logger(f"Languages in extractor: {list(extractor.lang_to_stats.keys())}", args.debug)
    for lang, layers in extractor.lang_to_stats.items():
        debug_logger(f"{lang}: {len(layers)} layers", args.debug)
        for i, layer in enumerate(layers):
            debug_logger(f"  Layer {i}: {layer['num_examples']} examples, {layer['num_tokens']} tokens", args.debug)
            if layer['over_zero_token'] is not None:
                total_activations = layer['over_zero_token'].sum().item()
                active_features = (layer['over_zero_token'] > 0).sum().item()
                debug_logger(f"    Total activations: {total_activations}, Active features: {active_features}", args.debug)
            else:
                debug_logger("    No activation data collected!", args.debug)
    
    # Run analysis based on ranking method
    ranking_method = getattr(args, 'ranking_method', 'sae_lape')
    debug_logger(f"\n===== Running {ranking_method.upper()} Analysis =====", args.debug)
    
    try:
        if ranking_method == 'magnitude':
            final_indices, features_info = extractor.compute_magnitude_ranking(
                top=getattr(args, 'top_k', 100),
                top_per_layer=getattr(args, 'top_per_layer', False),
                apply_filtering=getattr(args, 'apply_filtering', False)  # Default: no filtering (original behavior)
            )
            method_name = "magnitude"
        else:  # Default to sae_lape
            final_indices, features_info = extractor.compute_sae_lape(
                topk_threshold_ratio=getattr(args, 'topk_threshold_ratio', 0.8),
                example_rate=getattr(args, 'example_rate', 0.98),
                top=getattr(args, 'top_k', 100),
                lang_specific=getattr(args, 'lang_specific', True)
            )
            method_name = "sae_lape"
        
        # Check if results are empty
        if not final_indices or len(final_indices) == 0:
            logger.warning("No language-specific features found. This may indicate:")
            logger.warning("  - Insufficient data collected")
            logger.warning("  - Features are too shared across languages")
            logger.warning("  - Filtering criteria are too strict")
            return
        
        # Debug the results
        for i, lang_indices in enumerate(final_indices):
            if i < len(languages):
                lang = languages[i]
                total_features = sum(len(layer_indices) for layer_indices in lang_indices)
                debug_logger(f"{lang}: {total_features} features across {len(lang_indices)} layers", args.debug)
                for layer_idx, layer_indices in enumerate(lang_indices):
                    if len(layer_indices) > 0:
                        debug_logger(f"  Layer {layer_idx}: {len(layer_indices)} features", args.debug)
        
        for lang, info in features_info.items():
            debug_logger(f"{lang} features_info:", args.debug)
            debug_logger(f"  indices: {len(info['indices'])}", args.debug)
            if 'selected_probs' in info:
                debug_logger(f"  selected_probs shape: {info['selected_probs'].shape}", args.debug)
            if 'entropies' in info:
                debug_logger(f"  entropies shape: {info['entropies'].shape}", args.debug)
                if len(info['entropies']) > 0:
                    debug_logger(f"  entropy range: {info['entropies'].min().item():.6f} to {info['entropies'].max().item():.6f}", args.debug)
            if 'avg_activations' in info:
                debug_logger(f"  avg_activations shape: {info['avg_activations'].shape}", args.debug)
                if len(info['avg_activations']) > 0:
                    debug_logger(f"  avg_activations range: {info['avg_activations'].min().item():.6f} to {info['avg_activations'].max().item():.6f}", args.debug)
        
        # Save detailed features by layer and language
        from utils.feature_storage_utils import save_sae_lape_features
        
        save_sae_lape_features(
            final_indices=final_indices,
            features_info=features_info,
            sorted_langs=sorted(extractor.lang_to_stats.keys()),
            model_name=args.model_name,
            layer_names=args.layers,  # Pass the actual layer names
            dataset=args.dataset_name,
            split=args.split,
            method=method_name,
            top_k=getattr(args, 'top_k', 100),
            experiment_tag=getattr(args, 'experiment_tag', '')
        )

    except Exception as e:
        logger.error(f"{ranking_method.upper()} computation failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # Save combined results  
    # if all_results:
    #     combined_results = {
    #         "methods": list(all_results.keys()),
    #         "results": all_results,
    #         "sorted_lang": sorted(extractor.lang_to_stats.keys())
    #     }
        
    #     # Print summary for all methods
    #     print(f"\n[DEBUG] ===== Final Summary for All Methods =====")
    #     sorted_langs = combined_results["sorted_lang"]
        
    #     # for method, method_results in all_results.items():
    #     #     print(f"\n[DEBUG] === {method.upper()} Results ===")
    #     #     final_indices = method_results["final_indices"]
            
    #     #     for i, lang in enumerate(sorted_langs):
    #     #         if i < len(final_indices):
    #     #             num_features = sum(len(layer_features) for layer_features in final_indices[i])
    #     #             logger.info(f"{method} - {lang}: {num_features} language-specific features")
    #     #         else:
    #     #             logger.info(f"{method} - {lang}: 0 language-specific features (no data collected)")
    # else:
    #     logger.error("No methods completed successfully!")
    #     return
            
    # Additional debugging info
    logger.info("Data collection summary:")
    for lang, layers in extractor.lang_to_stats.items():
        total_examples = sum(layer.get("num_examples", 0) for layer in layers)
        total_tokens = sum(layer.get("num_tokens", 0) for layer in layers)
        logger.info(f"  {lang}: {total_examples} examples, {total_tokens} tokens")

if __name__ == "__main__":
    main()