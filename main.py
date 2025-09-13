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

def main():
    print(f"[DEBUG] Starting main with languages: {args.languages}")
    print(f"[DEBUG] Device: {args.device}")
    print(f"[DEBUG] Model path: {args.model_path}")
    print(f"[DEBUG] SAE model: {args.sae_model}")
    print(f"[DEBUG] Layers: {args.layers}")
    print(f"[DEBUG] Batch size: {args.batch_size}")
    print(f"[DEBUG] Max length: {args.max_length}")
    print(f"[DEBUG] Ranking method: {getattr(args, 'ranking_method', 'sae_lape')}")
    
    if args.languages:
        languages = args.languages
    else:
        languages = ["en", "es"]
    
    # Check minimum language requirement
    if len(languages) < 2:
        logger.error("Analysis requires at least 2 languages for meaningful comparison!")
        logger.error("With only 1 language, all features will have entropy ≈ 0")
        print(f"[DEBUG] ERROR: Only {len(languages)} language(s) specified. Need at least 2.")
        return
    
    # Initialize components
    print(f"\n[DEBUG] Initializing dataset manager...")
    dataset_manager = MultilingualDatasetManager(
        model_name=args.model_path,
        max_length=args.max_length,
        verbose=True
    )
    
    print(f"[DEBUG] Loading model...")
    model_loader = m_loader.HFModelLoader(args.model_path, args.model_type, args.device, logger)
    model = model_loader.model
    print(f"[DEBUG] Model loaded: {type(model)}")
    print(f"[DEBUG] Model device: {next(model.parameters()).device}")
    
    print(f"[DEBUG] Loading SAE...")
    sae_loader = m_loader.SAELoader(args.sae_model, args.layers, args.device, logger)
    saes = sae_loader.sae_model
    print(f"[DEBUG] SAE loaded: {type(saes)}")
    print(f"[DEBUG] SAE keys: {list(saes.keys())}")
    
    print(f"[DEBUG] Initializing StreamingExtractor...")
    extractor = StreamingExtractor(model=model, saes=saes, device=args.device)
    
    # Process each language
    for lang in languages:
        logger.info(f"Processing language: {lang}")
        print(f"\n[DEBUG] ===== Processing {lang} =====")
        
        try:
            print(f"[DEBUG] Downloading and caching dataset for {lang}...")
            dataset_manager.download_and_cache_dataset(
                args.dataset_name, languages=[lang], splits=[args.split]
            )
            
            print(f"[DEBUG] Creating dataloader for {lang}...")
            data_loader = dataset_manager.create_dataloader(
                args.dataset_name, lang, args.split,
                batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
            )
            
            print(f"[DEBUG] Dataloader created, type: {type(data_loader)}")
            print(f"[DEBUG] Starting extractor.run()...")
            
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
    print(f"\n[DEBUG] ===== Data Collection Summary =====")
    print(f"[DEBUG] Languages in extractor: {list(extractor.lang_to_stats.keys())}")
    for lang, layers in extractor.lang_to_stats.items():
        print(f"[DEBUG] {lang}: {len(layers)} layers")
        for i, layer in enumerate(layers):
            print(f"[DEBUG]   Layer {i}: {layer['num_examples']} examples, {layer['num_tokens']} tokens")
            if layer['over_zero_token'] is not None:
                total_activations = layer['over_zero_token'].sum().item()
                active_features = (layer['over_zero_token'] > 0).sum().item()
                print(f"[DEBUG]     Total activations: {total_activations}, Active features: {active_features}")
            else:
                print(f"[DEBUG]     No activation data collected!")
    
    # Run analysis based on ranking method
    ranking_method = getattr(args, 'ranking_method', 'sae_lape')
    print(f"\n[DEBUG] ===== Running {ranking_method.upper()} Analysis =====")
    
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
                print(f"[DEBUG] {lang}: {total_features} features across {len(lang_indices)} layers")
                for layer_idx, layer_indices in enumerate(lang_indices):
                    if len(layer_indices) > 0:
                        print(f"[DEBUG]   Layer {layer_idx}: {len(layer_indices)} features")
        
        for lang, info in features_info.items():
            print(f"[DEBUG] {lang} features_info:")
            print(f"[DEBUG]   indices: {len(info['indices'])}")
            if 'selected_probs' in info:
                print(f"[DEBUG]   selected_probs shape: {info['selected_probs'].shape}")
            if 'entropies' in info:
                print(f"[DEBUG]   entropies shape: {info['entropies'].shape}")
                if len(info['entropies']) > 0:
                    print(f"[DEBUG]   entropy range: {info['entropies'].min().item():.6f} to {info['entropies'].max().item():.6f}")
            if 'avg_activations' in info:
                print(f"[DEBUG]   avg_activations shape: {info['avg_activations'].shape}")
                if len(info['avg_activations']) > 0:
                    print(f"[DEBUG]   avg_activations range: {info['avg_activations'].min().item():.6f} to {info['avg_activations'].max().item():.6f}")
        
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