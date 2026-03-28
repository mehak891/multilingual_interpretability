CUDA_VISIBLE_DEVICES=1 python3 main.py --dataset "europarl" \
    --split "train" \
    --ranking_method "sae_lape" \
    --languages en es de it \
    --model_path "/home/models/meta-llama_Llama-3.2-1B" \
    --model_name "Llama-3.2-1B" \
    --sae_model "/home/models/sae-Llama-3.2-1B-131k" \
    --text_field "text" \
    --batch_size 2 \
    --layers layers.14.mlp \
    --experiment_tag "scratch" \
    --debug

# python3 main.py --dataset "flores_plus" \
#     --split "dev" \
#     --ranking_method "sae_lape" \
#     --languages en de \
#     --model_path "/home/models/meta-llama_Llama-3.2-1B" \
#     --model_name "Llama-3.2-1B" \
#     --sae_model "/home/models/sae-Llama-3.2-1B-131k" \
#     --text_field "text" \
#     --batch_size 16 \
#     --layers "layers.0.mlp" \
#     --experiment_tag "sae_lape_dev" \
#     --shuffle_words False
