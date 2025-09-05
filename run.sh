python3 main.py --dataset "flores_plus" \
    --languages en es fr \
    --model_path "/home/models/meta-llama_Llama-3.2-1B" \
    --model_name "Llama-3.2-1B" \
    --sae_model "/home/models/sae-Llama-3.2-1B-131k" \
    --text_field "text" \
    --batch_size 16 \
    --split "devtest" \
    --layers "layers.2.mlp"