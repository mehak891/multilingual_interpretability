# for layer_num in 11; do
#     CUDA_VISIBLE_DEVICES=3 python run_probing.py \
#     --model_path "/home/models/meta-llama_Llama-3.2-1B" \
#     --model_name "Llama-3.2-1B" \
#     --sae-model "/home/models/sae-Llama-3.2-1B-131k/" \
#     --layers "layers.${layer_num}.mlp" \
#     --langs en de fr it pt hi es ru tr ja ko zh ur bn\
#     --exp flores_plus-shared-flores_plus \
#     --features  'fam' 'id' 'geo' 'syntax_wals' 'phonology_wals' 'syntax_sswl' \
#                     'syntax_ethnologue' 'phonology_ethnologue' 'inventory_ethnologue' \
#                     'inventory_phoible_aa' 'inventory_phoible_gm' \
#                     'inventory_phoible_saphon' 'inventory_phoible_spa' \
#                     'inventory_phoible_ph' 'inventory_phoible_ra' \
#                     'inventory_phoible_upsid' 'syntax_knn' 'phonology_knn' \
#                     'inventory_knn' 'syntax_average' 'phonology_average' \
#                     'inventory_average'\
#     --split dev \
#     --batch-size 16 \
#     --device cuda \
#     --all-neurons 
# done
    
    # --all-neurons \
  #   --save-acts

# en de fr it

# python run_probing.py \
#   --model_path "/home/models/meta-llama_Llama-3.2-1B" \
#   --model_name "Llama-3.2-1B" \
#   --sae-model "/home/models/sae-Llama-3.2-1B-131k/" \
#   --layers "layers.1.mlp" \
#   --langs en de fr it pt hi es ru tr ja ko \
#   --exp jw300-thresh-50-shuffle \
#   --features 'syntax_wals' 'phonology_wals' 'syntax_sswl' \
#                    'syntax_ethnologue' 'phonology_ethnologue' 'inventory_ethnologue' \
#                    'inventory_phoible_aa' 'inventory_phoible_gm' \
#                    'inventory_phoible_saphon' 'inventory_phoible_spa' \
#                    'inventory_phoible_ph' 'inventory_phoible_ra' \
#                    'inventory_phoible_upsid' 'syntax_knn' 'phonology_knn' \
#                    'inventory_knn' 'syntax_average' 'phonology_average' \
#                    'inventory_average' 'fam' 'id' 'geo' \
#   --split train \
#   --batch-size 16 \
#   --device cuda \

#!/usr/bin/env bash

for layer_num in 11; do
# layer_num="$1"

  if [ -z "$layer_num" ]; then
    echo "Usage: $0 <layer_num>"
    exit 1
  fi

  # --- Hardcoded missing (layer, feature) pairs ---
  read -r -d '' MISSING << 'EOF'
2 fam
2 id
4 fam
4 id
6 fam
6 id
11 id
EOF

  # --- Extract features missing for the given layer ---
  missing_features=$(echo "$MISSING" | awk -v L="$layer_num" '$1==L {print $2}')

  if [ -z "$missing_features" ]; then
    echo "✅ No missing features for layer ${layer_num}"
    exit 0
  fi

  echo "⏳ Running missing features for layer $layer_num:"
  echo "$missing_features"
  echo



  CUDA_VISIBLE_DEVICES=3 python run_probing.py \
      --model_path "/home/models/meta-llama_Llama-3.2-1B" \
      --model_name "Llama-3.2-1B" \
      --sae-model "/home/models/sae-Llama-3.2-1B-131k/" \
      --layers "layers.${layer_num}.mlp" \
      --langs en de fr it pt hi es ru tr ja ko zh ur bn \
      --exp flores_plus-shared-flores_plus \
      --features $missing_features \
      --split dev \
      --batch-size 16 \
      --device cuda \
      --all-neurons
done