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

layer_num="$1"

if [ -z "$layer_num" ]; then
  echo "Usage: $0 <layer_num>"
  exit 1
fi

# --- Hardcoded missing (layer, feature) pairs ---
read -r -d '' MISSING << 'EOF'
2 id
2 geo
2 syntax_wals
2 phonology_wals
2 syntax_sswl
2 syntax_ethnologue
2 phonology_ethnologue
2 inventory_ethnologue
2 inventory_phoible_aa
2 inventory_phoible_gm
2 inventory_phoible_saphon
2 inventory_phoible_spa
2 inventory_phoible_ph
2 inventory_phoible_ra
2 inventory_phoible_upsid
2 syntax_knn
2 phonology_knn
2 inventory_knn
2 syntax_average
2 phonology_average
2 inventory_average
4 fam
4 id
4 geo
4 inventory_phoible_upsid
4 syntax_knn
4 phonology_knn
4 inventory_knn
4 syntax_average
4 phonology_average
4 inventory_average
6 fam
6 id
6 geo
6 inventory_phoible_upsid
6 syntax_knn
6 phonology_knn
6 inventory_knn
6 syntax_average
6 phonology_average
6 inventory_average
11 id
11 geo
11 syntax_wals
11 phonology_wals
11 syntax_sswl
11 syntax_ethnologue
11 phonology_ethnologue
11 inventory_ethnologue
11 inventory_phoible_aa
11 inventory_phoible_gm
11 inventory_phoible_saphon
11 inventory_phoible_spa
11 inventory_phoible_ph
11 inventory_phoible_ra
11 inventory_phoible_upsid
11 syntax_knn
11 phonology_knn
11 inventory_knn
11 syntax_average
11 phonology_average
11 inventory_average
12 id
12 geo
15 id
15 geo
15 syntax_wals
15 phonology_wals
15 syntax_sswl
15 syntax_ethnologue
15 phonology_ethnologue
15 inventory_ethnologue
15 inventory_phoible_aa
15 inventory_phoible_gm
15 inventory_phoible_saphon
15 inventory_phoible_spa
15 inventory_phoible_ph
15 inventory_phoible_ra
15 inventory_phoible_upsid
15 syntax_knn
15 phonology_knn
15 inventory_knn
15 syntax_average
15 phonology_average
15 inventory_average
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

CUDA_VISIBLE_DEVICES=1 python run_probing.py \
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