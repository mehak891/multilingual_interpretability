python3 run_probing.py \
    --model_path "/home/models/google-gemma-2-2b-base" \
    --model_name "Gemma-2-2b" \
    --sae-model "gemma-scope-2b-pt-mlp-canonical" \
    --layers $* \
    --langs en es ja de hi ar\
    --exp flores_plus \
    --features 'syntax_wals' 'phonology_wals' 'syntax_sswl' \
                    'syntax_ethnologue' 'phonology_ethnologue' 'inventory_ethnologue' \
                    'inventory_phoible_aa' 'inventory_phoible_gm' \
                    'inventory_phoible_saphon' 'inventory_phoible_spa' \
                    'inventory_phoible_ph' 'inventory_phoible_ra' \
                    'inventory_phoible_upsid' 'syntax_knn' 'phonology_knn' \
                    'inventory_knn' 'syntax_average' 'phonology_average' \
                    'inventory_average' 'fam' 'id' 'geo' \
    --split devtest \
    --batch-size 16 \
    --save-acts \
    --all-neurons