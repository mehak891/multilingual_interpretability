
for layer_num in {0..15}; do
    echo ">>> Running layer $layer_num"
    CUDA_VISIBLE_DEVICES=0 python stress_analysis.py \
    --base_dir ./identification \
    --k 100 \
    --layers layer_${layer_num} \
    --include dakshina \
    --exclude scratch \
    --experiment_tag "dakshina"
done