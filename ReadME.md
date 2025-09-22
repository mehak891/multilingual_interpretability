Multilingual Interpretability

1. Running language neuron detection methods: The languages should be a subset of the supported languages of the dataset mentioned in `data/multilingual_datasets/data_config.json`. Same goes for dataset split.You can just modify the `run.sh` file to run your experiments. 

```shell
bash run.sh
```

Full example:

```shell
CUDA_VISIBLE_DEVICES=1 python3 main.py \
    --dataset "europarl" \
    --split "train" \
    --ranking_method "sae_lape" \ # sae_lape or magnitude
    --languages en de fr it pt es \
    --model_path "/home/models/meta-llama_Llama-3.2-1B" \
    --model_name "Llama-3.2-1B" \
    --sae_model "/home/models/sae-Llama-3.2-1B-131k" \
    --text_field "text" \ # do not change
    --batch_size 16 \
    --layers "layers.1.mlp" \ # non-comma list
    --experiment_tag "thresh-80" \ # the suffix of the folder inside layer_x
    --suffle_words \ # to shuffle word order, optional
    --debug \ # to enable all stderr outputs, optional

# en de fr it pt hi es ru tr ja ko zh etc
```

2. Running overlap analyses of different configs
    - The `--layers` argument is required.

Example:

```shell
python3 stress_analysis.py --layers layer_1 layer_2
```

3. Running probing experiments (currently dataset for activation extraction is hardcoded to be jw300). Run the below shell script to execute.

```bash
CUDA_VISIBLE_DEVICES=0 bash run_l2v_probe.sh
```

Full example:

```shell
python run_probing.py \
  --model_path "/home/models/meta-llama_Llama-3.2-1B" \
  --model_name "Llama-3.2-1B" \
  --sae-model "/home/models/sae-Llama-3.2-1B-131k/" \
  --layers "layers.1.mlp" \
  --langs en de fr it pt hi es ru tr ja ko \ # union of all languages needed to be analysed
  --exp jw300-thresh-50 \ # according to folder inside layer_x
  --features 'syntax_wals' 'phonology_wals' 'syntax_sswl' \
                   'syntax_ethnologue' 'phonology_ethnologue' 'inventory_ethnologue' \
                   'inventory_phoible_aa' 'inventory_phoible_gm' \
                   'inventory_phoible_saphon' 'inventory_phoible_spa' \
                   'inventory_phoible_ph' 'inventory_phoible_ra' \
                   'inventory_phoible_upsid' 'syntax_knn' 'phonology_knn' \
                   'inventory_knn' 'syntax_average' 'phonology_average' \
                   'inventory_average' 'fam' 'id' 'geo' \
  --split train \
  --batch-size 16 \
  --device cuda \
  --save-acts \ #optional, to save activations, not preferred
```

4. Analysing and visualizing probing results

The results_dir should be the saved folder inside layer_x

```shell
python vis_probes.py --results_dir jw300-thresh-50
```

1. Running language neuron detection methods: The languages should be a subset of the supported languages of the dataset mentioned in `data/multilingual_datasets/data_config.json`. Same goes for dataset split.You can just modify the `run.sh` file to run your experiments. 

```shell
bash run.sh
```

Full example:

```shell
CUDA_VISIBLE_DEVICES=1 python3 main.py \
    --dataset "europarl" \
    --split "train" \
    --ranking_method "sae_lape" \ # sae_lape or magnitude
    --languages en de fr it pt es \
    --model_path "/home/models/meta-llama_Llama-3.2-1B" \
    --model_name "Llama-3.2-1B" \
    --sae_model "/home/models/sae-Llama-3.2-1B-131k" \
    --text_field "text" \ # do not change
    --batch_size 16 \
    --layers "layers.1.mlp" \ # non-comma list
    --experiment_tag "thresh-80" \ # the suffix of the folder inside layer_x
    --suffle_words \ # to shuffle word order, optional
    --debug \ # to enable all stderr outputs, optional

# en de fr it pt hi es ru tr ja ko zh etc
```

2. Running overlap analyses of different configs
    - The `--layers` argument is required.

Example:

```shell
python3 stress_analysis.py --layers layer_1 layer_2
```

3. Running probing experiments (currently dataset for activation extraction is hardcoded to be jw300). Run the below shell script to execute.

```bash
CUDA_VISIBLE_DEVICES=0 bash run_l2v_probe.sh
```

Full example:

```shell
python run_probing.py \
  --model_path "/home/models/meta-llama_Llama-3.2-1B" \
  --model_name "Llama-3.2-1B" \
  --sae-model "/home/models/sae-Llama-3.2-1B-131k/" \
  --layers "layers.1.mlp" \
  --langs en de fr it pt hi es ru tr ja ko \ # union of all languages needed to be analysed
  --exp jw300-thresh-50 \ # according to folder inside layer_x
  --features 'syntax_wals' 'phonology_wals' 'syntax_sswl' \
                   'syntax_ethnologue' 'phonology_ethnologue' 'inventory_ethnologue' \
                   'inventory_phoible_aa' 'inventory_phoible_gm' \
                   'inventory_phoible_saphon' 'inventory_phoible_spa' \
                   'inventory_phoible_ph' 'inventory_phoible_ra' \
                   'inventory_phoible_upsid' 'syntax_knn' 'phonology_knn' \
                   'inventory_knn' 'syntax_average' 'phonology_average' \
                   'inventory_average' 'fam' 'id' 'geo' \
  --split train \
  --batch-size 16 \
  --device cuda \
  --save-acts \ #optional, to save activations, not preferred
```

4. Analysing and visualizing probing results

The results_dir should be the saved folder inside layer_x

```shell
python vis_probes.py --results_dir jw300-thresh-50
```