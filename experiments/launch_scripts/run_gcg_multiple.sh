#!/bin/bash

export WANDB_MODE=disabled

# Optionally set the cache for transformers
# export TRANSFORMERS_CACHE='YOUR_PATH/huggingface'
# n for the prompt number
export n=5
export model=$1 # llama2 or vicuna
export setup=$2 # behaviors or my

# Create results folder if it doesn't exist
if [ ! -d "../results" ]; then
    mkdir "../results"
    echo "Folder '../results' created."
else
    echo "Folder '../results' already exists."
fi

python -u ../main.py \
    --config="../configs/transfer_${model}.py" \
    --config.attack=gcg \
    --config.train_data="../../data/advbench/harmful_${setup}.csv" \
    --config.result_prefix="../results/transfer_${model}_gcg_${n}_progressive" \
    --config.progressive_goals=True \
    --config.stop_on_success=True \
    --config.num_train_models=1 \
    --config.allow_non_ascii=False \
    --config.n_train_data=1 \
    --config.n_test_data=29 \
    --config.n_steps=1 \
    --config.test_steps=1 \
    --config.batch_size=400