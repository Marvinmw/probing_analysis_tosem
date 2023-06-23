#!/bin/bash

layer=$1
seed=$2
repeat=$3

data_folder=../datasets/java250/postag/
output_dir=./pos_res_java/plm/saved_codebertmodels/${repeat}/layer_$1
mkdir -p ${output_dir}
echo ${seed} > ${output_dir}/seed.txt

mkdir ${data_folder}/codebert
CUDA_LAUNCH_BLOCKING=1 python run.py \
    --output_dir=${output_dir} \
    --data_folder=${data_folder} \
    --config_name=microsoft/codebert-base \
    --model_name_or_path=microsoft/codebert-base \
    --token_config=microsoft/codebert-base \
    --model_name=codebert \
    --epoch 5 \
    --dataset postag \
    --do_train \
    --do_test \
    --do_eval \
    --layer ${layer} \
    --max_code_length 512 \
    --train_batch_size 64 \
    --eval_batch_size 32 \
    --learning_rate 1e-3 \
    --max_grad_norm 1.0 \
    --seed ${seed} 2>&1| tee ${output_dir}/train.log

output_dir=./pos_res_java/random/saved_codebertmodels/${repeat}/layer_$1
mkdir -p ${output_dir}
echo ${seed} > ${output_dir}/seed.txt
CUDA_LAUNCH_BLOCKING=1 python run.py \
    --output_dir=${output_dir} \
    --data_folder=${data_folder} \
    --config_name=microsoft/codebert-base \
    --model_name_or_path=microsoft/codebert-base \
    --token_config=microsoft/codebert-base \
    --model_name=codebert \
    --do_random \
    --do_train \
    --do_test \
    --do_eval \
    --epoch 5 \
    --dataset postag \
    --layer ${layer} \
    --max_code_length 512 \
    --train_batch_size 32 \
    --eval_batch_size 32 \
    --learning_rate 1e-3 \
    --max_grad_norm 1.0 \
    --seed ${seed} 2>&1| tee ${output_dir}/train.log