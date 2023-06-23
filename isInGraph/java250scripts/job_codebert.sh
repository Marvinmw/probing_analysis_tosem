#!/bin/bash
cd ..
graph_type=$1
seed=$3
repeat=$4
data_folder=../datasets/java250/isINGraphs
output_dir=../isInGraph/java250/plm/saved_codebertmodels/${repeat}/${graph_type}_$2

mkdir -p ${output_dir}
echo $seed >  ${output_dir}/seed.txt
mkdir ${data_folder}/codebert
python link_prediction_codebert_faster.py \
    --output_dir=${output_dir} \
    --data_folder=${data_folder} \
    --graph_type=${graph_type} \
    --config_name=microsoft/codebert-base \
    --model_name_or_path=microsoft/codebert-base \
    --token_config=microsoft/codebert-base \
    --model_name=codebert \
    --do_train \
    --do_test \
    --dataset java250 \
    --do_crossing \
    --epoch 1 \
    --layer $2 \
    --max_code_length 512 \
    --train_batch_size 64 \
    --eval_batch_size 32 \
    --learning_rate 1e-3 \
    --max_grad_norm 1.0 \
   --seed $seed 2>&1| tee ${output_dir}/train.log

output_dir=../isInGraph/java250/random/saved_codebertmodels/${repeat}/${graph_type}_$2
mkdir -p ${output_dir}
echo $seed >  ${output_dir}/seed.txt
python link_prediction_codebert_faster.py \
    --output_dir=${output_dir} \
    --data_folder=${data_folder} \
    --graph_type=$graph_type \
    --config_name=microsoft/codebert-base \
    --model_name_or_path=microsoft/codebert-base \
    --token_config=microsoft/codebert-base \
    --model_name=codebert \
    --do_train \
    --do_test \
    --do_random \
    --do_crossing \
    --dataset java250 \
    --epoch 1 \
    --layer $2 \
    --max_code_length 512 \
    --train_batch_size 64 \
    --eval_batch_size 32 \
    --learning_rate 1e-3 \
    --max_grad_norm 1.0 \
    --seed ${seed} 2>&1| tee ${output_dir}/train.log