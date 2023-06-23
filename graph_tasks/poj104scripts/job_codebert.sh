#!/bin/bash
cd ..
graph_type=$1
layer=$2
seed=$3
repeat=$4
data_folder=../datasets/poj-104/graphs/exp1

output_dir=../link_prediction/poj/plm/saved_codebertmodels/${repeat}/${graph_type}_layer_$2
mkdir -p ${output_dir}
mkdir ${data_folder}/codebert
echo ${seed} >  ${output_dir}/seed.txt

epoch=2


python link_prediction_codebert_faster.py \
    --output_dir=${output_dir} \
    --data_folder=${data_folder} \
    --graph_type=${graph_type} \
    --config_name=microsoft/codebert-base \
    --model_name_or_path=microsoft/codebert-base \
    --model_name=codebert \
    --token_config=microsoft/codebert-base \
    --do_train \
    --do_test \
    --do_crossing \
    --epoch $epoch \
    --dataset poj-104 \
    --layer ${layer} \
    --max_code_length 512 \
    --train_batch_size 64 \
    --eval_batch_size 32 \
    --learning_rate 1e-3 \
    --max_grad_norm 1.0 \
    --seed ${seed} 2>&1| tee ${output_dir}/train.log

output_dir=../link_prediction/poj/random/saved_codebertmodels/${repeat}/${graph_type}_layer_$2
mkdir -p ${output_dir}
echo ${seed} >  ${output_dir}/seed.txt
python link_prediction_codebert_faster.py \
    --output_dir=${output_dir} \
    --data_folder=${data_folder} \
    --graph_type=${graph_type} \
    --config_name=microsoft/codebert-base \
    --model_name_or_path=microsoft/codebert-base \
    --model_name=codebert \
    --token_config=microsoft/codebert-base \
    --do_train \
    --do_test \
    --do_random \
    --do_crossing \
    --epoch $epoch \
    --layer ${layer} \
    --dataset poj-104 \
    --max_code_length 512 \
    --train_batch_size 64 \
    --eval_batch_size 32 \
    --learning_rate 1e-3 \
    --max_grad_norm 1.0 \
    --seed ${seed} 2>&1| tee ${output_dir}/train.log

