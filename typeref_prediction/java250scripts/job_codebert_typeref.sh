#!/bin/bash
cd ..
layer=$1
seed=$2
repeat=$3

data_folder=../datasets/java250/typeref/
output_dir=./typeref_res/plm/saved_codebertmodels_typeref/${repeat}/layer_$1
mkdir -p ${output_dir}
echo ${seed} > ${output_dir}/seed.txt

mkdir ${data_folder}/codebert
python typeref_prediction_codebert.py \
    --output_dir=${output_dir} \
    --data_folder=${data_folder} \
    --config_name=microsoft/codebert-base \
    --model_name_or_path=microsoft/codebert-base \
    --token_config=microsoft/codebert-base \
    --model_name=codebert \
    --do_kfold \
    --do_crossing \
    --epoch 5 \
    --dataset java250 \
    --layer ${layer} \
    --max_code_length 512 \
    --train_batch_size 64 \
    --eval_batch_size 32 \
    --learning_rate 1e-3 \
    --max_grad_norm 1.0 \
    --seed ${seed} 2>&1| tee ${output_dir}/train.log

output_dir=./typeref_res/random/saved_codebertmodels_typeref/${repeat}/layer_$1
mkdir -p ${output_dir}
echo ${seed} > ${output_dir}/seed.txt
python typeref_prediction_codebert.py \
    --output_dir=${output_dir} \
    --data_folder=${data_folder} \
    --config_name=microsoft/codebert-base \
    --model_name_or_path=microsoft/codebert-base \
    --token_config=microsoft/codebert-base \
    --model_name=codebert \
    --do_kfold \
    --do_crossing \
    --do_random \
    --epoch 5 \
    --dataset java250 \
    --layer ${layer} \
    --max_code_length 512 \
    --train_batch_size 64 \
    --eval_batch_size 32 \
    --learning_rate 1e-3 \
    --max_grad_norm 1.0 \
    --seed ${seed} 2>&1| tee ${output_dir}/train.log