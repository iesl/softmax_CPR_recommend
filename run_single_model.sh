#!/bin/bash

model=$1
model_name=$2
model_config=$3
model_config_name=$4
dataset=$5
dataset_config=$6
dataset_name=$7
param=$8

python run_hyper.py --model=$model --dataset=${dataset} --config_files=./recbole/properties/dataset/${dataset_config}.yaml --params_file=hyper.${param} ${model_config//+/ } --efficient_mode='None' --hyper_results="hyper_results_hsz_128/hyper_${model_name}_${dataset_name}_${model_config_name}"
echo "python run_hyper.py --model=$model --dataset=${dataset} --config_files=./recbole/properties/dataset/${dataset_config}.yaml --params_file=hyper.${param} ${model_config//+/ } --efficient_mode='None' --hyper_results='hyper_results_hsz_128/hyper_${model_name}_${dataset_name}_${model_config_name}'"
