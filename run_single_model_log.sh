#!/bin/bash

# for time experiment

model=$1
model_name=$2
model_config=$3
model_config_name=$4
dataset=$5
dataset_config=$6
dataset_name=$7
param=$8

python run_recbole.py --model=$model --dataset=${dataset} --config_files=./recbole/properties/dataset/${dataset_config}.yaml ${model_config//+/ } --epochs=3 --dropout_prob=0 --learning_rate=0.001 --train_batch_size=32 --efficient_mode='None' 
echo "~/anaconda3/bin/python run_hyper.py --model=$model --dataset=${dataset} --config_files=./recbole/properties/dataset/${dataset_config}.yaml ${model_config//+/ } --epochs=3 --dropout_prob=0 --learning_rate=0.001 --train_batch_size=32 --efficient_mode='None'"
