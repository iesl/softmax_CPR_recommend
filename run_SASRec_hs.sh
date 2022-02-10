#!/bin/bash

#~/anaconda3/bin/
#conda activate hschang
#~/anaconda3/envs/hschang/bin/python ./run_recbole.py --model=SASRec --dataset=steam --config_files=./recbole/properties/dataset/steam.yaml
#~/anaconda3/envs/hschang/bin/python ./run_recbole.py --model=SASRec --dataset=steam --config_files=./recbole/properties/dataset/steam.yaml --n_facet=3 --n_facet_context=1 --n_facet_reranker=1 --n_facet_emb=2 --n_facet_all=7 --n_facet_hidden=2 --n_facet_window=-2 --n_facet_MLP=-1 --efficient_mode='None' #Multi-softmax + context + rerank + local 
#~/anaconda3/envs/hschang/bin/python ./run_recbole.py --model=SASRec --dataset=steam --config_files=./recbole/properties/dataset/steam.yaml --n_facet=3 --n_facet_context=3 --n_facet_reranker=1 --n_facet_all=9 --n_facet_hidden=2 --n_facet_window=-2 --n_facet_MLP=-1 --efficient_mode='None' #Multi-softmax + context + rerank 
#~/anaconda3/envs/hschang/bin/python ./run_recbole.py --model=SASRec --dataset=steam --config_files=./recbole/properties/dataset/steam.yaml --n_facet=3 --n_facet_context=1 --n_facet_reranker=1 --n_facet_all=7 --n_facet_hidden=2 --n_facet_window=-2 --n_facet_MLP=-1 --efficient_mode='None' #Multi-softmax + context + rerank 
#~/anaconda3/envs/hschang/bin/python ./run_recbole.py --model=SASRec --dataset=steam --config_files=./recbole/properties/dataset/steam.yaml --n_facet=3 --n_facet_context=1 --n_facet_all=4 --n_facet_hidden=2 --n_facet_window=-2 --n_facet_MLP=-1 --efficient_mode='None' #Multi-softmax + context

#~/anaconda3/envs/hschang/bin/python ./run_recbole.py --model=SASRec --dataset=steam --config_files=./recbole/properties/dataset/steam.yaml --n_facet=1 --n_facet_context=1 --n_facet_reranker=1 --n_facet_all=5 --n_facet_hidden=2 --n_facet_window=-2 --n_facet_MLP=-1 --efficient_mode='None' #context + rerank 

#~/anaconda3/envs/hschang/bin/python ./run_recbole.py --model=SASRec --dataset=steam --config_files=./recbole/properties/dataset/steam.yaml --n_facet=3 --n_facet_context=1 --n_facet_reranker=1 --n_facet_emb=2 --n_facet_all=7 --n_facet_hidden=2 --n_facet_window=-2 --n_facet_MLP=-1 --efficient_mode='None' --weight_mode=max_logits #Multi-softmax + context + rerank + local 
#~/anaconda3/envs/hschang/bin/python ./run_recbole.py --model=SASRec --dataset=steam --config_files=./recbole/properties/dataset/steam.yaml --n_facet=3 --n_facet_context=1 --n_facet_reranker=1 --n_facet_all=5 --n_facet_hidden=2 --n_facet_window=-2 --n_facet_MLP=-1 --efficient_mode='None' --weight_mode=max_logits #Multi-softmax + context + rerank 
~/anaconda3/envs/hschang/bin/python ./run_recbole.py --model=SASRec --dataset=steam --config_files=./recbole/properties/dataset/steam.yaml --n_facet=3 --n_facet_context=3 --n_facet_reranker=1 --n_facet_all=7 --n_facet_hidden=2 --n_facet_window=-2 --n_facet_MLP=-1 --efficient_mode='None' --weight_mode=max_logits #Multi-softmax + context + rerank 

#~/anaconda3/envs/hschang/bin/python ./run_recbole.py --model=SASRec --dataset=steam --config_files=./recbole/properties/dataset/steam.yaml --n_facet=3 --n_facet_all=3 --n_facet_hidden=2 --n_facet_window=-2 --n_facet_MLP=-1 --efficient_mode='None' --weight_mode=max_logits #MFS - partition
#~/anaconda3/envs/hschang/bin/python ./run_recbole.py --model=SASRec --dataset=steam --config_files=./recbole/properties/dataset/steam.yaml --n_facet=3 --n_facet_all=3 --n_facet_hidden=1 --n_facet_window=0 --n_facet_MLP=0 --efficient_mode='None' --weight_mode=max_logits #MoS

#~/anaconda3/envs/hschang/bin/python ./run_recbole.py --model=SASRec --dataset=steam --config_files=./recbole/properties/dataset/steam.yaml --n_facet=3 --n_facet_all=6 --n_facet_hidden=2 --n_facet_window=-2 --n_facet_MLP=-1 --efficient_mode='even_last_2' #MFS
#~/anaconda3/envs/hschang/bin/python ./run_recbole.py --model=SASRec --dataset=steam --config_files=./recbole/properties/dataset/steam.yaml --n_facet=3 --n_facet_all=3 --n_facet_hidden=2 --n_facet_window=-2 --n_facet_MLP=-1 --efficient_mode='None' #MFS - partition
#~/anaconda3/envs/hschang/bin/python ./run_recbole.py --model=SASRec --dataset=steam --config_files=./recbole/properties/dataset/steam.yaml --n_facet=3 --n_facet_all=3 --n_facet_hidden=1 --n_facet_window=0 --n_facet_MLP=0 --efficient_mode='None' #MoS
#~/anaconda3/envs/hschang/bin/python ./run_recbole.py --model=SASRec --dataset=steam --config_files=./recbole/properties/dataset/steam.yaml --n_facet=1 --n_facet_all=1 --n_facet_hidden=2 --n_facet_window=-2 --n_facet_MLP=-1 --efficient_mode='None' #Softmax + multi-hidden
#~/anaconda3/envs/hschang/bin/python ./run_recbole.py --model=SASRec --dataset=steam --config_files=./recbole/properties/dataset/steam.yaml --n_facet=1 --n_facet_all=1 --n_facet_hidden=1 --n_facet_window=0 --n_facet_MLP=0 --efficient_mode='None' #Softmax
