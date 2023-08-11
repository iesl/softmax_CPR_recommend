#!/bin/bash

#dataset_arr=( ml-1m )
#dataset_name_arr=( ml1m )
#dataset_config_arr=( ml-1m )
#param_arr=( GRU_test )

#dataset_arr=( ml-1m steam retailrocket amazon-beauty Amazon_Video_Games diginetica Twitch-100k algebra2008_2009 gowalla ml-10m tmall-buy yelp2018 Amazon_Books yoochoose-clicks )
#dataset_name_arr=( ml1m steam retailrocket beauty game diginetica twitch algebra gowalla ml10m tmall yelp book yoochoose )
#dataset_config_arr=( ml-1m steam retailrocket amazon-beauty amazon-beauty diginetica twitch algebra gowalla ml-1m tmall-buy yelp2018 amazon-book yoochoose )
#param_arr=( GRU_test GRU_test GRU_test GRU_test GRU_test GRU_test GRU_mem3_fast GRU_mem3_fast GRU_mem4_fast GRU_fast GRU_mem4_fast GRU_mem3_fast GRU_mem4_fast GRU_fast )
#param_arr=( ours_test ours_test ours_test ours_test ours_test ours_test ours_mem3_fast ours_mem3_fast ours_mem4_fast ours_fast ours_mem4_fast ours_mem3_fast ours_mem4_fast ours_fast )

#dataset_arr=( ml-1m steam amazon-beauty Amazon_Video_Games Twitch-100k algebra2008_2009 gowalla ml-10m tmall-buy yelp2018 Amazon_Books yoochoose-clicks )
#dataset_name_arr=( ml1m steam beauty game twitch algebra gowalla ml10m tmall yelp book yoochoose )
#dataset_config_arr=( ml-1m steam amazon-beauty amazon-beauty twitch algebra gowalla ml-1m tmall-buy yelp2018 amazon-book yoochoose )
#param_arr=( GRU_test GRU_test GRU_test GRU_test GRU_mem3_fast GRU_mem3_fast GRU_mem4_fast GRU_fast GRU_mem4_fast GRU_mem3_fast GRU_mem4_fast GRU_fast )
#param_arr=( ours_test ours_test ours_test ours_test ours_mem3_fast ours_mem3_fast ours_mem4_fast ours_fast ours_mem4_fast ours_mem3_fast ours_mem4_fast ours_fast )

#dataset_arr=( ml-1m amazon-beauty Amazon_Video_Games )
#dataset_name_arr=( ml1m beauty game )
#dataset_config_arr=( ml-1m amazon-beauty amazon-beauty )
#param_arr=( GRU_ml1m GRU_beauty GRU_game )
#param_arr=( SAS_ml1m SAS_beauty SAS_game )

#dataset_arr=( steam )
#dataset_name_arr=( steam)
#dataset_config_arr=( steam )
#param_arr=( GRU_steam_128 )
#param_arr=( SAS_steam_128 )

dataset_arr=( ml-1m steam amazon-beauty Amazon_Video_Games )
dataset_name_arr=( ml1m steam beauty game )
dataset_config_arr=( ml-1m steam amazon-beauty amazon-beauty )
param_arr=( GRU_ml1m GRU_steam GRU_beauty GRU_game )
#param_arr=( SAS_ml1m SAS_steam SAS_beauty SAS_game )
#param_arr=( ours_test2 ours_test2 ours_test2 ours_test2 )

#dataset_arr=( steam )
#dataset_name_arr=( steam )
#dataset_config_arr=( steam )
#param_arr=( GRU_test )

#dataset_arr=( Amazon_Books )
#dataset_name_arr=( book )
#dataset_config_arr=( amazon-book )
#param_arr=( GRU_test )
#param_arr=( ours_test )

#dataset_arr=( tmall-buy Twitch-100k ml-10m Amazon_Books yoochoose-clicks)
#dataset_name_arr=( tmall twitch ml10m book yoochoose )
#dataset_config_arr=( tmall-buy twitch ml-1m amazon-book yoochoose)
#param_arr=( GRU_mem4_fast GRU_mem3_fast GRU_fast GRU_mem4_fast GRU_fast)
#param_arr=( GRU_mem4_fast_d01 GRU_mem3_fast_d01 GRU_fast_d01 GRU_mem4_fast_d01 GRU_fast_d01)

#param_arr=( ours_mem4_fast ours_mem3_fast ours_fast ours_mem4_fast ours_fast)

total=${#dataset_arr[*]}

#model=SASRec
#model_name=SAS
model=GRU4Rec_Ours
model_name=GRU
#model=GRU4Rec
#model_name=GRUORG
#model=RepeatNet
#model_name=RepeatNet
#model_name=GRU-d02c
#model_config="--n_facet=1" #dummy for RepeatNet
#model_config_name="softmax_bias_fixed" #for RepeatNet
#model_config="--n_facet=1+--n_facet_context=1+--n_facet_reranker=1+--n_facet_emb=2+--n_facet_all=7+--n_facet_hidden=1+--n_facet_window=0+--n_facet_MLP=0+--context_norm=1+--partition_merging_mode=half+--reranker_merging_mode=add"
#model_config_name="softmax_CPR_half_Radd"
#model_config="--n_facet=1+--n_facet_context=1+--n_facet_reranker=1+--n_facet_emb=2+--n_facet_all=5+--n_facet_hidden=1+--n_facet_window=0+--n_facet_MLP=0+--context_norm=1+--reranker_CAN_NUM=100"
#model_config_name="softmax_CPR_replace_R100"
#model_config="--n_facet=1+--n_facet_context=1+--n_facet_reranker=1+--n_facet_emb=2+--n_facet_all=5+--n_facet_hidden=2+--n_facet_window=-2+--n_facet_MLP=-1+--context_norm=1+--partition_merging_mode=half+--reranker_CAN_NUM=100+--reranker_merging_mode=add"
#model_config_name="softmax_CPR_Mi_half_R100add"
#model_config="--n_facet=1+--n_facet_context=1+--n_facet_reranker=1+--n_facet_emb=2+--n_facet_all=5+--n_facet_hidden=2+--n_facet_window=-2+--n_facet_MLP=-1+--context_norm=1+--partition_merging_mode=replace+--reranker_CAN_NUM=100"
#model_config_name="softmax_CPR_Mi_replace_R100_init"
#model_config="--n_facet=1+--n_facet_context=1+--n_facet_reranker=1+--n_facet_emb=2+--n_facet_all=7+--n_facet_hidden=1+--n_facet_window=0+--n_facet_MLP=0+--context_norm=1"
#model_config_name="softmax_CPR_replace_bias_init"
#model_config="--n_facet=1+--n_facet_context=1+--n_facet_reranker=1+--n_facet_emb=2+--n_facet_all=5+--n_facet_hidden=2+--n_facet_window=-2+--n_facet_MLP=-1+--context_norm=1+--partition_merging_mode=replace+--reranker_CAN_NUM=100"
#model_config_name="softmax_CPR_Mi_replace_R100"
#model_config="--n_facet=1+--n_facet_all=4+--n_facet_context=1+--n_facet_emb=2+--n_facet_hidden=1+--n_facet_window=0+--n_facet_MLP=0+--context_norm=1"
#model_config_name="softmax_CP_replace"
#model_config="--n_facet=1+--n_facet_all=2+--n_facet_context=1+--n_facet_hidden=1+--n_facet_window=0+--n_facet_MLP=0+--context_norm=1"
#model_config_name="softmax_C_replace"

#model_config="--n_facet=3+--n_facet_all=3+--n_facet_hidden=1+--n_facet_window=0+--n_facet_MLP=0+--weight_mode=max_logits"
#model_config_name="MoSe_bias_fixed"

#model_config="--n_facet=1+--n_facet_all=1+--n_facet_hidden=1+--n_facet_window=0+--n_facet_MLP=0+--post_remove_context=1"
#model_config_name="softmax_post_bias_fixed"

#model_config="--n_facet=1+--n_facet_context=1+--n_facet_reranker=1+--n_facet_emb=2+--n_facet_all=7+--n_facet_hidden=1+--n_facet_window=0+--n_facet_MLP=0+--context_norm=1+--reranker_CAN_NUM=100"
#model_config_name="softmax_CPR_replace_R100"
#model_config="--n_facet=1+--n_facet_context=1+--n_facet_reranker=1+--n_facet_emb=2+--n_facet_all=7+--n_facet_hidden=2+--n_facet_window=-2+--n_facet_MLP=-1+--context_norm=0+--partition_merging_mode=replace"
#model_config_name="softmax_CPRf_Mi_replace"
#model_config="--n_facet=1+--n_facet_all=1+--n_facet_hidden=2+--n_facet_window=-2+--n_facet_MLP=-1"
#model_config_name="softmax_Mi_bias_fixed"
#model_config="--n_facet=1+--n_facet_all=1+--n_facet_hidden=1+--n_facet_window=0+--n_facet_MLP=0"
#model_config_name="softmax_bias_fixed"
#model_config="--n_facet=1+--n_facet_context=1+--n_facet_reranker=1+--n_facet_emb=2+--n_facet_all=7+--n_facet_hidden=2+--n_facet_window=-2+--n_facet_MLP=-1"
#model_config_name="softmax_CPRNf_Mi"
#model_config="--n_facet=1+--n_facet_context=1+--n_facet_reranker=1+--n_facet_emb=2+--n_facet_all=7+--n_facet_hidden=1+--n_facet_window=0+--n_facet_MLP=0"
#model_config_name="softmax_CPRNf"
#model_config="--n_facet=1+--n_facet_context=1+--n_facet_reranker=1+--n_facet_emb=2+--n_facet_all=7+--n_facet_hidden=1+--n_facet_window=0+--n_facet_MLP=0+--context_norm=1"
#model_config_name="softmax_CPR_replace"
#model_config="--n_facet=1+--n_facet_context=1+--n_facet_reranker=1+--n_facet_emb=2+--n_facet_all=7+--n_facet_hidden=2+--n_facet_window=-2+--n_facet_MLP=-1+--context_norm=1+--partition_merging_mode=replace"
#model_config_name="softmax_CPR_Mi_replace"
#model_config="--n_facet=1+--n_facet_context=1+--n_facet_reranker=1+--n_facet_emb=2+--n_facet_all=7+--n_facet_hidden=1+--n_facet_window=0+--n_facet_MLP=0+--context_norm=1"
#model_config_name="softmax_CPR_lR_bias_fixed"
#model_config="--n_facet=1+--n_facet_context=1+--n_facet_reranker=1+--n_facet_emb=2+--n_facet_all=7+--n_facet_hidden=2+--n_facet_window=-2+--n_facet_MLP=-1+--context_norm=1"
#model_config_name="softmax_CPR_Mi_lR_bias_fixed"
#model_config="--n_facet=1+--n_facet_context=1+--n_facet_reranker=1+--n_facet_emb=2+--n_facet_all=7+--n_facet_hidden=1+--n_facet_window=0+--n_facet_MLP=0+--context_norm=1+--reranker_CAN_NUM=100"
#model_config_name="softmax_CPR_lR100_bias_fixed"
#model_config="--n_facet=1+--n_facet_context=1+--n_facet_reranker=1+--n_facet_emb=2+--n_facet_all=7+--n_facet_hidden=2+--n_facet_window=-2+--n_facet_MLP=-1+--context_norm=1+--reranker_CAN_NUM=100+--reranker_merging_mode=replace"
#model_config_name="softmax_CPR_Mi_R100R_bias_fixed"
#model_config="--n_facet=1+--n_facet_context=1+--n_facet_reranker=1+--n_facet_emb=2+--n_facet_all=7+--n_facet_hidden=2+--n_facet_window=-2+--n_facet_MLP=-1+--context_norm=1+--reranker_merging_mode=replace"
#model_config_name="softmax_CPR_Mi_RR_bias_fixed"
#model_config="--n_facet=1+--n_facet_context=1+--n_facet_reranker=1+--n_facet_emb=2+--n_facet_all=7+--n_facet_hidden=1+--n_facet_window=0+--n_facet_MLP=0+--context_norm=1+--reranker_CAN_NUM=100+--reranker_merging_mode=replace"
#model_config_name="softmax_CPR_R100R_bias_fixed"
model_config="--n_facet=1+--n_facet_all=2+--n_facet_context=1+--n_facet_hidden=1+--n_facet_window=0+--n_facet_MLP=0+--context_norm=1"
model_config_name="softmax_C_bias_fixed"
#model_config="--n_facet=1+--n_facet_all=4+--n_facet_context=1+--n_facet_emb=2+--n_facet_hidden=1+--n_facet_window=0+--n_facet_MLP=0+--context_norm=1"
#model_config_name="softmax_CP_bias_fixed"
echo $model_config_name
echo $model_name

for (( i=0; i<=$(( $total -1 )); i++ ))
do
    echo ${param_arr[$i]}
    echo ${dataset_name_arr[$i]}
    #sbatch --time=7-00:00:00 --output=./slog/time_${dataset_name_arr[$i]}_${model_name}_${model_config_name}_%j.out --job-name=multi-rec -G 1 --partition=gypsum-m40 --cpus-per-task=12 --mem=251G run_single_model_log.sh $model $model_name $model_config $model_config_name ${dataset_arr[$i]} ${dataset_config_arr[$i]} ${dataset_name_arr[$i]} ${param_arr[$i]}
    if [ -f "hyper_results_hsz_128/hyper_${model_name}_${dataset_name_arr[$i]}_${model_config_name}" ]; then
        echo "hyper_results_hsz_128/hyper_${model_name}_${dataset_name_arr[$i]}_${model_config_name} exists" 
    else
        sbatch --time=7-00:00:00 --output=./slog/hyper_${dataset_name_arr[$i]}_${model_name}_${model_config_name}_%j.out --job-name=multi-rec -G 1 --partition=gypsum-m40 --cpus-per-task=2 --mem=40G run_single_model.sh $model $model_name $model_config $model_config_name ${dataset_arr[$i]} ${dataset_config_arr[$i]} ${dataset_name_arr[$i]} ${param_arr[$i]}
    fi
done

#~/anaconda3/bin/python run_hyper.py --model=SASRec --dataset=steam --config_files=./recbole/properties/dataset/steam.yaml --params_file=hyper.ours --n_facet=1 --n_facet_all=1 --n_facet_hidden=2 --n_facet_window=-2 --n_facet_MLP=-1 --efficient_mode='None' --hyper_results="hyper_results/hyper_SAS_steam_softmax_Mi" #Softmax
#~/anaconda3/bin/python run_hyper.py --model=GRU4Rec_Ours --dataset=ml-1m --config_files=./recbole/properties/dataset/ml-1m.yaml --params_file=hyper.GRU_test --n_facet=1 --n_facet_all=1 --n_facet_hidden=1 --n_facet_window=-2 --n_facet_MLP=-1 --efficient_mode='None' --hyper_results="hyper_results/hyper_GRU_ml1m_softmax_Mi" #Softmax
#~/anaconda3/bin/python run_hyper.py --model=GRU4Rec_Ours --dataset=ml-1m --config_files=./recbole/properties/dataset/ml-1m.yaml --params_file=hyper.GRU_test --n_facet=1 --n_facet_context=1 --n_facet_reranker=1 --n_facet_emb=2 --n_facet_all=7 --n_facet_hidden=1 --n_facet_window=-2 --n_facet_MLP=-1 --efficient_mode='None' --hyper_results="hyper_results/hyper_GRU_ml1m_softmax_CPR_Mi" #Softmax
