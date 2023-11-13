#!/bin/bash

datasets_mode="all"
#datasets_mode="for_hidden_plot"
#datasets_mode="time"

model_name="GRU"
#model_name="SAS"
#model_name="RepeatNet"

#softmax_mode="softmax"
#softmax_mode="softmax_Mi"
#softmax_mode="softmax_C"
#softmax_mode="softmax_CP"
#softmax_mode="softmax_CPR:100"
softmax_mode="softmax_CPR:100_Mi"
#softmax_mode="softmax_CPR:20,100,500_Mi"
#softmax_mode="MoS"
#softmax_mode="softmax_post"

if [[ "$datasets_mode" == "all" ]]; then
    dataset_arr=( ml-1m steam amazon-beauty Amazon_Video_Games Twitch-100k algebra2008_2009 gowalla ml-10m tmall-buy yelp2018 Amazon_Books yoochoose-clicks )
    dataset_name_arr=( ml1m steam beauty game twitch algebra gowalla ml10m tmall yelp book yoochoose )
    dataset_config_arr=( ml-1m steam amazon-beauty amazon-beauty twitch algebra gowalla ml-1m tmall-buy yelp2018 amazon-book yoochoose )
    if [[ "$model_mode" == "SAS" ]]; then
        param_arr=( ours_test ours_test ours_test ours_test ours_mem3_fast ours_mem3_fast ours_mem4_fast ours_fast ours_mem4_fast ours_mem3_fast ours_mem4_fast ours_fast )
    else
        param_arr=( GRU_test GRU_test GRU_test GRU_test GRU_mem3_fast GRU_mem3_fast GRU_mem4_fast GRU_fast GRU_mem4_fast GRU_mem3_fast GRU_mem4_fast GRU_fast )
    fi
elif [[ "$datasets_mode" == "for_hidden_plot" ]]; then
    dataset_arr=( ml-1m steam amazon-beauty Amazon_Video_Games )
    dataset_name_arr=( ml1m steam beauty game )
    dataset_config_arr=( ml-1m steam amazon-beauty amazon-beauty )
    if [[ "$model_mode" == "SAS" ]]; then
        param_arr=( SAS_ml1m SAS_steam SAS_beauty SAS_game )
    else
        param_arr=( GRU_ml1m GRU_steam GRU_beauty GRU_game )
    fi
elif [[ "$datasets_mode" == "time" ]]; then
    dataset_arr=( Amazon_Books )
    dataset_name_arr=( book )
    dataset_config_arr=( amazon-book )
    if [[ "$model_mode" == "SAS" ]]; then
        param_arr=( ours_test )
    elif [[ "$model_mode" == "GRU" ]]; then
        param_arr=( GRU_test )
    fi
fi

total=${#dataset_arr[*]}

if [[ "$model_name" == "SAS" ]]; then
    model=SASRec
elif [[ "$model_name" == "GRU" ]]; then
    model=GRU4Rec_Ours
elif [[ "$model_name" == "RepeatNet" ]]; then
    model=RepeatNet
fi

model_config_name=$softmax_mode
if [[ "$model_mode" == "RepeatNet" ]]; then
    model_config="--n_facet=1" #dummy for RepeatNet
    model_config_name="softmax" #for RepeatNet
elif [[ "$softmax_mode" == "softmax" ]]; then
    model_config="--n_facet=1+--n_facet_all=1+--n_facet_hidden=1+--n_facet_window=0+--n_facet_MLP=0"
elif [[ "$softmax_mode" == "softmax_Mi" ]]; then
    model_config="--n_facet=1+--n_facet_all=1+--n_facet_hidden=2+--n_facet_window=-2+--n_facet_MLP=-1"
elif [[ "$softmax_mode" == "softmax_C" ]]; then
    model_config="--n_facet=1+--n_facet_all=2+--n_facet_context=1+--n_facet_hidden=1+--n_facet_window=0+--n_facet_MLP=0+--context_norm=1"
elif [[ "$softmax_mode" == "softmax_CP" ]]; then
    model_config="--n_facet=1+--n_facet_all=4+--n_facet_context=1+--n_facet_emb=2+--n_facet_hidden=1+--n_facet_window=0+--n_facet_MLP=0+--context_norm=1"
elif [[ "$softmax_mode" == "softmax_CPR:100" ]]; then
    model_config="--n_facet=1+--n_facet_context=1+--n_facet_reranker=1+--n_facet_emb=2+--n_facet_all=5+--n_facet_hidden=1+--n_facet_window=0+--n_facet_MLP=0+--context_norm=1+--reranker_CAN_NUM=100+--reranker_merging_mode=replace"
elif [[ "$softmax_mode" == "softmax_CPR:100_Mi" ]]; then
    model_config="--n_facet=1+--n_facet_context=1+--n_facet_reranker=1+--n_facet_emb=2+--n_facet_all=5+--n_facet_hidden=2+--n_facet_window=-2+--n_facet_MLP=-1+--context_norm=1+--reranker_CAN_NUM=100+--reranker_merging_mode=replace"
elif [[ "$softmax_mode" == "softmax_CPR:20,100,500_Mi" ]]; then
    model_config="--n_facet=1+--n_facet_context=1+--n_facet_reranker=1+--n_facet_emb=2+--n_facet_all=7+--n_facet_hidden=2+--n_facet_window=-2+--n_facet_MLP=-1+--context_norm=1+--reranker_merging_mode=replace"
elif [[ "$softmax_mode" == "MoS" ]]; then
    model_config="--n_facet=3+--n_facet_all=3+--n_facet_hidden=1+--n_facet_window=0+--n_facet_MLP=0+--weight_mode=max_logits"
elif [[ "$softmax_mode" == "softmax_post" ]]; then
    model_config="--n_facet=1+--n_facet_all=1+--n_facet_hidden=1+--n_facet_window=0+--n_facet_MLP=0+--post_remove_context=1"
fi

echo $model_config_name
echo $model_config
echo $model_name
echo $model

for (( i=0; i<=$(( $total -1 )); i++ ))
do
    echo ${param_arr[$i]}
    echo ${dataset_name_arr[$i]}
    if [[ "$datasets_mode" == "time" ]]; then
        sbatch --partition=gypsum-m40 --output=./slog/time_${dataset_name_arr[$i]}_${model_name}_${model_config_name}_%j.out --job-name=multi-rec -G 1 --cpus-per-task=12 --mem=251G run_single_model_log.sh $model $model_name $model_config $model_config_name ${dataset_arr[$i]} ${dataset_config_arr[$i]} ${dataset_name_arr[$i]} ${param_arr[$i]} #for time experiment
    else
        if [ -f "hyper_results/hyper_${model_name}_${dataset_name_arr[$i]}_${model_config_name}" ]; then
            echo "hyper_results/hyper_${model_name}_${dataset_name_arr[$i]}_${model_config_name} exists" 
        else
            sbatch --partition=gypsum-m40 --output=./slog/hyper_${dataset_name_arr[$i]}_${model_name}_${model_config_name}_%j.out --job-name=multi-rec -G 1 --cpus-per-task=2 --mem=40G run_single_model.sh $model $model_name $model_config $model_config_name ${dataset_arr[$i]} ${dataset_config_arr[$i]} ${dataset_name_arr[$i]} ${param_arr[$i]}
        fi
    fi
done

