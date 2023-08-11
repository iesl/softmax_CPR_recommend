#!/bin/bash

dataset_arr=( ml-1m steam amazon-beauty Amazon_Video_Games Twitch-100k algebra2008_2009 gowalla ml-10m tmall-buy yelp2018 Amazon_Books yoochoose-clicks )
dataset_name_arr=( ml1m steam beauty game twitch algebra gowalla ml10m tmall yelp book yoochoose )
dataset_config_arr=( ml-1m steam amazon-beauty amazon-beauty twitch algebra gowalla ml-1m tmall-buy yelp2018 amazon-book yoochoose )
param_arr=( GRU_test GRU_test GRU_test GRU_test GRU_mem3_fast GRU_mem3_fast GRU_mem4_fast GRU_fast GRU_mem4_fast GRU_mem3_fast GRU_mem4_fast GRU_fast )
#param_arr=( ours_test ours_test ours_test ours_test ours_mem3_fast ours_mem3_fast ours_mem4_fast ours_fast ours_mem4_fast ours_mem3_fast ours_mem4_fast ours_fast )

#dataset_arr=( ml-1m steam amazon-beauty Amazon_Video_Games )
#dataset_name_arr=( ml1m steam beauty game )
#dataset_config_arr=( ml-1m steam amazon-beauty amazon-beauty )
#param_arr=( GRU_ml1m GRU_steam GRU_beauty GRU_game )
#param_arr=( SAS_ml1m SAS_steam SAS_beauty SAS_game )

#dataset_arr=( Amazon_Books )
#dataset_name_arr=( book )
#dataset_config_arr=( amazon-book )
#param_arr=( GRU_test )
#param_arr=( ours_test )

total=${#dataset_arr[*]}

#model=SASRec
#model_name=SAS
model=GRU4Rec_Ours
model_name=GRU
#model=RepeatNet
#model_name=RepeatNet

#model_config="--n_facet=1" #dummy for RepeatNet
#model_config_name="softmax_bias_fixed" #for RepeatNet
#model_config="--n_facet=3+--n_facet_all=3+--n_facet_hidden=1+--n_facet_window=0+--n_facet_MLP=0+--weight_mode=max_logits"
#model_config_name="MoSe_bias_fixed"
#model_config="--n_facet=1+--n_facet_all=1+--n_facet_hidden=1+--n_facet_window=0+--n_facet_MLP=0+--post_remove_context=1"
#model_config_name="softmax_post_bias_fixed"

#model_config="--n_facet=1+--n_facet_all=1+--n_facet_hidden=2+--n_facet_window=-2+--n_facet_MLP=-1"
#model_config_name="softmax_Mi_bias_fixed"
#model_config="--n_facet=1+--n_facet_all=1+--n_facet_hidden=1+--n_facet_window=0+--n_facet_MLP=0"
#model_config_name="softmax_bias_fixed"
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
    if [ -f "hyper_results_hsz_128/hyper_${model_name}_${dataset_name_arr[$i]}_${model_config_name}" ]; then
        echo "hyper_results_hsz_128/hyper_${model_name}_${dataset_name_arr[$i]}_${model_config_name} exists" 
    else
        ./run_single_model.sh $model $model_name $model_config $model_config_name ${dataset_arr[$i]} ${dataset_config_arr[$i]} ${dataset_name_arr[$i]} ${param_arr[$i]}
    fi
done

