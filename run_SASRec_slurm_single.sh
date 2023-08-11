#!/bin/bash

#SBATCH --job-name=multi-rec
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
##SBATCH --cpus-per-task=2
#SBATCH --time=7-00:00:00
#SBATCH --mem=251G
##SBATCH --mem=40G
##SBATCH --output=./slog/Steam_softmax_Mi_%j.out
##SBATCH --output=./slog/Steam_GRU_softmax_Mi_%j.out
##SBATCH --output=./slog/beauty_softmax_CPR_Mi_%j.out
##SBATCH --output=./slog/beauty_softmax_Mi_%j.out
##SBATCH --output=./slog/beauty_GRU_softmax_Mi_%j.out
##SBATCH --output=./slog/retailrocket_softmax_Mi_%j.out
##SBATCH --output=./slog/retailrocket_GRU_softmax_Mi_%j.out

##SBATCH --output=./slog/hyper_fashion_SAS_softmax_CPR_Mi_%j.out
##SBATCH --output=./slog/hyper_fashion_RepeatNet_%j.out
##SBATCH --output=./slog/hyper_fashion_GRU_softmax_CPR_Mi_%j.out
##SBATCH --output=./slog/hyper_fashion_GRU_softmax_Mi_%j.out
##SBATCH --output=./slog/hyper_fashion_SAS_softmax_Mi_%j.out

##SBATCH --output=./slog/hyper_ml1m_SAS_softmax_CPR_Mi_%j.out
##SBATCH --output=./slog/hyper_steam_SAS_softmax_CPR_Mi_%j.out
##SBATCH --output=./slog/hyper_retailrocket_SAS_softmax_CPR_Mi_%j.out
##SBATCH --output=./slog/hyper_beauty_SAS_softmax_CPR_Mi_%j.out
##SBATCH --output=./slog/hyper_game_SAS_softmax_CPR_Mi_%j.out
##SBATCH --output=./slog/hyper_diginetica_SAS_softmax_CPR_Mi_%j.out
##SBATCH --output=./slog/hyper_twitch_SAS_softmax_CPR_Mi_%j.out
##SBATCH --output=./slog/hyper_algebra_SAS_softmax_CPR_Mi_%j.out
##SBATCH --output=./slog/hyper_gowalla_SAS_softmax_CPR_Mi_%j.out
##SBATCH --output=./slog/hyper_ml10m_SAS_softmax_CPR_Mi_%j.out
##SBATCH --output=./slog/hyper_tmall_SAS_softmax_CPR_Mi_%j.out
##SBATCH --output=./slog/hyper_yelp2018_SAS_softmax_CPR_Mi_%j.out
##SBATCH --output=./slog/hyper_book_SAS_softmax_CPR_Mi_%j.out
##SBATCH --output=./slog/hyper_yoo_SAS_softmax_CPR_Mi_%j.out

#SBATCH --output=./slog/time_book_RepeatNet_%j.out
##SBATCH --output=./slog/time_steam_RepeatNet_%j.out

##SBATCH --output=./slog/hyper_ml1m_RepeatNet_%j.out
##SBATCH --output=./slog/hyper_steam_RepeatNet_%j.out
##SBATCH --output=./slog/hyper_retailrocket_RepeatNet_%j.out
##SBATCH --output=./slog/hyper_beauty_RepeatNet_%j.out
##SBATCH --output=./slog/hyper_game_RepeatNet_%j.out
##SBATCH --output=./slog/hyper_diginetica_RepeatNet_%j.out
##SBATCH --output=./slog/hyper_twitch_RepeatNet_%j.out
##SBATCH --output=./slog/hyper_algebra_RepeatNet_%j.out
##SBATCH --output=./slog/hyper_gowalla_RepeatNet_%j.out
##SBATCH --output=./slog/hyper_ml10m_RepeatNet_%j.out
##SBATCH --output=./slog/hyper_tmall_RepeatNet_%j.out
##SBATCH --output=./slog/hyper_yelp2018_RepeatNet_%j.out
##SBATCH --output=./slog/hyper_book_RepeatNet_%j.out
##SBATCH --output=./slog/hyper_yoo_RepeatNet_%j.out

##SBATCH --output=./slog/hyper_ml1m_GRU_softmax_CPR_Mi_%j.out
##SBATCH --output=./slog/hyper_steam_GRU_softmax_CPR_Mi_%j.out
##SBATCH --output=./slog/hyper_retailrocket_GRU_softmax_CPR_Mi_%j.out
##SBATCH --output=./slog/hyper_beauty_GRU_softmax_CPR_Mi_%j.out
##SBATCH --output=./slog/hyper_game_GRU_softmax_CPR_Mi_%j.out
##SBATCH --output=./slog/hyper_diginetica_GRU_softmax_CPR_Mi_%j.out
##SBATCH --output=./slog/hyper_twitch_GRU_softmax_CPR_Mi_%j.out
##SBATCH --output=./slog/hyper_algebra_GRU_softmax_CPR_Mi_%j.out
##SBATCH --output=./slog/hyper_gowalla_GRU_softmax_CPR_Mi_%j.out
##SBATCH --output=./slog/hyper_ml10m_GRU_softmax_CPR_Mi_%j.out
##SBATCH --output=./slog/hyper_tmall_GRU_softmax_CPR_Mi_%j.out
##SBATCH --output=./slog/hyper_yelp2018_GRU_softmax_CPR_Mi_%j.out
##SBATCH --output=./slog/hyper_book_GRU_softmax_CPR_Mi_%j.out
##SBATCH --output=./slog/hyper_yoo_GRU_softmax_CPR_Mi_%j.out

##SBATCH --output=./slog/hyper_ml1m_SAS_softmax_Mi_%j.out
##SBATCH --output=./slog/hyper_steam_SAS_softmax_Mi_%j.out
##SBATCH --output=./slog/hyper_retailrocket_SAS_softmax_Mi_%j.out
##SBATCH --output=./slog/hyper_beauty_SAS_softmax_Mi_%j.out
##SBATCH --output=./slog/hyper_game_SAS_softmax_Mi_%j.out
##SBATCH --output=./slog/hyper_diginetica_SAS_softmax_Mi_%j.out
##SBATCH --output=./slog/hyper_twitch_SAS_softmax_Mi_%j.out
##SBATCH --output=./slog/hyper_algebra_SAS_softmax_Mi_%j.out
##SBATCH --output=./slog/hyper_gowalla_SAS_softmax_Mi_%j.out
##SBATCH --output=./slog/hyper_ml10m_SAS_softmax_Mi_%j.out
##SBATCH --output=./slog/hyper_tmall_SAS_softmax_Mi_%j.out
##SBATCH --output=./slog/hyper_yelp2018_SAS_softmax_Mi_%j.out
##SBATCH --output=./slog/hyper_book_SAS_softmax_Mi_%j.out
##SBATCH --output=./slog/hyper_yoo_SAS_softmax_Mi_%j.out

##SBATCH --output=./slog/hyper_ml1m_GRU_softmax_Mi_%j.out
##SBATCH --output=./slog/hyper_steam_GRU_softmax_Mi_%j.out
##SBATCH --output=./slog/hyper_retailrocket_GRU_softmax_Mi_%j.out
##SBATCH --output=./slog/hyper_beauty_GRU_softmax_Mi_%j.out
##SBATCH --output=./slog/hyper_game_GRU_softmax_Mi_%j.out
##SBATCH --output=./slog/hyper_diginetica_GRU_softmax_Mi_%j.out
##SBATCH --output=./slog/hyper_twitch_GRU_softmax_Mi_%j.out
##SBATCH --output=./slog/hyper_algebra_GRU_softmax_Mi_%j.out
##SBATCH --output=./slog/hyper_gowalla_GRU_softmax_Mi_%j.out
##SBATCH --output=./slog/hyper_ml10m_GRU_softmax_Mi_%j.out
##SBATCH --output=./slog/hyper_tmall_GRU_softmax_Mi_%j.out
##SBATCH --output=./slog/hyper_yelp2018_GRU_softmax_Mi_%j.out
##SBATCH --output=./slog/hyper_book_GRU_softmax_Mi_%j.out
##SBATCH --output=./slog/hyper_yoo_GRU_softmax_Mi_%j.out

##SBATCH --output=./slog/hyper_ml1m_SAS_softmax_%j.out
##SBATCH --output=./slog/hyper_steam_SAS_softmax_%j.out
##SBATCH --output=./slog/hyper_retailrocket_SAS_softmax_%j.out
##SBATCH --output=./slog/hyper_beauty_SAS_softmax_%j.out
##SBATCH --output=./slog/hyper_game_SAS_softmax_%j.out
##SBATCH --output=./slog/hyper_diginetica_SAS_softmax_%j.out
##SBATCH --output=./slog/hyper_twitch_SAS_softmax_%j.out
##SBATCH --output=./slog/hyper_algebra_SAS_softmax_%j.out
##SBATCH --output=./slog/hyper_gowalla_SAS_softmax_%j.out
##SBATCH --output=./slog/hyper_ml10m_SAS_softmax_%j.out
##SBATCH --output=./slog/hyper_tmall_SAS_softmax_%j.out
##SBATCH --output=./slog/hyper_yelp2018_SAS_softmax_%j.out
##SBATCH --output=./slog/hyper_book_SAS_softmax_%j.out

##SBATCH --output=./slog/hyper_softmax_CPR_Mi_%j.out


#~/anaconda3/bin/python ./run_recbole.py --model=SASRec --dataset=steam --config_files=./recbole/properties/dataset/steam.yaml --n_facet=1 --n_facet_all=1 --n_facet_hidden=2 --n_facet_window=-2 --n_facet_MLP=-1  

#~/anaconda3/bin/python run_hyper.py --model=GRU4Rec_Ours --dataset=Amazon_Fashion --config_files=./recbole/properties/dataset/amazon-beauty.yaml --params_file=hyper_config/hyper.GRU_test  --n_facet=1 --n_facet_context=1 --n_facet_reranker=1 --n_facet_emb=2 --n_facet_all=7 --n_facet_hidden=1 --n_facet_window=-2 --n_facet_MLP=-1 --efficient_mode='None' --hyper_results="hyper_results/hyper_GRU_fashion_softmax_CPR_Mi"  
#~/anaconda3/bin/python run_hyper.py --model=SASRec --dataset=Amazon_Fashion --config_files=./recbole/properties/dataset/amazon-beauty.yaml --params_file=hyper_config/hyper.ours --n_facet=1 --n_facet_context=1 --n_facet_reranker=1 --n_facet_emb=2 --n_facet_all=7 --n_facet_hidden=2 --n_facet_window=-2 --n_facet_MLP=-1 --efficient_mode='None'  --hyper_results="hyper_results/hyper_SAS_fashion_softmax_CPR_Mi"  
#~/anaconda3/bin/python run_hyper.py --model=RepeatNet --dataset=Amazon_Fashion --config_files=./recbole/properties/dataset/amazon-beauty.yaml --params_file=hyper_config/hyper.GRU_test  --hyper_results="hyper_results/hyper_RepeatNet_fashion"  
#~/anaconda3/bin/python run_hyper.py --model=SASRec --dataset=Amazon_Fashion --config_files=./recbole/properties/dataset/amazon-beauty.yaml --params_file=hyper_config/hyper.ours --n_facet=1 --n_facet_all=1 --n_facet_hidden=2 --n_facet_window=-2 --n_facet_MLP=-1 --efficient_mode='None'  --hyper_results="hyper_results/hyper_SAS_fashion_softmax_Mi"  
#~/anaconda3/bin/python run_hyper.py --model=GRU4Rec_Ours --dataset=Amazon_Fashion --config_files=./recbole/properties/dataset/amazon-beauty.yaml --params_file=hyper_config/hyper.GRU_test  --n_facet=1 --n_facet_all=1 --n_facet_hidden=1 --n_facet_window=-2 --n_facet_MLP=-1 --efficient_mode='None' --hyper_results="hyper_results/hyper_GRU_fashion_softmax_Mi"  


#~/anaconda3/bin/python run_hyper.py --model=SASRec --dataset=ml-1m --config_files=./recbole/properties/dataset/ml-1m.yaml --params_file=hyper_config/hyper.ours --n_facet=1 --n_facet_context=1 --n_facet_reranker=1 --n_facet_emb=2 --n_facet_all=7 --n_facet_hidden=2 --n_facet_window=-2 --n_facet_MLP=-1 --efficient_mode='None' --hyper_results="hyper_results/hyper_SAS_ml1m_softmax_CPR_Mi"  
#~/anaconda3/bin/python run_hyper.py --model=SASRec --dataset=steam --config_files=./recbole/properties/dataset/steam.yaml --params_file=hyper_config/hyper.ours --n_facet=1 --n_facet_context=1 --n_facet_reranker=1 --n_facet_emb=2 --n_facet_all=7 --n_facet_hidden=2 --n_facet_window=-2 --n_facet_MLP=-1 --efficient_mode='None' --hyper_results="hyper_results/hyper_SAS_steam_softmax_CPR_Mi"  
#~/anaconda3/bin/python run_hyper.py --model=SASRec --dataset=retailrocket --config_files=./recbole/properties/dataset/retailrocket.yaml --params_file=hyper_config/hyper.ours --n_facet=1 --n_facet_context=1 --n_facet_reranker=1 --n_facet_emb=2 --n_facet_all=7 --n_facet_hidden=2 --n_facet_window=-2 --n_facet_MLP=-1 --efficient_mode='None' --hyper_results="hyper_results/hyper_SAS_retailrocket_softmax_CPR_Mi"  
#~/anaconda3/bin/python run_hyper.py --model=SASRec --dataset=amazon-beauty --config_files=./recbole/properties/dataset/amazon-beauty.yaml --params_file=hyper_config/hyper.ours --n_facet=1 --n_facet_context=1 --n_facet_reranker=1 --n_facet_emb=2 --n_facet_all=7 --n_facet_hidden=2 --n_facet_window=-2 --n_facet_MLP=-1 --efficient_mode='None'  --hyper_results="hyper_results/hyper_SAS_beauty_softmax_CPR_Mi"  
#~/anaconda3/bin/python run_hyper.py --model=SASRec --dataset=Amazon_Video_Games --config_files=./recbole/properties/dataset/amazon-beauty.yaml --params_file=hyper_config/hyper.ours --n_facet=1 --n_facet_context=1 --n_facet_reranker=1 --n_facet_emb=2 --n_facet_all=7 --n_facet_hidden=2 --n_facet_window=-2 --n_facet_MLP=-1 --efficient_mode='None'  --hyper_results="hyper_results/hyper_SAS_game_softmax_CPR_Mi"  
#~/anaconda3/bin/python run_hyper.py --model=SASRec --dataset=diginetica --config_files=./recbole/properties/dataset/diginetica.yaml --params_file=hyper_config/hyper.ours --n_facet=1 --n_facet_context=1 --n_facet_reranker=1 --n_facet_emb=2 --n_facet_all=7 --n_facet_hidden=2 --n_facet_window=-2 --n_facet_MLP=-1 --efficient_mode='None'  --hyper_results="hyper_results/hyper_SAS_diginetica_softmax_CPR_Mi"  
#~/anaconda3/bin/python run_hyper.py --model=SASRec --dataset=Twitch-100k --config_files=./recbole/properties/dataset/twitch.yaml --params_file=hyper_config/hyper.ours_mem3_fast --n_facet=1 --n_facet_context=1 --n_facet_reranker=1 --n_facet_emb=2 --n_facet_all=7 --n_facet_hidden=2 --n_facet_window=-2 --n_facet_MLP=-1 --efficient_mode='None'  --hyper_results="hyper_results/hyper_SAS_twitch_softmax_CPR_Mi"  
#~/anaconda3/bin/python run_hyper.py --model=SASRec --dataset=algebra2008_2009 --config_files=./recbole/properties/dataset/algebra.yaml --params_file=hyper_config/hyper.ours_mem3_fast --n_facet=1 --n_facet_context=1 --n_facet_reranker=1 --n_facet_emb=2 --n_facet_all=7 --n_facet_hidden=2 --n_facet_window=-2 --n_facet_MLP=-1 --efficient_mode='None'  --hyper_results="hyper_results/hyper_SAS_algebra_softmax_CPR_Mi"  
#~/anaconda3/bin/python run_hyper.py --model=SASRec --dataset=gowalla --config_files=./recbole/properties/dataset/gowalla.yaml --params_file=hyper_config/hyper.ours_mem4_fast --n_facet=1 --n_facet_context=1 --n_facet_reranker=1 --n_facet_emb=2 --n_facet_all=7 --n_facet_hidden=2 --n_facet_window=-2 --n_facet_MLP=-1 --efficient_mode='None'  --hyper_results="hyper_results/hyper_SAS_gowalla_softmax_CPR_Mi"  
#~/anaconda3/bin/python run_hyper.py --model=SASRec --dataset=ml-10m --config_files=./recbole/properties/dataset/ml-1m.yaml --params_file=hyper_config/hyper.ours_fast --n_facet=1 --n_facet_context=1 --n_facet_reranker=1 --n_facet_emb=2 --n_facet_all=7 --n_facet_hidden=2 --n_facet_window=-2 --n_facet_MLP=-1 --efficient_mode='None'  --hyper_results="hyper_results/hyper_SAS_ml-10m_softmax_CPR_Mi"  
#~/anaconda3/bin/python run_hyper.py --model=SASRec --dataset=tmall-buy --config_files=./recbole/properties/dataset/tmall-buy.yaml --params_file=hyper_config/hyper.ours_mem4_fast --n_facet=1 --n_facet_context=1 --n_facet_reranker=1 --n_facet_emb=2 --n_facet_all=7 --n_facet_hidden=2 --n_facet_window=-2 --n_facet_MLP=-1 --efficient_mode='None'  --hyper_results="hyper_results/hyper_SAS_tmall_softmax_CPR_Mi"  
#~/anaconda3/bin/python run_hyper.py --model=SASRec --dataset=yelp2018 --config_files=./recbole/properties/dataset/yelp2018.yaml --params_file=hyper_config/hyper.ours_mem3_fast --n_facet=1 --n_facet_context=1 --n_facet_reranker=1 --n_facet_emb=2 --n_facet_all=7 --n_facet_hidden=2 --n_facet_window=-2 --n_facet_MLP=-1 --efficient_mode='None'  --hyper_results="hyper_results/hyper_SAS_yelp2018_softmax_CPR_Mi"  
#~/anaconda3/bin/python run_hyper.py --model=SASRec --dataset=Amazon_Books --config_files=./recbole/properties/dataset/amazon-book.yaml --params_file=hyper_config/hyper.ours_mem4_fast --n_facet=1 --n_facet_context=1 --n_facet_reranker=1 --n_facet_emb=2 --n_facet_all=7 --n_facet_hidden=2 --n_facet_window=-2 --n_facet_MLP=-1 --efficient_mode='None'  --hyper_results="hyper_results/hyper_SAS_book_softmax_CPR_Mi"  
#~/anaconda3/bin/python run_hyper.py --model=SASRec --dataset=yoochoose-clicks --config_files=./recbole/properties/dataset/yoochoose.yaml --params_file=hyper_config/hyper.ours_fast --n_facet=1 --n_facet_context=1 --n_facet_reranker=1 --n_facet_emb=2 --n_facet_all=7 --n_facet_hidden=2 --n_facet_window=-2 --n_facet_MLP=-1 --efficient_mode='None'  --hyper_results="hyper_results/hyper_SAS_yoochoose_softmax_CPR_Mi"  

#~/anaconda3/bin/python run_recbole.py --model=RepeatNet --dataset=Amazon_Books --config_files=./recbole/properties/dataset/amazon-book.yaml --dropout_prob=0 --learning_rate=0.001 --train_batch_size=32 --hidden_size=64  
#~/anaconda3/bin/python run_recbole.py --model=RepeatNet --dataset=steam --config_files=./recbole/properties/dataset/steam.yaml --dropout_prob=0 --learning_rate=0.001 --train_batch_size=128
#~/anaconda3/bin/python run_hyper.py --model=RepeatNet --dataset=ml-1m --config_files=./recbole/properties/dataset/ml-1m.yaml --params_file=hyper_config/hyper.GRU_test --hyper_results="hyper_results/hyper_RepeatNet_ml1m"
#~/anaconda3/bin/python run_hyper.py --model=RepeatNet --dataset=steam --config_files=./recbole/properties/dataset/steam.yaml --params_file=hyper_config/hyper.GRU_test --hyper_results="hyper_results/hyper_RepeatNet_steam"
#~/anaconda3/bin/python run_hyper.py --model=RepeatNet --dataset=retailrocket --config_files=./recbole/properties/dataset/retailrocket.yaml --params_file=hyper_config/hyper.GRU_test --hyper_results="hyper_results/hyper_RepeatNet_retailrocket"
#~/anaconda3/bin/python run_hyper.py --model=RepeatNet --dataset=amazon-beauty --config_files=./recbole/properties/dataset/amazon-beauty.yaml --params_file=hyper_config/hyper.GRU_test --hyper_results="hyper_results/hyper_RepeatNet_amazon-beauty"
#~/anaconda3/bin/python run_hyper.py --model=RepeatNet --dataset=Amazon_Video_Games --config_files=./recbole/properties/dataset/amazon-beauty.yaml --params_file=hyper_config/hyper.GRU_test  --hyper_results="hyper_results/hyper_RepeatNet_game"  
#~/anaconda3/bin/python run_hyper.py --model=RepeatNet --dataset=diginetica --config_files=./recbole/properties/dataset/diginetica.yaml --params_file=hyper_config/hyper.GRU_test  --hyper_results="hyper_results/hyper_RepeatNet_diginetica"  
#~/anaconda3/bin/python run_hyper.py --model=RepeatNet --dataset=Twitch-100k --config_files=./recbole/properties/dataset/twitch.yaml --params_file=hyper_config/hyper.GRU_mem3_fast   --hyper_results="hyper_results/hyper_RepeatNet_twitch"  
#~/anaconda3/bin/python run_hyper.py --model=RepeatNet --dataset=algebra2008_2009 --config_files=./recbole/properties/dataset/algebra.yaml --params_file=hyper_config/hyper.GRU_mem3_fast   --hyper_results="hyper_results/hyper_RepeatNet_algebra"  
#~/anaconda3/bin/python run_hyper.py --model=RepeatNet --dataset=gowalla --config_files=./recbole/properties/dataset/gowalla.yaml --params_file=hyper_config/hyper.GRU_mem4_fast   --hyper_results="hyper_results/hyper_RepeatNet_gowalla"  
#~/anaconda3/bin/python run_hyper.py --model=RepeatNet --dataset=ml-10m --config_files=./recbole/properties/dataset/ml-1m.yaml --params_file=hyper_config/hyper.GRU_fast   --hyper_results="hyper_results/hyper_RepeatNet_ml-10m"  
#~/anaconda3/bin/python run_hyper.py --model=RepeatNet --dataset=tmall-buy --config_files=./recbole/properties/dataset/tmall-buy.yaml --params_file=hyper_config/hyper.GRU_mem4_fast   --hyper_results="hyper_results/hyper_RepeatNet_tmall"  
#~/anaconda3/bin/python run_hyper.py --model=RepeatNet --dataset=yelp2018 --config_files=./recbole/properties/dataset/yelp2018.yaml --params_file=hyper_config/hyper.GRU_mem3_fast   --hyper_results="hyper_results/hyper_RepeatNet_yelp2018"  
#~/anaconda3/bin/python run_hyper.py --model=RepeatNet --dataset=Amazon_Books --config_files=./recbole/properties/dataset/amazon-book.yaml --params_file=hyper_config/hyper.GRU_mem4_fast   --hyper_results="hyper_results/hyper_RepeatNet_book" 
#~/anaconda3/bin/python run_hyper.py --model=RepeatNet --dataset=yoochoose-clicks --config_files=./recbole/properties/dataset/yoochoose.yaml --params_file=hyper_config/hyper.GRU_fast   --hyper_results="hyper_results/hyper_RepeatNet_yoochoose" 

#~/anaconda3/bin/python run_hyper.py --model=GRU4Rec_Ours --dataset=ml-1m --config_files=./recbole/properties/dataset/ml-1m.yaml --params_file=hyper_config/hyper.GRU_test --n_facet=1 --n_facet_context=1 --n_facet_reranker=1 --n_facet_emb=2 --n_facet_all=7 --n_facet_hidden=1 --n_facet_window=-2 --n_facet_MLP=-1 --efficient_mode='None' --hyper_results="hyper_results/hyper_GRU_ml1m_softmax_CPR_Mi" 
#~/anaconda3/bin/python run_hyper.py --model=GRU4Rec_Ours --dataset=steam --config_files=./recbole/properties/dataset/steam.yaml --params_file=hyper_config/hyper.GRU_test --n_facet=1 --n_facet_context=1 --n_facet_reranker=1 --n_facet_emb=2 --n_facet_all=7 --n_facet_hidden=1 --n_facet_window=-2 --n_facet_MLP=-1 --efficient_mode='None' --hyper_results="hyper_results/hyper_GRU_steam_softmax_CPR_Mi" 
#~/anaconda3/bin/python run_hyper.py --model=GRU4Rec_Ours --dataset=retailrocket --config_files=./recbole/properties/dataset/retailrocket.yaml --params_file=hyper_config/hyper.GRU_test --n_facet=1 --n_facet_context=1 --n_facet_reranker=1 --n_facet_emb=2 --n_facet_all=7 --n_facet_hidden=1 --n_facet_window=-2 --n_facet_MLP=-1 --efficient_mode='None' --hyper_results="hyper_results/hyper_GRU_retailrocket_softmax_CPR_Mi" 
#~/anaconda3/bin/python run_hyper.py --model=GRU4Rec_Ours --dataset=amazon-beauty --config_files=./recbole/properties/dataset/amazon-beauty.yaml --params_file=hyper_config/hyper.GRU_test --n_facet=1 --n_facet_context=1 --n_facet_reranker=1 --n_facet_emb=2 --n_facet_all=7 --n_facet_hidden=1 --n_facet_window=-2 --n_facet_MLP=-1 --efficient_mode='None' --hyper_results="hyper_results/hyper_GRU_beauty_softmax_CPR_Mi" 
#~/anaconda3/bin/python run_hyper.py --model=GRU4Rec_Ours --dataset=Amazon_Video_Games --config_files=./recbole/properties/dataset/amazon-beauty.yaml --params_file=hyper_config/hyper.GRU_test  --n_facet=1 --n_facet_context=1 --n_facet_reranker=1 --n_facet_emb=2 --n_facet_all=7 --n_facet_hidden=1 --n_facet_window=-2 --n_facet_MLP=-1 --efficient_mode='None' --hyper_results="hyper_results/hyper_GRU_game_softmax_CPR_Mi"  
#~/anaconda3/bin/python run_hyper.py --model=GRU4Rec_Ours --dataset=diginetica --config_files=./recbole/properties/dataset/diginetica.yaml --params_file=hyper_config/hyper.GRU_test  --n_facet=1 --n_facet_context=1 --n_facet_reranker=1 --n_facet_emb=2 --n_facet_all=7 --n_facet_hidden=1 --n_facet_window=-2 --n_facet_MLP=-1 --efficient_mode='None' --hyper_results="hyper_results/hyper_GRU_diginetica_softmax_CPR_Mi"  
#~/anaconda3/bin/python run_hyper.py --model=GRU4Rec_Ours --dataset=Twitch-100k --config_files=./recbole/properties/dataset/twitch.yaml --params_file=hyper_config/hyper.GRU_mem3_fast   --n_facet=1 --n_facet_context=1 --n_facet_reranker=1 --n_facet_emb=2 --n_facet_all=7 --n_facet_hidden=1 --n_facet_window=-2 --n_facet_MLP=-1 --efficient_mode='None' --hyper_results="hyper_results/hyper_GRU_twitch_softmax_CPR_Mi"  
#~/anaconda3/bin/python run_hyper.py --model=GRU4Rec_Ours --dataset=algebra2008_2009 --config_files=./recbole/properties/dataset/algebra.yaml --params_file=hyper_config/hyper.GRU_mem3_fast   --n_facet=1 --n_facet_context=1 --n_facet_reranker=1 --n_facet_emb=2 --n_facet_all=7 --n_facet_hidden=1 --n_facet_window=-2 --n_facet_MLP=-1 --efficient_mode='None' --hyper_results="hyper_results/hyper_GRU_algebra_softmax_CPR_Mi"  
#~/anaconda3/bin/python run_hyper.py --model=GRU4Rec_Ours --dataset=gowalla --config_files=./recbole/properties/dataset/gowalla.yaml --params_file=hyper_config/hyper.GRU_mem4_fast   --n_facet=1 --n_facet_context=1 --n_facet_reranker=1 --n_facet_emb=2 --n_facet_all=7 --n_facet_hidden=1 --n_facet_window=-2 --n_facet_MLP=-1 --efficient_mode='None' --hyper_results="hyper_results/hyper_GRU_gowalla_softmax_CPR_Mi"  
#~/anaconda3/bin/python run_hyper.py --model=GRU4Rec_Ours --dataset=ml-10m --config_files=./recbole/properties/dataset/ml-1m.yaml --params_file=hyper_config/hyper.GRU_fast   --n_facet=1 --n_facet_context=1 --n_facet_reranker=1 --n_facet_emb=2 --n_facet_all=7 --n_facet_hidden=1 --n_facet_window=-2 --n_facet_MLP=-1 --efficient_mode='None' --hyper_results="hyper_results/hyper_GRU_ml-10m_softmax_CPR_Mi"  
#~/anaconda3/bin/python run_hyper.py --model=GRU4Rec_Ours --dataset=tmall-buy --config_files=./recbole/properties/dataset/tmall-buy.yaml --params_file=hyper_config/hyper.GRU_mem4_fast   --n_facet=1 --n_facet_context=1 --n_facet_reranker=1 --n_facet_emb=2 --n_facet_all=7 --n_facet_hidden=1 --n_facet_window=-2 --n_facet_MLP=-1 --efficient_mode='None' --hyper_results="hyper_results/hyper_GRU_tmall_softmax_CPR_Mi"  
#~/anaconda3/bin/python run_hyper.py --model=GRU4Rec_Ours --dataset=yelp2018 --config_files=./recbole/properties/dataset/yelp2018.yaml --params_file=hyper_config/hyper.GRU_mem3_fast   --n_facet=1 --n_facet_context=1 --n_facet_reranker=1 --n_facet_emb=2 --n_facet_all=7 --n_facet_hidden=1 --n_facet_window=-2 --n_facet_MLP=-1 --efficient_mode='None' --hyper_results="hyper_results/hyper_GRU_yelp2018_softmax_CPR_Mi"  
#~/anaconda3/bin/python run_hyper.py --model=GRU4Rec_Ours --dataset=Amazon_Books --config_files=./recbole/properties/dataset/amazon-book.yaml --params_file=hyper_config/hyper.GRU_mem4_fast   --n_facet=1 --n_facet_context=1 --n_facet_reranker=1 --n_facet_emb=2 --n_facet_all=7 --n_facet_hidden=1 --n_facet_window=-2 --n_facet_MLP=-1 --efficient_mode='None' --hyper_results="hyper_results/hyper_GRU_book_softmax_CPR_Mi" 
#~/anaconda3/bin/python run_hyper.py --model=GRU4Rec_Ours --dataset=yoochoose-clicks --config_files=./recbole/properties/dataset/yoochoose.yaml --params_file=hyper_config/hyper.GRU_fast   --n_facet=1 --n_facet_context=1 --n_facet_reranker=1 --n_facet_emb=2 --n_facet_all=7 --n_facet_hidden=1 --n_facet_window=-2 --n_facet_MLP=-1 --efficient_mode='None' --hyper_results="hyper_results/hyper_GRU_yoochoose_softmax_CPR_Mi" 

#~/anaconda3/bin/python run_hyper.py --model=SASRec --dataset=ml-1m --config_files=./recbole/properties/dataset/ml-1m.yaml --params_file=hyper_config/hyper.ours --n_facet=1 --n_facet_all=1 --n_facet_hidden=2 --n_facet_window=-2 --n_facet_MLP=-1 --efficient_mode='None' --hyper_results="hyper_results/hyper_SAS_ml1m_softmax_Mi"  
#~/anaconda3/bin/python run_hyper.py --model=SASRec --dataset=steam --config_files=./recbole/properties/dataset/steam.yaml --params_file=hyper_config/hyper.ours --n_facet=1 --n_facet_all=1 --n_facet_hidden=2 --n_facet_window=-2 --n_facet_MLP=-1 --efficient_mode='None' --hyper_results="hyper_results/hyper_SAS_steam_softmax_Mi"  
#~/anaconda3/bin/python run_hyper.py --model=SASRec --dataset=retailrocket --config_files=./recbole/properties/dataset/retailrocket.yaml --params_file=hyper_config/hyper.ours --n_facet=1 --n_facet_all=1 --n_facet_hidden=2 --n_facet_window=-2 --n_facet_MLP=-1 --efficient_mode='None' --hyper_results="hyper_results/hyper_SAS_retailrocket_softmax_Mi"  
#~/anaconda3/bin/python run_hyper.py --model=SASRec --dataset=amazon-beauty --config_files=./recbole/properties/dataset/amazon-beauty.yaml --params_file=hyper_config/hyper.ours --n_facet=1 --n_facet_all=1 --n_facet_hidden=2 --n_facet_window=-2 --n_facet_MLP=-1 --efficient_mode='None'  --hyper_results="hyper_results/hyper_SAS_beauty_softmax_Mi"  
#~/anaconda3/bin/python run_hyper.py --model=SASRec --dataset=Amazon_Video_Games --config_files=./recbole/properties/dataset/amazon-beauty.yaml --params_file=hyper_config/hyper.ours --n_facet=1 --n_facet_all=1 --n_facet_hidden=2 --n_facet_window=-2 --n_facet_MLP=-1 --efficient_mode='None'  --hyper_results="hyper_results/hyper_SAS_game_softmax_Mi"  
#~/anaconda3/bin/python run_hyper.py --model=SASRec --dataset=diginetica --config_files=./recbole/properties/dataset/diginetica.yaml --params_file=hyper_config/hyper.ours --n_facet=1 --n_facet_all=1 --n_facet_hidden=2 --n_facet_window=-2 --n_facet_MLP=-1 --efficient_mode='None'  --hyper_results="hyper_results/hyper_SAS_diginetica_softmax_Mi"  
#~/anaconda3/bin/python run_hyper.py --model=SASRec --dataset=Twitch-100k --config_files=./recbole/properties/dataset/twitch.yaml --params_file=hyper_config/hyper.ours_mem3_fast --n_facet=1 --n_facet_all=1 --n_facet_hidden=2 --n_facet_window=-2 --n_facet_MLP=-1 --efficient_mode='None'  --hyper_results="hyper_results/hyper_SAS_twitch_softmax_Mi"  
#~/anaconda3/bin/python run_hyper.py --model=SASRec --dataset=algebra2008_2009 --config_files=./recbole/properties/dataset/algebra.yaml --params_file=hyper_config/hyper.ours_mem3_fast --n_facet=1 --n_facet_all=1 --n_facet_hidden=2 --n_facet_window=-2 --n_facet_MLP=-1 --efficient_mode='None'  --hyper_results="hyper_results/hyper_SAS_algebra_softmax_Mi"  
#~/anaconda3/bin/python run_hyper.py --model=SASRec --dataset=gowalla --config_files=./recbole/properties/dataset/gowalla.yaml --params_file=hyper_config/hyper.ours_mem4_fast --n_facet=1 --n_facet_all=1 --n_facet_hidden=2 --n_facet_window=-2 --n_facet_MLP=-1 --efficient_mode='None'  --hyper_results="hyper_results/hyper_SAS_gowalla_softmax_Mi"  
#~/anaconda3/bin/python run_hyper.py --model=SASRec --dataset=ml-10m --config_files=./recbole/properties/dataset/ml-1m.yaml --params_file=hyper_config/hyper.ours_fast --n_facet=1 --n_facet_all=1 --n_facet_hidden=2 --n_facet_window=-2 --n_facet_MLP=-1 --efficient_mode='None'  --hyper_results="hyper_results/hyper_SAS_ml-10m_softmax_Mi"  
#~/anaconda3/bin/python run_hyper.py --model=SASRec --dataset=tmall-buy --config_files=./recbole/properties/dataset/tmall-buy.yaml --params_file=hyper_config/hyper.ours_mem4_fast --n_facet=1 --n_facet_all=1 --n_facet_hidden=2 --n_facet_window=-2 --n_facet_MLP=-1 --efficient_mode='None'  --hyper_results="hyper_results/hyper_SAS_tmall_softmax_Mi"  
#~/anaconda3/bin/python run_hyper.py --model=SASRec --dataset=yelp2018 --config_files=./recbole/properties/dataset/yelp2018.yaml --params_file=hyper_config/hyper.ours_mem3_fast --n_facet=1 --n_facet_all=1 --n_facet_hidden=2 --n_facet_window=-2 --n_facet_MLP=-1 --efficient_mode='None'  --hyper_results="hyper_results/hyper_SAS_yelp2018_softmax_Mi"  
#~/anaconda3/bin/python run_hyper.py --model=SASRec --dataset=Amazon_Books --config_files=./recbole/properties/dataset/amazon-book.yaml --params_file=hyper_config/hyper.ours_mem4_fast --n_facet=1 --n_facet_all=1 --n_facet_hidden=2 --n_facet_window=-2 --n_facet_MLP=-1 --efficient_mode='None'  --hyper_results="hyper_results/hyper_SAS_book_softmax_Mi" 
#~/anaconda3/bin/python run_hyper.py --model=SASRec --dataset=yoochoose-clicks --config_files=./recbole/properties/dataset/yoochoose.yaml --params_file=hyper_config/hyper.ours_fast --n_facet=1 --n_facet_all=1 --n_facet_hidden=2 --n_facet_window=-2 --n_facet_MLP=-1 --efficient_mode='None'  --hyper_results="hyper_results/hyper_SAS_yoochoose_softmax_Mi" 

#~/anaconda3/bin/python run_hyper.py --model=GRU4Rec_Ours --dataset=ml-1m --config_files=./recbole/properties/dataset/ml-1m.yaml --params_file=hyper_config/hyper.GRU_test --n_facet=1 --n_facet_all=1 --n_facet_hidden=1 --n_facet_window=-2 --n_facet_MLP=-1 --efficient_mode='None' --hyper_results="hyper_results/hyper_GRU_ml1m_softmax_Mi" 
#~/anaconda3/bin/python run_hyper.py --model=GRU4Rec_Ours --dataset=steam --config_files=./recbole/properties/dataset/steam.yaml --params_file=hyper_config/hyper.GRU_test --n_facet=1 --n_facet_all=1 --n_facet_hidden=1 --n_facet_window=-2 --n_facet_MLP=-1 --efficient_mode='None' --hyper_results="hyper_results/hyper_GRU_steam_softmax_Mi" 
#~/anaconda3/bin/python run_hyper.py --model=GRU4Rec_Ours --dataset=retailrocket --config_files=./recbole/properties/dataset/retailrocket.yaml --params_file=hyper_config/hyper.GRU_test --n_facet=1 --n_facet_all=1 --n_facet_hidden=1 --n_facet_window=-2 --n_facet_MLP=-1 --efficient_mode='None' --hyper_results="hyper_results/hyper_GRU_retailrocket_softmax_Mi" 
#~/anaconda3/bin/python run_hyper.py --model=GRU4Rec_Ours --dataset=amazon-beauty --config_files=./recbole/properties/dataset/amazon-beauty.yaml --params_file=hyper_config/hyper.GRU_test --n_facet=1 --n_facet_all=1 --n_facet_hidden=1 --n_facet_window=-2 --n_facet_MLP=-1 --efficient_mode='None' --hyper_results="hyper_results/hyper_GRU_beauty_softmax_Mi" 
#~/anaconda3/bin/python run_hyper.py --model=GRU4Rec_Ours --dataset=Amazon_Video_Games --config_files=./recbole/properties/dataset/amazon-beauty.yaml --params_file=hyper_config/hyper.GRU_test  --n_facet=1 --n_facet_all=1 --n_facet_hidden=1 --n_facet_window=-2 --n_facet_MLP=-1 --efficient_mode='None' --hyper_results="hyper_results/hyper_GRU_game_softmax_Mi"  
#~/anaconda3/bin/python run_hyper.py --model=GRU4Rec_Ours --dataset=diginetica --config_files=./recbole/properties/dataset/diginetica.yaml --params_file=hyper_config/hyper.GRU_test  --n_facet=1 --n_facet_all=1 --n_facet_hidden=1 --n_facet_window=-2 --n_facet_MLP=-1 --efficient_mode='None' --hyper_results="hyper_results/hyper_GRU_diginetica_softmax_Mi"  
#~/anaconda3/bin/python run_hyper.py --model=GRU4Rec_Ours --dataset=Twitch-100k --config_files=./recbole/properties/dataset/twitch.yaml --params_file=hyper_config/hyper.GRU_mem3_fast   --n_facet=1 --n_facet_all=1 --n_facet_hidden=1 --n_facet_window=-2 --n_facet_MLP=-1 --efficient_mode='None' --hyper_results="hyper_results/hyper_GRU_twitch_softmax_Mi"  
#~/anaconda3/bin/python run_hyper.py --model=GRU4Rec_Ours --dataset=algebra2008_2009 --config_files=./recbole/properties/dataset/algebra.yaml --params_file=hyper_config/hyper.GRU_mem3_fast   --n_facet=1 --n_facet_all=1 --n_facet_hidden=1 --n_facet_window=-2 --n_facet_MLP=-1 --efficient_mode='None' --hyper_results="hyper_results/hyper_GRU_algebra_softmax_Mi"  
#~/anaconda3/bin/python run_hyper.py --model=GRU4Rec_Ours --dataset=gowalla --config_files=./recbole/properties/dataset/gowalla.yaml --params_file=hyper_config/hyper.GRU_mem4_fast   --n_facet=1 --n_facet_all=1 --n_facet_hidden=1 --n_facet_window=-2 --n_facet_MLP=-1 --efficient_mode='None' --hyper_results="hyper_results/hyper_GRU_gowalla_softmax_Mi"  
#~/anaconda3/bin/python run_hyper.py --model=GRU4Rec_Ours --dataset=ml-10m --config_files=./recbole/properties/dataset/ml-1m.yaml --params_file=hyper_config/hyper.GRU_fast   --n_facet=1 --n_facet_all=1 --n_facet_hidden=1 --n_facet_window=-2 --n_facet_MLP=-1 --efficient_mode='None' --hyper_results="hyper_results/hyper_GRU_ml-10m_softmax_Mi"  
#~/anaconda3/bin/python run_hyper.py --model=GRU4Rec_Ours --dataset=tmall-buy --config_files=./recbole/properties/dataset/tmall-buy.yaml --params_file=hyper_config/hyper.GRU_mem4_fast   --n_facet=1 --n_facet_all=1 --n_facet_hidden=1 --n_facet_window=-2 --n_facet_MLP=-1 --efficient_mode='None' --hyper_results="hyper_results/hyper_GRU_tmall_softmax_Mi"  
#~/anaconda3/bin/python run_hyper.py --model=GRU4Rec_Ours --dataset=yelp2018 --config_files=./recbole/properties/dataset/yelp2018.yaml --params_file=hyper_config/hyper.GRU_mem3_fast   --n_facet=1 --n_facet_all=1 --n_facet_hidden=1 --n_facet_window=-2 --n_facet_MLP=-1 --efficient_mode='None' --hyper_results="hyper_results/hyper_GRU_yelp2018_softmax_Mi"  
#~/anaconda3/bin/python run_hyper.py --model=GRU4Rec_Ours --dataset=Amazon_Books --config_files=./recbole/properties/dataset/amazon-book.yaml --params_file=hyper_config/hyper.GRU_mem4_fast   --n_facet=1 --n_facet_all=1 --n_facet_hidden=1 --n_facet_window=-2 --n_facet_MLP=-1 --efficient_mode='None' --hyper_results="hyper_results/hyper_GRU_book_softmax_Mi" 
#~/anaconda3/bin/python run_hyper.py --model=GRU4Rec_Ours --dataset=yoochoose-clicks --config_files=./recbole/properties/dataset/yoochoose.yaml --params_file=hyper_config/hyper.GRU_fast   --n_facet=1 --n_facet_all=1 --n_facet_hidden=1 --n_facet_window=-2 --n_facet_MLP=-1 --efficient_mode='None' --hyper_results="hyper_results/hyper_GRU_yoochoose_softmax_Mi" 

#~/anaconda3/bin/python run_hyper.py --model=SASRec --dataset=ml-1m --config_files=./recbole/properties/dataset/ml-1m.yaml --params_file=hyper_config/hyper.ours  --efficient_mode='None' --hyper_results="hyper_results/hyper_SAS_ml1m_softmax"  
#~/anaconda3/bin/python run_hyper.py --model=SASRec --dataset=steam --config_files=./recbole/properties/dataset/steam.yaml --params_file=hyper_config/hyper.ours --efficient_mode='None' --hyper_results="hyper_results/hyper_SAS_steam_softmax"  
#~/anaconda3/bin/python run_hyper.py --model=SASRec --dataset=retailrocket --config_files=./recbole/properties/dataset/retailrocket.yaml --params_file=hyper_config/hyper.ours --efficient_mode='None' --hyper_results="hyper_results/hyper_SAS_retailrocket_softmax"  
#~/anaconda3/bin/python run_hyper.py --model=SASRec --dataset=amazon-beauty --config_files=./recbole/properties/dataset/amazon-beauty.yaml --params_file=hyper_config/hyper.ours --efficient_mode='None'  --hyper_results="hyper_results/hyper_SAS_beauty_softmax"  
#~/anaconda3/bin/python run_hyper.py --model=SASRec --dataset=Amazon_Video_Games --config_files=./recbole/properties/dataset/amazon-beauty.yaml --params_file=hyper_config/hyper.ours --efficient_mode='None'  --hyper_results="hyper_results/hyper_SAS_game_softmax"  
#~/anaconda3/bin/python run_hyper.py --model=SASRec --dataset=diginetica --config_files=./recbole/properties/dataset/diginetica.yaml --params_file=hyper_config/hyper.ours --efficient_mode='None'  --hyper_results="hyper_results/hyper_SAS_diginetica_softmax"  
#~/anaconda3/bin/python run_hyper.py --model=SASRec --dataset=Twitch-100k --config_files=./recbole/properties/dataset/twitch.yaml --params_file=hyper_config/hyper.ours_mem3_fast --efficient_mode='None'  --hyper_results="hyper_results/hyper_SAS_twitch_softmax"  
#~/anaconda3/bin/python run_hyper.py --model=SASRec --dataset=algebra2008_2009 --config_files=./recbole/properties/dataset/algebra.yaml --params_file=hyper_config/hyper.ours_mem3_fast --efficient_mode='None'  --hyper_results="hyper_results/hyper_SAS_algebra_softmax"  
#~/anaconda3/bin/python run_hyper.py --model=SASRec --dataset=gowalla --config_files=./recbole/properties/dataset/gowalla.yaml --params_file=hyper_config/hyper.ours_mem4_fast --efficient_mode='None'  --hyper_results="hyper_results/hyper_SAS_gowalla_softmax"  
#~/anaconda3/bin/python run_hyper.py --model=SASRec --dataset=ml-10m --config_files=./recbole/properties/dataset/ml-1m.yaml --params_file=hyper_config/hyper.ours_fast --efficient_mode='None'  --hyper_results="hyper_results/hyper_SAS_ml-10m_softmax"  
#~/anaconda3/bin/python run_hyper.py --model=SASRec --dataset=tmall-buy --config_files=./recbole/properties/dataset/tmall-buy.yaml --params_file=hyper_config/hyper.ours_mem4_fast --efficient_mode='None'  --hyper_results="hyper_results/hyper_SAS_tmall_softmax"  
#~/anaconda3/bin/python run_hyper.py --model=SASRec --dataset=yelp2018 --config_files=./recbole/properties/dataset/yelp2018.yaml --params_file=hyper_config/hyper.ours_mem3_fast --efficient_mode='None'  --hyper_results="hyper_results/hyper_SAS_yelp2018_softmax"  
#~/anaconda3/bin/python run_hyper.py --model=SASRec --dataset=Amazon_Books --config_files=./recbole/properties/dataset/amazon-book.yaml --params_file=hyper_config/hyper.ours_mem4_fast --efficient_mode='None'  --hyper_results="hyper_results/hyper_SAS_book_softmax" 
#~/anaconda3/bin/python run_hyper.py --model=SASRec --dataset=yoochoose-clicks --config_files=./recbole/properties/dataset/yoochoose.yaml --params_file=hyper_config/hyper.ours_fast --efficient_mode='None'  --hyper_results="hyper_results/hyper_SAS_yoochoose_softmax" 

#~/anaconda3/bin/python run_hyper.py --model=SASRec --dataset=ml-1m --config_files=./recbole/properties/dataset/ml-1m.yaml --params_file=hyper_config/hyper.ours_test --n_facet=1 --n_facet_context=1 --n_facet_reranker=1 --n_facet_emb=2 --n_facet_all=7 --n_facet_hidden=2 --n_facet_window=-2 --n_facet_MLP=-1 --efficient_mode='None' --hyper_results="hyper_results/hyper_softmax_CPR_Mi_test_2"  + context + rerank + local + max_logits


#~/anaconda3/bin/python ./run_recbole.py --model=SASRec --dataset=ml-1m --config_files=./recbole/properties/dataset/ml-1m.yaml --n_facet=1 --n_facet_all=1 --n_facet_hidden=2 --n_facet_window=-2 --n_facet_MLP=-1 --efficient_mode='None' --weight_mode=max_logits  
#~/anaconda3/bin/python ./run_recbole.py --model=SASRec --dataset=ml-1m --config_files=./recbole/properties/dataset/ml-1m.yaml --n_facet=1 --n_facet_context=1 --n_facet_reranker=1 --n_facet_emb=2 --n_facet_all=7 --n_facet_hidden=2 --n_facet_window=-2 --n_facet_MLP=-1 --efficient_mode='None' --weight_mode=max_logits #Multi-softmax + context + rerank + local + max_logits - multi-hidden
#~/anaconda3/bin/python ./run_recbole.py --model=SASRec --dataset=ml-1m --config_files=./recbole/properties/dataset/ml-1m.yaml --n_facet=1 --n_facet_context=1 --n_facet_all=2 --n_facet_hidden=2 --n_facet_window=-2 --n_facet_MLP=-1 --efficient_mode='None' --weight_mode=max_logits  
#~/anaconda3/bin/python ./run_recbole.py --model=SASRec --dataset=steam --config_files=./recbole/properties/dataset/steam.yaml --n_facet=1 --n_facet_all=1 --n_facet_hidden=2 --n_facet_window=-2 --n_facet_MLP=-1 --efficient_mode='None' --weight_mode=max_logits  
#~/anaconda3/bin/python ./run_recbole.py --model=SASRec --dataset=steam --config_files=./recbole/properties/dataset/steam.yaml --n_facet=1 --n_facet_context=1 --n_facet_reranker=1 --n_facet_emb=2 --n_facet_all=7 --n_facet_hidden=2 --n_facet_window=-2 --n_facet_MLP=-1 --efficient_mode='None' --weight_mode=max_logits #Multi-softmax + context + rerank + local + max_logits - multi-hidden
#~/anaconda3/bin/python ./run_recbole.py --model=SASRec --dataset=steam --config_files=./recbole/properties/dataset/steam.yaml --n_facet=1 --n_facet_context=1 --n_facet_all=2 --n_facet_hidden=2 --n_facet_window=-2 --n_facet_MLP=-1 --efficient_mode='None' --weight_mode=max_logits  
#~/anaconda3/bin/python ./run_recbole.py --model=SASRec --dataset=amazon-beauty --config_files=./recbole/properties/dataset/amazon-beauty.yaml --n_facet=1 --n_facet_all=1 --n_facet_hidden=2 --n_facet_window=-2 --n_facet_MLP=-1 --efficient_mode='None' --weight_mode=max_logits  
#~/anaconda3/bin/python ./run_recbole.py --model=SASRec --dataset=amazon-beauty --config_files=./recbole/properties/dataset/amazon-beauty.yaml --n_facet=1 --n_facet_context=1 --n_facet_reranker=1 --n_facet_emb=2 --n_facet_all=7 --n_facet_hidden=2 --n_facet_window=-2 --n_facet_MLP=-1 --efficient_mode='None' --weight_mode=max_logits #Multi-softmax + context + rerank + local + max_logits - multi-hidden
#~/anaconda3/bin/python ./run_recbole.py --model=SASRec --dataset=amazon-beauty --config_files=./recbole/properties/dataset/amazon-beauty.yaml --n_facet=1 --n_facet_context=1 --n_facet_all=2 --n_facet_hidden=2 --n_facet_window=-2 --n_facet_MLP=-1 --efficient_mode='None' --weight_mode=max_logits  
#~/anaconda3/bin/python ./run_recbole.py --model=SASRec --dataset=retailrocket --config_files=./recbole/properties/dataset/retailrocket.yaml --n_facet=1 --n_facet_all=1 --n_facet_hidden=2 --n_facet_window=-2 --n_facet_MLP=-1 --efficient_mode='None' --weight_mode=max_logits  
#~/anaconda3/bin/python ./run_recbole.py --model=SASRec --dataset=retailrocket --config_files=./recbole/properties/dataset/retailrocket.yaml --n_facet=1 --n_facet_context=1 --n_facet_reranker=1 --n_facet_emb=2 --n_facet_all=7 --n_facet_hidden=2 --n_facet_window=-2 --n_facet_MLP=-1 --efficient_mode='None' --weight_mode=max_logits #Multi-softmax + context + rerank + local + max_logits - multi-hidden
#~/anaconda3/bin/python ./run_recbole.py --model=SASRec --dataset=retailrocket --config_files=./recbole/properties/dataset/retailrocket.yaml --n_facet=1 --n_facet_context=1 --n_facet_all=2 --n_facet_hidden=2 --n_facet_window=-2 --n_facet_MLP=-1 --efficient_mode='None' --weight_mode=max_logits  

#~/anaconda3/bin/python ./run_recbole.py --model=GRU4Rec_Ours --dataset=ml-1m --config_files=./recbole/properties/dataset/ml-1m.yaml --train_batch_size=64 --n_facet=1 --n_facet_all=1 --n_facet_hidden=1 --n_facet_window=-2 --n_facet_MLP=-1 --efficient_mode='None' --weight_mode=max_logits  
#~/anaconda3/bin/python ./run_recbole.py --model=GRU4Rec_Ours --dataset=ml-1m --config_files=./recbole/properties/dataset/ml-1m.yaml --train_batch_size=64 --n_facet=1 --n_facet_context=1 --n_facet_reranker=1 --n_facet_emb=2 --n_facet_all=7 --n_facet_hidden=1 --n_facet_window=-2 --n_facet_MLP=-1 --efficient_mode='None' --weight_mode=max_logits  
#~/anaconda3/bin/python ./run_recbole.py --model=GRU4Rec_Ours --dataset=ml-1m --config_files=./recbole/properties/dataset/ml-1m.yaml --train_batch_size=64 --n_facet=1 --n_facet_context=1 --n_facet_all=2 --n_facet_hidden=1 --n_facet_window=-2 --n_facet_MLP=-1 --efficient_mode='None' --weight_mode=max_logits  
#~/anaconda3/bin/python ./run_recbole.py --model=GRU4Rec_Ours --dataset=steam --config_files=./recbole/properties/dataset/steam.yaml --train_batch_size=64 --n_facet=1 --n_facet_all=1 --n_facet_hidden=1 --n_facet_window=-2 --n_facet_MLP=-1 --efficient_mode='None' --weight_mode=max_logits  
#~/anaconda3/bin/python ./run_recbole.py --model=GRU4Rec_Ours --dataset=steam --config_files=./recbole/properties/dataset/steam.yaml --train_batch_size=64 --n_facet=1 --n_facet_context=1 --n_facet_reranker=1 --n_facet_emb=2 --n_facet_all=7 --n_facet_hidden=1 --n_facet_window=-2 --n_facet_MLP=-1 --efficient_mode='None' --weight_mode=max_logits  
#~/anaconda3/bin/python ./run_recbole.py --model=GRU4Rec_Ours --dataset=steam --config_files=./recbole/properties/dataset/steam.yaml --train_batch_size=64 --n_facet=1 --n_facet_context=1 --n_facet_all=2 --n_facet_hidden=1 --n_facet_window=-2 --n_facet_MLP=-1 --efficient_mode='None' --weight_mode=max_logits  
#~/anaconda3/bin/python ./run_recbole.py --model=GRU4Rec_Ours --dataset=amazon-beauty --config_files=./recbole/properties/dataset/amazon-beauty.yaml --train_batch_size=64 --n_facet=1 --n_facet_all=1 --n_facet_hidden=1 --n_facet_window=-2 --n_facet_MLP=-1 --efficient_mode='None' --weight_mode=max_logits  
#~/anaconda3/bin/python ./run_recbole.py --model=GRU4Rec_Ours --dataset=amazon-beauty --config_files=./recbole/properties/dataset/amazon-beauty.yaml --train_batch_size=64 --n_facet=1 --n_facet_context=1 --n_facet_reranker=1 --n_facet_emb=2 --n_facet_all=7 --n_facet_hidden=1 --n_facet_window=-2 --n_facet_MLP=-1 --efficient_mode='None' --weight_mode=max_logits  
#~/anaconda3/bin/python ./run_recbole.py --model=GRU4Rec_Ours --dataset=amazon-beauty --config_files=./recbole/properties/dataset/amazon-beauty.yaml --train_batch_size=64 --n_facet=1 --n_facet_context=1 --n_facet_all=2 --n_facet_hidden=1 --n_facet_window=-2 --n_facet_MLP=-1 --efficient_mode='None' --weight_mode=max_logits  
#~/anaconda3/bin/python ./run_recbole.py --model=GRU4Rec_Ours --dataset=retailrocket --config_files=./recbole/properties/dataset/retailrocket.yaml --train_batch_size=64 --n_facet=1 --n_facet_all=1 --n_facet_hidden=1 --n_facet_window=-2 --n_facet_MLP=-1 --efficient_mode='None' --weight_mode=max_logits  
#~/anaconda3/bin/python ./run_recbole.py --model=GRU4Rec_Ours --dataset=retailrocket --config_files=./recbole/properties/dataset/retailrocket.yaml --train_batch_size=64 --n_facet=1 --n_facet_context=1 --n_facet_reranker=1 --n_facet_emb=2 --n_facet_all=7 --n_facet_hidden=1 --n_facet_window=-2 --n_facet_MLP=-1 --efficient_mode='None' --weight_mode=max_logits  
#~/anaconda3/bin/python ./run_recbole.py --model=GRU4Rec_Ours --dataset=retailrocket --config_files=./recbole/properties/dataset/retailrocket.yaml --train_batch_size=64 --n_facet=1 --n_facet_context=1 --n_facet_all=2 --n_facet_hidden=1 --n_facet_window=-2 --n_facet_MLP=-1 --efficient_mode='None' --weight_mode=max_logits  


#~/anaconda3/bin/python ./run_recbole.py --model=GRU4Rec_Ours --dataset=ml-1m --config_files=./recbole/properties/dataset/ml-1m.yaml --train_batch_size=64 --n_facet=3 --n_facet_all=3 --n_facet_hidden=1 --n_facet_window=0 --n_facet_MLP=0 --efficient_mode='None' --weight_mode=max_logits  
#~/anaconda3/bin/python ./run_recbole.py --model=SASRec --dataset=ml-1m --config_files=./recbole/properties/dataset/ml-1m.yaml --n_facet=3 --n_facet_all=3 --n_facet_hidden=1 --n_facet_window=0 --n_facet_MLP=0 --efficient_mode='None' --weight_mode=max_logits  
#~/anaconda3/bin/python ./run_recbole.py --model=GRU4Rec_Ours --dataset=amazon-beauty --config_files=./recbole/properties/dataset/amazon-beauty.yaml --train_batch_size=64 --n_facet=3  --n_facet_all=3 --n_facet_hidden=1 --n_facet_window=0 --n_facet_MLP=0 --efficient_mode='None' --weight_mode=max_logits  
#~/anaconda3/bin/python ./run_recbole.py --model=GRU4Rec_Ours --dataset=steam --config_files=./recbole/properties/dataset/steam.yaml --train_batch_size=64 --n_facet=3 --n_facet_all=3 --n_facet_hidden=1 --n_facet_window=0 --n_facet_MLP=0 --efficient_mode='None' --weight_mode=max_logits  
#~/anaconda3/bin/python ./run_recbole.py --model=GRU4Rec_Ours --dataset=retailrocket --config_files=./recbole/properties/dataset/retailrocket.yaml --train_batch_size=64 --n_facet=3  --n_facet_all=3 --n_facet_hidden=1 --n_facet_window=0 --n_facet_MLP=0 --efficient_mode='None' --weight_mode=max_logits  
#~/anaconda3/bin/python ./run_recbole.py --model=SASRec --dataset=amazon-beauty --config_files=./recbole/properties/dataset/amazon-beauty.yaml --n_facet=3  --n_facet_all=3 --n_facet_hidden=1 --n_facet_window=0 --n_facet_MLP=0 --efficient_mode='None' --weight_mode=max_logits
#~/anaconda3/bin/python ./run_recbole.py --model=SASRec --dataset=steam --config_files=./recbole/properties/dataset/steam.yaml --n_facet=3 --n_facet_all=3 --n_facet_hidden=1 --n_facet_window=0 --n_facet_MLP=0 --efficient_mode='None' --weight_mode=max_logits
#~/anaconda3/bin/python ./run_recbole.py --model=SASRec --dataset=retailrocket --config_files=./recbole/properties/dataset/retailrocket.yaml --n_facet=3  --n_facet_all=3 --n_facet_hidden=1 --n_facet_window=0 --n_facet_MLP=0 --efficient_mode='None' --weight_mode=max_logits

#~/anaconda3/bin/python ./run_recbole.py --model=GRU4Rec_Ours --dataset=ml-1m --config_files=./recbole/properties/dataset/ml-1m.yaml --train_batch_size=64 --n_facet=3 --n_facet_context=1 --n_facet_reranker=1 --n_facet_emb=2 --n_facet_all=9 --n_facet_hidden=1 --n_facet_window=0 --n_facet_MLP=0 --efficient_mode='None' --weight_mode=max_logits  
#~/anaconda3/bin/python ./run_recbole.py --model=SASRec --dataset=ml-1m --config_files=./recbole/properties/dataset/ml-1m.yaml --n_facet=3 --n_facet_context=1 --n_facet_reranker=1 --n_facet_emb=2 --n_facet_all=9 --n_facet_hidden=1 --n_facet_window=0 --n_facet_MLP=0 --efficient_mode='None' --weight_mode=max_logits #Multi-softmax + context + rerank + local + max_logits - multi-hidden
#~/anaconda3/bin/python ./run_recbole.py --model=GRU4Rec_Ours --dataset=amazon-beauty --config_files=./recbole/properties/dataset/amazon-beauty.yaml --train_batch_size=64 --n_facet=3 --n_facet_context=1 --n_facet_reranker=1 --n_facet_emb=2 --n_facet_all=9 --n_facet_hidden=1 --n_facet_window=0 --n_facet_MLP=0 --efficient_mode='None' --weight_mode=max_logits  
#~/anaconda3/bin/python ./run_recbole.py --model=GRU4Rec_Ours --dataset=steam --config_files=./recbole/properties/dataset/steam.yaml --train_batch_size=64 --n_facet=3 --n_facet_context=1 --n_facet_reranker=1 --n_facet_emb=2 --n_facet_all=9 --n_facet_hidden=1 --n_facet_window=0 --n_facet_MLP=0 --efficient_mode='None' --weight_mode=max_logits  
#~/anaconda3/bin/python ./run_recbole.py --model=GRU4Rec_Ours --dataset=retailrocket --config_files=./recbole/properties/dataset/retailrocket.yaml --train_batch_size=64 --n_facet=3 --n_facet_context=1 --n_facet_reranker=1 --n_facet_emb=2 --n_facet_all=9 --n_facet_hidden=1 --n_facet_window=0 --n_facet_MLP=0 --efficient_mode='None' --weight_mode=max_logits  
#~/anaconda3/bin/python ./run_recbole.py --model=SASRec --dataset=amazon-beauty --config_files=./recbole/properties/dataset/amazon-beauty.yaml --n_facet=3 --n_facet_context=1 --n_facet_reranker=1 --n_facet_emb=2 --n_facet_all=9 --n_facet_hidden=1 --n_facet_window=0 --n_facet_MLP=0 --efficient_mode='None' --weight_mode=max_logits #Multi-softmax + context + rerank + local + max_logits - multi-hidden
#~/anaconda3/bin/python ./run_recbole.py --model=SASRec --dataset=steam --config_files=./recbole/properties/dataset/steam.yaml --n_facet=3 --n_facet_context=1 --n_facet_reranker=1 --n_facet_emb=2 --n_facet_all=9 --n_facet_hidden=1 --n_facet_window=0 --n_facet_MLP=0 --efficient_mode='None' --weight_mode=max_logits #Multi-softmax + context + rerank + local + max_logits - multi-hidden
#~/anaconda3/bin/python ./run_recbole.py --model=SASRec --dataset=retailrocket --config_files=./recbole/properties/dataset/retailrocket.yaml --n_facet=3 --n_facet_context=1 --n_facet_reranker=1 --n_facet_emb=2 --n_facet_all=9 --n_facet_hidden=1 --n_facet_window=0 --n_facet_MLP=0 --efficient_mode='None' --weight_mode=max_logits #Multi-softmax + context + rerank + local + max_logits - multi-hidden


#~/anaconda3/bin/python ./run_recbole.py --model=GRU4Rec_Ours --dataset=ml-1m --config_files=./recbole/properties/dataset/ml-1m.yaml --train_batch_size=64 --n_facet=1 --n_facet_all=1 --n_facet_hidden=1 --n_facet_window=0 --n_facet_MLP=0 --efficient_mode='None' --weight_mode=max_logits  
#~/anaconda3/bin/python ./run_recbole.py --model=SASRec --dataset=ml-1m --config_files=./recbole/properties/dataset/ml-1m.yaml --n_facet=1 --n_facet_all=1 --n_facet_hidden=1 --n_facet_window=0 --n_facet_MLP=0 --efficient_mode='None' --weight_mode=max_logits  
#~/anaconda3/bin/python ./run_recbole.py --model=GRU4Rec_Ours --dataset=amazon-beauty --config_files=./recbole/properties/dataset/amazon-beauty.yaml --train_batch_size=64 --n_facet=1  --n_facet_all=1 --n_facet_hidden=1 --n_facet_window=0 --n_facet_MLP=0 --efficient_mode='None' --weight_mode=max_logits  
#~/anaconda3/bin/python ./run_recbole.py --model=GRU4Rec_Ours --dataset=steam --config_files=./recbole/properties/dataset/steam.yaml --train_batch_size=64 --n_facet=1 --n_facet_all=1 --n_facet_hidden=1 --n_facet_window=0 --n_facet_MLP=0 --efficient_mode='None' --weight_mode=max_logits  
#~/anaconda3/bin/python ./run_recbole.py --model=GRU4Rec_Ours --dataset=retailrocket --config_files=./recbole/properties/dataset/retailrocket.yaml --train_batch_size=64 --n_facet=1  --n_facet_all=1 --n_facet_hidden=1 --n_facet_window=0 --n_facet_MLP=0 --efficient_mode='None' --weight_mode=max_logits  
#~/anaconda3/bin/python ./run_recbole.py --model=SASRec --dataset=amazon-beauty --config_files=./recbole/properties/dataset/amazon-beauty.yaml --n_facet=1  --n_facet_all=1 --n_facet_hidden=1 --n_facet_window=0 --n_facet_MLP=0 --efficient_mode='None' --weight_mode=max_logits
#~/anaconda3/bin/python ./run_recbole.py --model=SASRec --dataset=steam --config_files=./recbole/properties/dataset/steam.yaml --n_facet=1 --n_facet_all=1 --n_facet_hidden=1 --n_facet_window=0 --n_facet_MLP=0 --efficient_mode='None' --weight_mode=max_logits
#~/anaconda3/bin/python ./run_recbole.py --model=SASRec --dataset=retailrocket --config_files=./recbole/properties/dataset/retailrocket.yaml --n_facet=1  --n_facet_all=1 --n_facet_hidden=1 --n_facet_window=0 --n_facet_MLP=0 --efficient_mode='None' --weight_mode=max_logits



#~/anaconda3/bin/python ./run_recbole.py --model=SASRec --dataset=steam --config_files=./recbole/properties/dataset/steam.yaml --n_facet=3 --n_facet_context=1 --n_facet_reranker=1 --n_facet_emb=2 --n_facet_all=9 --n_facet_hidden=1 --n_facet_window=0 --n_facet_MLP=0 --efficient_mode='None' --weight_mode=max_logits --eval_batch_size 64 #Multi-softmax + context + rerank + local + max_logits - multi-hidden
#~/anaconda3/bin/python ./run_recbole.py --model=GRU4Rec --dataset=steam --config_files=./recbole/properties/dataset/steam.yaml  
#~/anaconda3/bin/python ./run_recbole.py --model=GRU4Rec --dataset=amazon-beauty --config_files=./recbole/properties/dataset/amazon-beauty.yaml  
#~/anaconda3/bin/python ./run_recbole.py --model=GRU4Rec --dataset=ml-1m --config_files=./recbole/properties/dataset/ml-1m.yaml --train_batch_size=64 --eval_batch_size=64 
#~/anaconda3/bin/python ./run_recbole.py --model=GRU4Rec_Ours --dataset=ml-1m --config_files=./recbole/properties/dataset/ml-1m.yaml --train_batch_size=64 --n_facet=1 --n_facet_context=1 --n_facet_reranker=1 --n_facet_emb=2 --n_facet_all=7 --n_facet_hidden=1 --n_facet_window=0 --n_facet_MLP=0 --efficient_mode='None' --weight_mode=max_logits  
#~/anaconda3/bin/python ./run_recbole.py --model=GRU4Rec_Ours --dataset=ml-1m --config_files=./recbole/properties/dataset/ml-1m.yaml --train_batch_size=64 --n_facet=1 --n_facet_context=1 --n_facet_all=2 --n_facet_hidden=1 --n_facet_window=0 --n_facet_MLP=0 --efficient_mode='None' --weight_mode=max_logits  
#~/anaconda3/bin/python ./run_recbole.py --model=SASRec --dataset=ml-1m --config_files=./recbole/properties/dataset/ml-1m.yaml --n_facet=1 --n_facet_context=1 --n_facet_reranker=1 --n_facet_emb=2 --n_facet_all=7 --n_facet_hidden=1 --n_facet_window=0 --n_facet_MLP=0 --efficient_mode='None' --weight_mode=max_logits #Multi-softmax + context + rerank + local + max_logits - multi-hidden
#~/anaconda3/bin/python ./run_recbole.py --model=SASRec --dataset=ml-1m --config_files=./recbole/properties/dataset/ml-1m.yaml --n_facet=1 --n_facet_context=1 --n_facet_all=2 --n_facet_hidden=1 --n_facet_window=0 --n_facet_MLP=0 --efficient_mode='None' --weight_mode=max_logits  
#~/anaconda3/bin/python ./run_recbole.py --model=GRU4Rec_Ours --dataset=amazon-beauty --config_files=./recbole/properties/dataset/amazon-beauty.yaml --train_batch_size=64 --n_facet=1 --n_facet_context=1 --n_facet_reranker=1 --n_facet_emb=2 --n_facet_all=7 --n_facet_hidden=1 --n_facet_window=0 --n_facet_MLP=0 --efficient_mode='None' --weight_mode=max_logits  
#~/anaconda3/bin/python ./run_recbole.py --model=GRU4Rec_Ours --dataset=steam --config_files=./recbole/properties/dataset/steam.yaml --train_batch_size=64 --n_facet=1 --n_facet_context=1 --n_facet_reranker=1 --n_facet_emb=2 --n_facet_all=7 --n_facet_hidden=1 --n_facet_window=0 --n_facet_MLP=0 --efficient_mode='None' --weight_mode=max_logits  
#~/anaconda3/bin/python ./run_recbole.py --model=GRU4Rec_Ours --dataset=retailrocket --config_files=./recbole/properties/dataset/retailrocket.yaml --train_batch_size=64 --n_facet=1 --n_facet_context=1 --n_facet_reranker=1 --n_facet_emb=2 --n_facet_all=7 --n_facet_hidden=1 --n_facet_window=0 --n_facet_MLP=0 --efficient_mode='None' --weight_mode=max_logits  
#~/anaconda3/bin/python ./run_recbole.py --model=GRU4Rec_Ours --dataset=amazon-beauty --config_files=./recbole/properties/dataset/amazon-beauty.yaml --train_batch_size=64 --n_facet=1 --n_facet_context=1 --n_facet_all=2 --n_facet_hidden=1 --n_facet_window=0 --n_facet_MLP=0 --efficient_mode='None' --weight_mode=max_logits  
#~/anaconda3/bin/python ./run_recbole.py --model=GRU4Rec_Ours --dataset=steam --config_files=./recbole/properties/dataset/steam.yaml --train_batch_size=64 --n_facet=1 --n_facet_context=1 --n_facet_all=2 --n_facet_hidden=1 --n_facet_window=0 --n_facet_MLP=0 --efficient_mode='None' --weight_mode=max_logits  
#~/anaconda3/bin/python ./run_recbole.py --model=GRU4Rec_Ours --dataset=retailrocket --config_files=./recbole/properties/dataset/retailrocket.yaml --train_batch_size=64 --n_facet=1 --n_facet_context=1 --n_facet_all=2 --n_facet_hidden=1 --n_facet_window=0 --n_facet_MLP=0 --efficient_mode='None' --weight_mode=max_logits  
#~/anaconda3/bin/python ./run_recbole.py --model=SASRec --dataset=amazon-beauty --config_files=./recbole/properties/dataset/amazon-beauty.yaml --n_facet=1 --n_facet_context=1 --n_facet_all=2 --n_facet_hidden=1 --n_facet_window=0 --n_facet_MLP=0 --efficient_mode='None' --weight_mode=max_logits
#~/anaconda3/bin/python ./run_recbole.py --model=SASRec --dataset=steam --config_files=./recbole/properties/dataset/steam.yaml --n_facet=1 --n_facet_context=1 --n_facet_all=2 --n_facet_hidden=1 --n_facet_window=0 --n_facet_MLP=0 --efficient_mode='None' --weight_mode=max_logits
#~/anaconda3/bin/python ./run_recbole.py --model=SASRec --dataset=retailrocket --config_files=./recbole/properties/dataset/retailrocket.yaml --n_facet=1 --n_facet_context=1 --n_facet_all=2 --n_facet_hidden=1 --n_facet_window=0 --n_facet_MLP=0 --efficient_mode='None' --weight_mode=max_logits
#~/anaconda3/bin/python ./run_recbole.py --model=SASRec --dataset=amazon-beauty --config_files=./recbole/properties/dataset/amazon-beauty.yaml --n_facet=1 --n_facet_context=1 --n_facet_reranker=1 --n_facet_emb=2 --n_facet_all=7 --n_facet_hidden=1 --n_facet_window=0 --n_facet_MLP=0 --efficient_mode='None' --weight_mode=max_logits #Multi-softmax + context + rerank + local + max_logits - multi-hidden
#~/anaconda3/bin/python ./run_recbole.py --model=SASRec --dataset=steam --config_files=./recbole/properties/dataset/steam.yaml --n_facet=1 --n_facet_context=1 --n_facet_reranker=1 --n_facet_emb=2 --n_facet_all=7 --n_facet_hidden=1 --n_facet_window=0 --n_facet_MLP=0 --efficient_mode='None' --weight_mode=max_logits #Multi-softmax + context + rerank + local + max_logits - multi-hidden
#~/anaconda3/bin/python ./run_recbole.py --model=SASRec --dataset=retailrocket --config_files=./recbole/properties/dataset/retailrocket.yaml --n_facet=1 --n_facet_context=1 --n_facet_reranker=1 --n_facet_emb=2 --n_facet_all=7 --n_facet_hidden=1 --n_facet_window=0 --n_facet_MLP=0 --efficient_mode='None' --weight_mode=max_logits #Multi-softmax + context + rerank + local + max_logits - multi-hidden




