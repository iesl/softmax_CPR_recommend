## Step 1: Prepare the datasets

Download the datasets from https://drive.google.com/drive/folders/1ahiLmzU7cGRPXf5qGMqtAChte2eYp9gI and put the data .inter files into the corresponding folders in ./dataset

Our code expect the folder names in ./dataset to be algebra2008_2009  amazon-beauty  Amazon_Books  Amazon_Video_Games gowalla  ml-10m  ml-1m  steam  tmall-buy  Twitch-100k  yelp2018  yoochoose-clicks

Each folder should contain one file called {folder_name}.inter (e.g., amazon-beauty.inter) 

## Step 2: Run the code

If your server supports slurm, you can just run run_hyper_slurm.sh (parallelly). Otherwise, run run_hyper_loop.sh (sequentially)

You can change the variables (param_arr, model, and model_name) in the script to switch between GRU4Rec and SASRec, and change the config (model_config and model_config_name) to switch the softmax alternatives

## RecoBole Support
If you encounter questions about Recbole, check the readme of recbole: README_RECBOLE_EN.md

