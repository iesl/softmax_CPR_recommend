# Softmax-CPR <img src="https://github.com/iesl/Softmax-CPR/blob/main/imgs/automated-external-defibrillators-g7991e1588_640.png?raw=true" width="20" height="20"> for Sequential Recommendation

<p align="center"><img src="https://github.com/iesl/softmax_CPR_recommend/blob/master/softmax_limits.png?raw=true" width="651" height="365"></p>

## News

You can use our models (gru4reccpr and sasreccpr) in the latest version of RecBole. To adjust the hyperparameters, please check the comments in the following config files: [GRU4RecCPR](https://github.com/RUCAIBox/RecBole/blob/master/recbole/properties/model/GRU4RecCPR.yaml) and [SASRecCPR](https://github.com/RUCAIBox/RecBole/blob/master/recbole/properties/model/SASRecCPR.yaml).

## Introduction

The similarity structure of the global item embedding in the softmax layer sometimes forces the single hidden state embedding to be close to new items when copying is a better choice, while sometimes forcing the hidden state to be close to the items from the input inappropriately. To alleviate the problem, we adapt the recently-proposed softmax alternatives such as softmax-CPR code from https://github.com/iesl/Softmax-CPR to sequential recommendation tasks and demonstrate that the new softmax architectures unleash the capability of the neural encoder on learning when to copy and when to exclude the items from the input sequence. By only making some simple modifications on the output softmax layer for SASRec and GRU4Rec, softmax-CPR achieves consistent improvement in 12 datasets. With almost the same model size, our best method not only improves the average NDCG@10 of GRU4Rec in 5 datasets with duplicated items by 10% (4%-17% individually) but also improves 7 datasets without duplicated items by 24% (8%-39%)!

<p align="center"><img src="https://github.com/iesl/softmax_CPR_recommend/blob/master/softmax_cpr_method.png?raw=true" width="651" height="538"></p>

## How to Run

### Step 1: Prepare the datasets

Download the datasets from RecBole ([Google Drive](https://drive.google.com/drive/folders/1ahiLmzU7cGRPXf5qGMqtAChte2eYp9gI) or https://github.com/RUCAIBox/RecSysDatasets) and put the data .inter files into the corresponding folders in ./dataset

Our code expect the folder names in ./dataset to be algebra2008_2009, amazon-beauty, Amazon_Books, Amazon_Video_Games, gowalla, ml-10m,  ml-1m, steam, tmall-buy, Twitch-100k, yelp2018, and yoochoose-clicks

Each folder should contain one file called {folder_name}.inter (e.g., amazon-beauty.inter) 

### Step 2: Run the code

If your server supports slurm, you can just run run_hyper_slurm.sh (parallelly). Otherwise, run run_hyper_loop.sh (sequentially)

You can change the variables (model_name) in the script to switch between GRU4Rec, SASRec, and RepeatNet, and change the config (softmax_mode) to switch the softmax alternatives.

### RecoBole Support
If you encounter questions about Recbole, check the readme of recbole: README_RECBOLE_EN.md (English) or README_RECBOLE_CN.md (Chinese)

Notice that our current code does not support the negative sampling during the training or inference time.

## Questions
If you have any questions or find any bugs, please send an email to Haw-Shiuan Chang (ken77921@gmail.com).

## Citation
If you use the code or Softmax-CPR in your work, please cite the following paper (https://arxiv.org/abs/2310.14079):

Haw-Shiuan Chang, Nikhil Agarwal, and Andrew McCallum. 2024. To Copy, or not to Copy; That is a Critical Issue of the Output Softmax Layer in Neural Sequential Recommenders. In Proceedings of The 17th ACM Inernational Conference on Web Search and Data Mining (WSDM 24).

```
@inproceedings{chang2024copy,
  title={To Copy, or not to Copy; That is a Critical Issue of the Output Softmax Layer in Neural Sequential Recommenders},
  abbr={WSDM},
  author={Haw-Shiuan Chang and Nikhil Agarwal and Andrew McCallum},
  booktitle={Proceedings of The 17th ACM Inernational Conference on Web Search and Data Mining},
  year={2024},
}
```

## License
All RecBole data and code are under [MIT License](https://github.com/RUCAIBox/RecBole/blob/master/LICENSE) and can only be used for academic purposes (as indicated in https://github.com/RUCAIBox/RecBole#license on 10/25/2023).
Our modification  (e.g., Softmax-CPR implementation) is under Apache-2.0 license and could be used for commercial purpose.
