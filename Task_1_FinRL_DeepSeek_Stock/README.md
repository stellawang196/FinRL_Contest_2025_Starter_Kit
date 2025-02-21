## FinRL-DeepSeek task starter kit

This task is about developing automated stock trading agents trained on stock prices and financial news data, by combining reinforcement learning and large language models (LLMs).

In this starter kit, participants are introduced to the code of the FinRL-DeepSeek paper (arXiv:2502.07393)

## Installation of dependencies 
run `installation_script.sh` on Ubuntu server (128 GB RAM recommended, CPU sufficient)

## Datasets and data preprocessing 

The basic dataset is FNSPID:
https://huggingface.co/datasets/Zihan1004/FNSPID (the relevant file is `Stock_news/nasdaq_exteral_data.csv`)

https://github.com/Zdong104/FNSPID_Financial_News_Dataset

https://arxiv.org/abs/2402.06698

To be processed by the trading agent, LLM signals (1. Sentiment/stock recommendation and 2. Risk assessment)



