## FinRL-DeepSeek task starter kit

This task is about developing automated stock trading agents trained on stock prices and financial news data, by combining reinforcement learning and large language models (LLMs).

In this starter kit, participants are introduced to the [code](https://github.com/benstaf/FinRL_DeepSeek) of the [FinRL-DeepSeek paper](https://arxiv.org/abs/2502.07393)

## Installation of dependencies 
run `installation_script.sh` on Ubuntu server (128 GB RAM CPU instance recommended)

## Datasets and data preprocessing 

The basic dataset is FNSPID:
https://huggingface.co/datasets/Zihan1004/FNSPID (the relevant file is `Stock_news/nasdaq_exteral_data.csv`)

https://github.com/Zdong104/FNSPID_Financial_News_Dataset

https://arxiv.org/abs/2402.06698

LLM signals are added by running `sentiment_deepseek_deepinfra.py` and `risk_deepseek_deepinfra.py`, to obtain:  
- https://huggingface.co/datasets/benstaf/nasdaq_news_sentiment
- https://huggingface.co/datasets/benstaf/risk_nasdaq

Then this data is processed by `train_trade_data_deepseek_sentiment.py` and `train_trade_data_deepseek_risk.py` to generate agent-ready datasets.  
For plain PPO and CPPO, `train_trade_data.py` is used.

## Training and Environments  
- For training PPO, run:  
  `nohup mpirun --allow-run-as-root -np 8 python train_ppo.py > output_ppo.log 2>&1 &`



- For CPPO: `train_cppo.py`  
- For PPO-DeepSeek: `train_ppo_llm.py`  
- For CPPO-DeepSeek: `train_cppo_llm_risk.py`  

Environment files are:  
- `env_stocktrading.py` for PPO and CPPO, same as in the original FinRL  
- `env_stocktrading_llm.py` or `env_stocktrading_llm_01.py` for PPO-DeepSeek (depending on the desired LLM influence. More tweaking would be interesting)  
- `env_stocktrading_llm_risk.py` or `env_stocktrading_llm_risk_01.py` for CPPO-DeepSeek  

Log files are `output_ppo.log`, etc., and should be monitored during training, especially:  
- `AverageEpRet`  
- `KL`  
- `ClipFrac`  

## Evaluation  
Evaluation in the trading phase (2019-2023) happens in the `FinRL_DeepSeek_backtest.ipynb` Colab notebook.  
Metrics used are `Information Ratio`, `CVaR`, and `Rachev Ratio`, but adding `Outperformance frequency` would be nice.

## Submission  
Please submit GitHub, Hugging Face, and Colab notebook links for evaluation, in addition to your paper.

