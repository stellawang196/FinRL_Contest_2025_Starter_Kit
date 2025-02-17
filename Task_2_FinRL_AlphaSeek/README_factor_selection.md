
--- 

You can use ChatGPT and other LLMs to translate the following content into English, 
and add prompts to maintain the Markdown structure.

---
# English
## File Structure

### Supervised Training of Deep Learning Recurrent Networks
- `seq_data.py`: Reads BTC's CSV data and generates Alpha101 weak factors.
  - Function `convert_csv_to_level5_csv`: Reads a CSV file into a DataFrame, then extracts the required level5 data, and saves it back as a CSV.
  - Function `convert_btc_csv_to_btc_npy`: Reads a CSV file, saves it as an array, and linearly transforms it to between ±1 using min-max values, saving as an npy. (z-score)

- `seq_net.py`: Feeds a time series into a recurrent network `(LSTM+GRU + MLP)` to predict another time series as the label.
  - Class `RnnRegNet`: Processes `input_seq → Concatenate(LSTM, GRU) → RegressionMLP → label_seq`

- `seq_run.py`: Inputs Alpha101 weak factor series to train the deep learning recurrent network, predicting future price movement labels.
  - Class `SeqData`: Prepares the input and output sequences for training the neural network, using the function `sample_for_train` to randomly cut sequences for training.
  - Function `train_model`: Uses the condition "number of steps without improvement in loss reaches the set limit" as the criterion for early stopping during training.

- `seq_record.py`: Records the training process logs and plots the loss function graph.
  - Class `Evaluator`: Evaluates model performance during training, tracking accuracy and loss values.
  - Class `Validator`: Visualizes evaluation results during training.

### Reinforcement Learning DQN Algorithm Training in a Market Replay Simulator

- `trade_simulator.py`: Contains a market replay simulator for single commodity.
  - Class `TradeSimulator`: A training-focused market replay simulator, complying with the older gym-style API requirements.
  - Class `EvalTradeSimulator`: A market replay simulator for evaluation.

- `erl_config.py`: Configuration file for reinforcement learning training.

- `erl_replay_buffer.py`: Serves as a training dataset for the reinforcement learning market replay simulator.

- `erl_agent.py`: Contains only the DQN class algorithm for reinforcement learning; other unrelated parts have been removed.

- `erl_net.py`: Neural network structures used in the reinforcement learning algorithm.

- `erl_run.py`: Loads the simulator and trains the reinforcement learning agent.

- `erl_evaluator.py`: Evaluates the performance of the reinforcement learning agent.


# Chinese
## 文件结构（就是项目流程）

### 深度学习循环网络的监督学习训练

- `seq_data.py` 读取BTC的csv数据，然后生成 Alpha101弱因子
  - 函数`convert_csv_to_level5_csv` 读取csv文件为df，然后提取出要用的level5数据，重新保存为csv
  - 函数`convert_btc_csv_to_btc_npy` 读取csv文件，然后保存为 array格式，并用min max value 线性变换到 ±1 之间，保存为 npy. (z-score)
- `seq_net.py` 输入时间序列到循环网络`(LSTM+GRU + MLP)` 内，去预测另一个时间序列作为标签
  - 类`RnnRegNet` 由 `input_seq → Concatnate(LSTM, GRU) → RegressionMLP → label_seq`
- `seq_run.py` 输入 Alpha101弱因子序列，去训练深度学习循环网络，去预测未来价格变化趋势标签
  - 类`SeqData` 准备用于训练神经网络的输入与输出序列，用 函数`sample_for_train` 随机截取序列用于训练
  - 函数`train_model` 模型训练过程中，将“loss连续不得到更优的数值的步数达到设置上限” 作为 训练早停(early stopping) 的条件。
- `seq_reocrd.py` 记录训练过程产生的日志信息，画出损失函数图像。
  - 类`Evaluator` 在训练过程中评估模型表现，统计准确率以及loss值
  - 类`Validator` 在训练中根据评估结果绘制评估结果的可视化内容

### 强化学习的DQN算法在行情回放模拟器里训练

- `trade_simulator.py` 存放了（单品种的）行情回放模拟器
  - 类`TradeSimulator` 是用于训练的行情回放模拟器，较快口符合旧版的 gym-style API要求
  - 类`EvalTradeSimulator` 是用于评估的行情回放模拟器
- `erl_config.py` 强化学习训练所用的配置文件
- `erl_replay_buffer.py` 强化学习的行情回放模拟器，相当于训练数据集
- `erl_agent.py` 强化学习算法，里面仅有DQN类的算法，其他与项目无关的已经删掉
- `erl_net.py` 强化学习算法用到的神经网络结构
- `erl_run.py` 加载模拟器，训练强化学习智能体
- `erl_evaluator.py` 评估强化学习智能体表现

### 训练结束后的策略性能评估

`metric.py` 这个就需要 可仪 来补充了。

## 流程

### 1. 生成描述市场的技术因子 Alpha101
运行 `seq_data.py` 的 `convert_btc_csv_to_btc_npy`:

columns:
- col1: AlphaID from Alpha1 to Alpha101
- col2: Used time (second) total
- col3: Used time (second) of single Alpha
- col4: alpha shape
- col5: number of nan `nan_rate= nan_number/alpha.shape[0]`

```markdown
| save in ./data/BTC_1sec_2021-04-07_2021-04-19_label.npy
  1    69   69 (1030728,) 12
  2    69    0 (1030728,) 0
  3    69    0 (1030728,) 0
  4   168   99 (1030728,) 8
  5   168    0 (1030728,) 3
  6   169    0 (1030728,) 0
  7   271  102 (1030728,) 32
  8   271    0 (1030728,) 15
  9   271    0 (1030728,) 6
 10   271    0 (1030728,) 6
 11   271    0 (1030728,) 9
 12   271    0 (1030728,) 6
 13   272    0 (1030728,) 12
 14   272    0 (1030728,) 4
 15   273    0 (1030728,) 2
 16   273    0 (1030728,) 12
 17   471  198 (1030728,) 29
 18   472    0 (1030728,) 12
 19   472    0 (1030728,) 250
 20   472    0 (1030728,) 4
 21   472    0 (1030728,) 0
 22   473    0 (1030728,) 39
 23   473    0 (1030728,) 0
 24   473    0 (1030728,) 8
 25   473    0 (1030728,) 20
 26   672  199 (1030728,) 2
 27   672    1 (1030728,) 86360
 28   673    0 (1030728,) 0
 29   772   99 (1030728,) 25
 30   772    0 (1030728,) 23
 31   812   40 (1030728,) 8
 32   812    0 (1030728,) 365
 33   812    0 (1030728,) 3
 34   813    0 (1030728,) 6
 35  1114  301 (1030728,) 35
 36  1213  100 (1030728,) 41434
 37  1214    0 (1030728,) 220
 38  1313   99 (1030728,) 9
 39  1353   40 (1030728,) 250
 40  1353    0 (1030728,) 171532
 41  1353    0 (1030728,) 0
 42  1353    0 (1030728,) 3
 43  1552  199 (1030728,) 38
 44  1552    0 (1030728,) 0
 45  1553    1 (1030728,) 105247
 46  1553    0 (1030728,) 5
 47  1553    0 (1030728,) 20
 48  1554    0 (1030728,) 20
 49  1554    0 (1030728,) 5
 50  1554    0 (1030728,) 179058
 51  1554    0 (1030728,) 5
 52  1653   99 (1030728,) 240
 53  1654    0 (1030728,) 15
 54  1654    0 (1030728,) 3
 55  1654    1 (1030728,) 0
 56  1654    0 (1030728,) 3
 57  1762  108 (1030728,) 3
 58  1762    0 (1030728,) 6
 59  1831   69 (1030728,) 22
 60  1900   69 (1030728,) 22
 61  1901    0 (1030728,) 0
 62  1901    1 (1030728,) 93362
 63  1902    0 (1030728,) 1482
 64  1902    0 (1030728,) 0
 65  1903    0 (1030728,) 0
 66  2079  176 (1030728,) 13263
 67  2163   84 (1030728,) 0
 68  2247   83 (1030728,) 0
 69  2247    0 (1030728,) 0
 70  2248    0 (1030728,) 0
 71  2722  474 (1030728,) 16682
 72  2999  278 (1030728,) 20100
 73  3177  178 (1030728,) 16
 74  3178    1 (1030728,) 0
 75  3179    1 (1030728,) 0
 76  3180    1 (1030728,) 0
 77  3259   79 (1030728,) 2700
 78  3259    1 (1030728,) 117932
 79  3260    1 (1030728,) 95321
 80  3295   35 (1030728,) 0
 81  3331   35 (1030728,) 0
 82  3401   70 (1030728,) 0
 83  3402    0 (1030728,) 16
 84  3501   99 (1030728,) 23
 85  3700  199 (1030728,) 21016
 86  3770   70 (1030728,) 0
 87  3870  100 (1030728,) 8
 88  4248  378 (1030728,) 0
 89  4248    0 (1030728,) 0
 90  4248    0 (1030728,) 15
 91  4249    0 (1030728,) 39
 92  4513  264 (1030728,) 160171
 93  4806  293 (1030728,) 41592
 94  5099  293 (1030728,) 50653
 95  5198  100 (1030728,) 0
 96  5613  414 (1030728,) 737242
 97  5858  245 (1030728,) 89731
 98  6104  246 (1030728,) 98871
 99  6105    1 (1030728,) 0
100  6105    1 (1030728,) 0
101  6105    0 (1030728,) 3
/home/develop/workspace/factor2policy/seq_data.py:1038: RuntimeWarning: invalid value encountered in true_divide
  arys = 2 * (arys - min_vals) / (max_vals - min_vals) - 1
| save in ./data/BTC_1sec_2021-04-07_2021-04-19_input.npy
```
### 2. Supervised Learning Training Loop Network to Aggregate Multiple Factor Sequences into Fewer Strong Signal Factors

Run `seq_run.py`'s `train_model()`:

The fitting result shows that the loss value on the validation set keeps decreasing, which is highly unusual and may indicate that the sequence input to the prediction model leaks future information.
Next, we should check the Alpha101 factors.

```markdown
$ python3 seq_run.py 7
| train_seq_len 823862  train_times 1609
     0       0 sec  patience 0   | train 1.457e-01  valid 1.298e-01  %0[134 131 128 127 127 128 130 132]
   128      15 sec  patience 0   | train 1.237e-01  valid 1.163e-01  %0[ 85 107 115 121 125 126 125 122]
   256      30 sec  patience 1   | train 1.208e-01  valid 1.312e-01  %0[ 90 120 132 140 140 137 138 148]
   384      45 sec  patience 2   | train 1.203e-01  valid 1.233e-01  %0[ 89 118 125 128 131 129 131 132]
   512      59 sec  patience 0   | train 1.181e-01  valid 1.075e-01  %0[ 82 103 111 114 117 118 110 100]
   640      74 sec  patience 1   | train 1.182e-01  valid 1.217e-01  %0[ 94 121 126 131 128 123 121 125]
   768      89 sec  patience 2   | train 1.155e-01  valid 1.079e-01  %0[ 86 109 112 110 108 108 109 117]
   896     104 sec  patience 0   | train 1.034e-01  valid 9.660e-02  %0[ 90 110 112 103  92  85  82  94]
  1024     119 sec  patience 0   | train 8.936e-02  valid 8.031e-02  %0[ 86 103  98  84  69  59  63  77]
  1152     134 sec  patience 0   | train 7.639e-02  valid 7.035e-02  %0[83 97 89 71 54 44 50 70]
  1280     149 sec  patience 0   | train 6.725e-02  valid 6.238e-02  %0[81 93 80 60 44 37 40 60]
  1408     164 sec  patience 0   | train 5.779e-02  valid 5.132e-02  %0[74 78 62 44 32 28 32 56]
  1536     179 sec  patience 0   | train 4.942e-02  valid 4.552e-02  %0[71 68 51 34 30 29 30 47]
  1608     188 sec  patience 0   | train 4.374e-02  valid 4.049e-02  %0[67 59 41 29 27 27 27 44]
| save network in ./data/BTC_1sec_2021-04-07_2021-04-19_predict.pth
| valid_seq_len 204942  valid_times 400
| save predict in ./data/BTC_1sec_2021-04-07_2021-04-19_predict.npy
| valid_seq_len 1028804  valid_times 1004
| save predict in ./data/BTC_1sec_2021-04-07_2021-04-19_predict.npy
```

### 3. Train Reinforcement Learning Strategy

Run `erl_run.py`'s `train_model()`:


```markdown
$ python3 erl_run.py 7
| Arguments Remove cwd: ./TradeSimulator-v0_D3QN_7
| Evaluator:
| `step`: Number of samples, or total training steps, or running times of `env.step()`.
| `time`: Time spent from the start of training to this moment.
| `avgR`: Average value of cumulative rewards, which is the sum of rewards in an episode.
| `stdR`: Standard dev of cumulative rewards, which is the sum of rewards in an episode.
| `avgS`: Average of steps in an episode.
| `objC`: Objective of Critic network. Or call it loss function of critic network.
| `objA`: Objective of Actor network. It is the average Q value of the critic network.
################################################################################
ID     Step    Time |    avgR   stdR   avgS  stdS |    expR   objC   objA   etc.
;;;                                                                        [219 710  71] [597 370  33]
;;;;;;                                                                                                                [ 49 902  49] [160 838   2]
7  9.48e+03     111 | -381.56  658.5   2370     0 |   -0.11   3.83  -0.02
;;;                                                                        [212 725  62] [572 398  30]
;;;;;;                                                                                                                [  7 985   7] [ 10 968  21]
7  1.90e+04     223 |  154.26  279.2   2370     0 |   -0.11   3.73  -0.01
;;;                                                                        [213 727  59] [538 431  30]
;;;;;;                                                                                                                [  8 984   8] [  7 962  30]
7  2.84e+04     338 |  190.60  308.0   2370     0 |   -0.11   3.67  -0.03
;;;                                                                        [204 738  57] [492 475  32]
;;;;;;                                                                                                                [  2 996   2] [  1 994   4]
7  3.79e+04     449 |   23.88   89.8   2370     0 |   -0.10   3.63  -0.05
;;;                                                                        [182 764  54] [440 527  32]
;;;;;;                                                                                                                [  1 998   1] [  1 997   2]
7  4.74e+04     560 |    5.80   42.7   2370     0 |   -0.07   3.54  -0.07
;;;                                                                        [162 786  52] [390 575  34]
;;;;;;                                                                                                                [  3 994   3] [  3 986  10]
7  5.69e+04     671 |   96.64  222.4   2370     0 |   -0.06   3.40  -0.10
;;;                                                                        [147 801  52] [348 614  37]
;;;;;;                                                                                                                [  3 993   3] [ 10 985   4]
7  6.64e+04     783 |  121.50  286.7   2370     0 |   -0.04   3.23  -0.14
;;;                                                                        [130 821  48] [314 648  37]
;;;;;;                                                                                                                [ 11 977  11] [ 18 926  55]
7  7.58e+04     889 |  459.63  550.6   2370     0 |   -0.03   3.05  -0.17
;;;                                                                        [120 832  48] [290 671  39]
;;;;;;                                                                                                                [ 16 967  16] [ 15 902  83]
7  8.53e+04    1001 |  589.99  670.0   2370     0 |   -0.00   2.88  -0.19
;;;                                                                        [110 842  47] [266 693  41]
;;;;;;                                                                                                                [ 39 921  39] [ 64 749 187]
7  9.48e+04    1114 | 1097.06  850.9   2370     0 |   -0.00   2.70  -0.21
;;;                                                                        [101 854  45] [248 711  40]
;;;;;;                                                                                                                [ 57 885  57] [ 83 648 268]
7  1.04e+05    1227 | 1323.78  972.9   2370     0 |    0.02   2.51  -0.22
;;;                                                                        [ 96 859  44] [236 722  42]
;;;;;;                                                                                                                [ 72 856  72] [134 558 307]
7  1.14e+05    1338 | 1629.63 1122.5   2370     0 |    0.03   2.31  -0.24
;;;                                                                        [ 95 859  45] [229 728  43]
;;;;;;                                                                                                                [ 83 833  83] [137 505 358]
7  1.23e+05    1450 | 1809.65 1201.4   2370     0 |    0.04   2.16  -0.25
;;;                                                                        [ 91 864  44] [220 737  42]
;;;;;;                                                                                                                [101 797 101] [181 418 400]
7  1.33e+05    1564 | 1859.00 1181.5   2370     0 |    0.04   2.03  -0.25
;;;                                                                        [ 88 869  43] [217 742  41]
;;;;;;                                                                                                                [108 783 108] [203 352 444]
7  1.42e+05    1677 | 1942.89 1019.5   2370     0 |    0.05   1.91  -0.25
;;;                                                                        [ 89 866  44] [216 741  42]
;;;;;;                                                                                                                [122 755 122] [235 280 485]
7  1.52e+05    1790 | 2091.68 1153.8   2370     0 |    0.06   1.82  -0.25
;;;                                                                        [ 88 868  44] [214 743  42]
;;;;;;                                                                                                                [131 738 131] [260 255 485]
7  1.61e+05    1907 | 2255.07 1340.2   2370     0 |    0.07   1.73  -0.24
;;;                                                                        [ 90 864  45] [216 741  43]
;;;;;;                                                                                                                [132 735 132] [256 265 478]
7  1.71e+05    2020 | 2221.66 1308.9   2370     0 |    0.08   1.69  -0.23
;;;                                                                        [ 92 862  46] [219 736  44]
;;;;;;                                                                                                                [130 739 130] [239 270 491]
7  1.80e+05    2135 | 2138.70 1295.2   2370     0 |    0.09   1.66  -0.22
;;;                                                                        [ 89 867  44] [216 741  42]
;;;;;;                                                                                                                [124 752 124] [228 297 474]
7  1.90e+05    2247 | 2131.91 1354.6   2370     0 |    0.10   1.65  -0.20
;;;                                                                        [ 91 864  44] [223 734  43]
;;;;;;                                                                                                                [140 720 140] [276 239 485]
7  1.99e+05    2366 | 2344.53 1355.8   2370     0 |    0.12   1.64  -0.17
;;;                                                                        [ 96 857  47] [229 724  47]
;;;;;;                                                                                                                [145 710 145] [250 223 526]
7  2.09e+05    2477 | 2267.71 1359.8   2370     0 |    0.15   1.65  -0.14
;;;                                                                        [ 97 854  49] [234 714  52]
;;;;;;                                                                                                                [143 714 143] [271 228 500]
7  2.18e+05    2591 | 2328.20 1361.2   2370     0 |    0.18   1.68  -0.11
;;;                                                                        [104 841  55] [245 692  62]
;;;;;;                                                                                                                [140 719 140] [274 231 494]
7  2.28e+05    2703 | 2331.56 1323.5   2370     0 |    0.22   1.74  -0.06
;;;                                                                        [109 826  65] [253 671  76]
;;;;;;                                                                                                                [159 682 159] [274 207 519]
7  2.37e+05    2814 | 2293.06 1371.3   2370     0 |    0.27   1.83  -0.00
;;;                                                                        [120 786  94] [258 633 109]
;;;;;;                                                                                                                [146 707 146] [272 226 502]
7  2.46e+05    2927 | 2321.02 1277.3   2370     0 |    0.39   1.96   0.08
;;;                                                                        [133 739 127] [262 598 139]
;;;;;;                                                                                                                [160 680 160] [276 202 522]
7  2.56e+05    3038 | 2309.63 1369.6   2370     0 |    0.47   2.14   0.18
;;;                                                                        [149 678 173] [261 558 180]
;;;;;;                                                                                                                [155 690 155] [254 199 547]
7  2.65e+05    3158 | 2347.24 1428.8   2370     0 |    0.54   2.35   0.30
;;;                                                                        [166 592 241] [256 507 237]
;;;;;;                                                                                                                [166 667 166] [265 192 542]
7  2.75e+05    3269 | 2256.72 1319.2   2370     0 |    0.64   2.60   0.46
;;;                                                                        [179 448 373] [232 422 345]
;;;;;;                                                                                                                [162 676 162] [303 187 510]
7  2.84e+05    3381 | 2463.95 1377.8   2370     0 |    0.74   2.90   0.65
;;;                                                                        [190 287 523] [207 332 460]
;;;;;;                                                                                                                [163 674 163] [294 189 517]
7  2.94e+05    3493 | 2477.48 1445.6   2370     0 |    0.79   3.24   0.87
;;;                                                                        [201 176 623] [188 281 530]
;;;;;;                                                                                                                [164 672 164] [314 182 504]
7  3.03e+05    3604 | 2408.11 1321.6   2370     0 |    0.79   3.59   1.12
;;;                                                                        [209 110 681] [171 256 572]
;;;;;;                                                                                                                [164 671 164] [292 189 519]
7  3.13e+05    3718 | 2375.53 1372.0   2370     0 |    0.75   3.94   1.40
;;;                                                                        [206  76 718] [156 245 599]
;;;;;;                                                                                                                [164 671 164] [309 185 506]
7  3.22e+05    3837 | 2532.90 1469.8   2370     0 |    0.74   4.24   1.69
| UsedTime:    3837 | SavedDir: ./TradeSimulator-v0_D3QN_7
| save valid_position in valid_position.npy
...

```

The action distribution [-1, 0, 1] and position distribution [-1, 0, 1] 
`[219 710  71] [597 370  33]`

![LearningCurve](https://github.com/user-attachments/assets/3ed53bd8-c3ef-42fc-b14f-0dc032fb6bf2)
