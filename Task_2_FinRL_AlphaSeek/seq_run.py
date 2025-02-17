import os
import sys

import numpy as np
import torch as th
from torch.nn.utils import clip_grad_norm_
from typing import Optional
from seq_data import ConfigData, convert_btc_csv_to_btc_npy

TEN = th.Tensor


class SeqData:
    def __init__(self, args: ConfigData, train_ratio: float = 0.8):
        input_ary_path = args.input_ary_path
        label_ary_path = args.label_ary_path

        '''加载或者生成数据'''
        if not os.path.exists(label_ary_path) or not os.path.exists(input_ary_path):
            convert_btc_csv_to_btc_npy(args=args)

        input_seq = np.load(input_ary_path)
        input_seq = np.nan_to_num(input_seq, nan=0.0, neginf=0.0, posinf=0.0)
        input_seq = th.tensor(input_seq, dtype=th.float32)
        assert th.isnan(input_seq).sum() == 0

        label_seq = np.load(label_ary_path)
        # label_seq = np.nan_to_num(label_seq, nan=0.0, neginf=0.0, posinf=0.0)
        label_seq = th.tensor(label_seq, dtype=th.float32)
        assert th.isnan(label_seq).sum() == 0

        seq_len = label_seq.shape[0]
        self.input_dim = input_seq.shape[1]
        self.label_dim = label_seq.shape[1]

        seq_i0 = 0
        seq_i1 = int(seq_len * train_ratio)
        seq_i2 = int(seq_len - 1024) if 0 < train_ratio < 1 else seq_len
        self.train_seq_len = seq_i1 - seq_i0
        self.valid_seq_len = seq_i2 - seq_i1

        self.train_input_seq = input_seq[seq_i0:seq_i1]
        self.valid_input_seq = input_seq[seq_i1:seq_i2]
        self.train_label_seq = label_seq[seq_i0:seq_i1]
        self.valid_label_seq = label_seq[seq_i1:seq_i2]

    def sample_for_train(self, batch_size: int = 32, seq_len: int = 4096, device: th.device = th.device('cpu')):
        i0s = np.random.randint(1024, self.train_seq_len - seq_len, size=batch_size)
        ids = np.arange(seq_len)[None, :].repeat(batch_size, axis=0) + i0s[:, None]

        input_ary = self.train_input_seq[ids, :].permute(1, 0, 2).to(device)
        label_seq = self.train_label_seq[ids, :].permute(1, 0, 2).to(device)
        return input_ary, label_seq


def _update_network(optimizer, obj, clip_grad_norm):
    optimizer.zero_grad()
    obj.backward()
    clip_grad_norm_(parameters=optimizer.param_groups[0]["params"], max_norm=clip_grad_norm)
    optimizer.step()


def train_model(gpu_id: int):
    device = th.device(f"cuda:{gpu_id}" if (th.cuda.is_available() and (gpu_id >= 0)) else "cpu")

    '''Config for training'''
    # 显存占用，模型参数
    batch_size = 256  # 每个训练批次的样本数量
    mid_dim = 128  # 循环网络中隐藏层的维度
    num_layers = 4  # 循环网络中的层数，数值越大，循环网络能记忆的内容越多

    # 训练时长，拟合程度
    epoch = 2 ** 8
    wup_dim = 64  # 循环网络用于预先热身的序列长度。预热阶段不会计算输出的loss，预热只为了得到循环网络隐藏状态
    valid_gap = 128  # 验证数据的间隔，即每隔多少个训练批次进行一次验证，并print出来，建议整个训练过程print 10~1000次

    num_patience = 8  # 模型训练过程中，能容忍loss连续不得到更优的数值的步数。
    weight_decay = 1e-4  # 权重衰减，用于控制正则化项的强度，防止过拟合
    learning_rate = 1e-3  # 学习率，控制每次迭代时参数更新的步长
    clip_grad_norm = 2  # 梯度裁剪的阈值，用于控制梯度的大小，防止梯度爆炸问题

    out_dir = './output'
    if_report = True

    '''数据'''
    args = ConfigData()
    seq_data = SeqData(args=args, train_ratio=0.8)
    input_dim = seq_data.input_dim
    label_dim = seq_data.label_dim

    '''模型'''
    from seq_net import RnnRegNet
    net = RnnRegNet(inp_dim=input_dim, mid_dim=mid_dim, out_dim=label_dim, num_layers=num_layers).to(device)
    optimizer = th.optim.AdamW(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = th.nn.MSELoss(reduction='none')

    '''记录'''
    from seq_record import Evaluator, Validator
    evaluator = Evaluator(out_dir=out_dir)
    validator = Validator(out_dir=out_dir, if_report=if_report)

    seq_len = 2 ** 8
    train_times = int(seq_data.train_seq_len / seq_len / batch_size * epoch)
    print(f"| train_seq_len {seq_data.train_seq_len}  train_times {train_times}")
    for step_idx in range(train_times):
        th.set_grad_enabled(True)
        net.train()
        inp, lab = seq_data.sample_for_train(batch_size=batch_size, seq_len=seq_len, device=device)
        out, _ = net(inp)
        obj = net.get_obj_value(criterion=criterion, out=out, lab=lab, wup_dim=wup_dim)
        _update_network(optimizer, obj.mean(), clip_grad_norm)

        evaluator.update_obj_train(obj=obj)

        if (step_idx % valid_gap == 0) or (step_idx == train_times - 1):
            th.set_grad_enabled(False)
            evaluator.update_obj_train(obj=None)
            validator.reset_list()

            '''update_obj_valid'''
            net.eval()
            for _ in range(int(seq_data.valid_seq_len / seq_len / batch_size)):
                inp, lab = seq_data.sample_for_train(batch_size=batch_size, seq_len=seq_len, device=device)
                out, _ = net(inp)

                seq_len = min(out.shape[0], lab.shape[0])
                out = out[wup_dim:seq_len, :, :]
                lab = lab[wup_dim:seq_len, :, :]
                obj = criterion(out, lab)
                validator.record_accuracy_tpr_fpr(out=out[wup_dim:, :, :],
                                                  lab=lab[wup_dim:, :, :])
                evaluator.update_obj_valid(obj=obj)
            del inp, lab, out

            evaluator.update_obj_valid(obj=None)

            evaluator.log_print(step_idx=step_idx)
            evaluator.draw_train_valid_loss_curve(gpu_id=gpu_id)
            validator.draw_roc_curve_and_accuracy_curve(gpu_id=gpu_id, step_idx=0)

            if evaluator.patience > num_patience:
                break
            if evaluator.patience == 0:
                best_valid_loss = evaluator.best_valid_loss
                th.save(net.state_dict(), f'{out_dir}/net_{step_idx:06}_{best_valid_loss:06.3f}.pth')
                validator.validate_save(f'{out_dir}_result.csv')

    predict_net_path = args.predict_net_path
    th.save(net.state_dict(), predict_net_path)
    print(f'| save network in {predict_net_path}')

    predict_ary = np.empty_like(seq_data.valid_label_seq)
    hid: Optional[TEN] = None

    print(f"| valid_seq_len {seq_data.valid_seq_len}  valid_times {seq_data.valid_seq_len // seq_len}")
    for seq_i0 in range(0, seq_data.valid_seq_len, seq_len):
        seq_i1 = seq_i0 + seq_len
        inp = seq_data.valid_input_seq[seq_i0:seq_i1].to(device)
        out, hid = net.forward(inp[:, None, :], hid)
        predict_ary[seq_i0:seq_i1] = out.data.cpu().numpy().squeeze(1)
    predict_ary_path = args.predict_ary_path
    np.save(predict_ary_path, predict_ary)
    print(f'| save predict in {predict_ary_path}')


def valid_model(gpu_id: int):
    device = th.device(f"cuda:{gpu_id}" if (th.cuda.is_available() and (gpu_id >= 0)) else "cpu")
    th.set_grad_enabled(False)

    '''Config for training'''
    # 显存占用，模型参数
    mid_dim = 128  # 循环网络中隐藏层的维度
    num_layers = 4  # 循环网络中的层数，数值越大，循环网络能记忆的内容越多

    '''数据'''
    args = ConfigData()
    seq_data = SeqData(args=args, train_ratio=0.0)
    input_dim = seq_data.input_dim
    label_dim = seq_data.label_dim

    predict_net_path = args.predict_net_path
    predict_ary_path = args.predict_ary_path

    '''模型'''
    from seq_net import RnnRegNet
    net = RnnRegNet(inp_dim=input_dim, mid_dim=mid_dim, out_dim=label_dim, num_layers=num_layers).to(device)
    net.load_state_dict(th.load(predict_net_path, map_location=lambda storage, loc: storage))

    predict_ary = np.empty_like(seq_data.valid_label_seq)
    hid: Optional[TEN] = None

    seq_len = 2 ** 9
    print(f"| valid_seq_len {seq_data.valid_seq_len}  valid_times {seq_data.valid_seq_len // seq_len}")
    for seq_i0 in range(0, seq_data.valid_seq_len, seq_len):
        seq_i1 = seq_i0 + seq_len
        inp = seq_data.valid_input_seq[seq_i0:seq_i1].to(device)
        out, hid = net.forward(inp[:, None, :], hid)
        predict_ary[seq_i0:seq_i1] = out.data.cpu().numpy().squeeze(1)
    np.save(predict_ary_path, predict_ary)
    print(f'| save predict in {predict_ary_path}')
    """
self.factor_ary.shape
Out[2]: (1029828, 8)
self.price_ary.shape
Out[3]: (1030728, 3)
    """


if __name__ == '__main__':
    GPU_ID = int(sys.argv[1]) if len(sys.argv) > 1 else -1  # 从命令行参数里获得GPU_ID
    convert_btc_csv_to_btc_npy()  # 数据预处理，用行情信息以及代码生成弱因子Alpha101
    train_model(gpu_id=GPU_ID)  # Using weak factor Alpha101 to train recurrent network RNN ​​(LSTM+GRU + Regression)
    valid_model(gpu_id=GPU_ID)  # Generate prediction results using the trained recurrent network and save them to the directory specified by ConfigData
