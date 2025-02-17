import os
import time
import numpy as np
import torch as th
import pandas as pd

TEN = th.Tensor


def import_matplotlib_in_server():
    import matplotlib as mpl
    mpl.use('Agg')
    """Generating matplotlib graphs without a running X server [duplicate]
    write `mpl.use('Agg')` before `import matplotlib.pyplot as plt`
    https://stackoverflow.com/a/4935945/9293137
    """

    import matplotlib.pyplot as plt
    return plt


def skip_method_if_report_disabled(method):
    def wrapper(self, *args, **kwargs):
        if not self.if_report:
            return None
        return method(self, *args, **kwargs)

    return wrapper


class Evaluator:
    def __init__(self, out_dir: str):
        self.out_dir = out_dir

        '''训练日志'''
        self.step_idx = []
        self.step_sec = []

        self.tmp_train = []
        self.obj_train = []

        self.tmp_valid = []
        self.obj_valid = []

        '''计时模块'''
        self.start_time = time.time()

        '''自动停止训练的组件'''
        self.patience = 0
        self.best_valid_loss = th.inf

    def update_obj_train(self, obj=None):
        if obj is None:
            obj_avg = th.stack(self.tmp_train).mean(dim=0)
            self.tmp_train[:] = []
            self.obj_train.append(obj_avg)
        else:
            self.tmp_train.append(obj.mean(dim=(0, 1)).detach().cpu())

    def update_obj_valid(self, obj=None):
        if obj is None:
            obj_avg = th.stack(self.tmp_valid).mean(dim=0)
            self.tmp_valid[:] = []
            self.obj_valid.append(obj_avg)
        else:
            self.tmp_valid.append(obj.mean(dim=(0, 1)).detach().cpu())

    def update_patience_and_best_valid_loss(self):
        valid_loss = self.obj_valid[-1].mean()
        if valid_loss < self.best_valid_loss:
            self.best_valid_loss = valid_loss
            self.patience = 0
        else:
            self.patience += 1

    def log_print(self, step_idx: int):
        self.step_idx.append(step_idx)
        self.step_sec.append(int(time.time()))

        avg_train = self.obj_train[-1].numpy()
        avg_valid = self.obj_valid[-1].numpy()
        time_used = int(time.time() - self.start_time)

        '''update_patience_and_best_valid_loss'''
        valid_loss = self.obj_valid[-1].mean()
        if valid_loss < self.best_valid_loss:
            self.best_valid_loss = valid_loss
            self.patience = 0
        else:
            self.patience += 1

        avg_valid_percent = (avg_valid * 1000).astype(int)
        print(f"{step_idx:>6}  {time_used:>6} sec  patience {self.patience:<4}"
              f"| train {avg_train.mean():9.3e}  valid {avg_valid.mean():9.3e}  %0{avg_valid_percent}")

    def draw_train_valid_loss_curve(self, gpu_id: int = -1, figure_path='', ignore_ratio: float = 0.05):
        figure_path = figure_path if figure_path else f"{self.out_dir}/a_figure_loss_curve_{gpu_id}.jpg"
        step_idx: list = self.step_idx  # write behind `self.log_print` update step_idx
        step_sec: list = self.step_sec  # write behind `self.log_print` update step_sec

        curve_num = len(step_idx)
        if curve_num < 2:
            return

        '''ignore_ratio'''
        ignore_num = int(curve_num * ignore_ratio)
        step_idx = step_idx[ignore_num:]
        step_sec = step_sec[ignore_num:]
        assert len(step_idx) == len(step_sec)

        '''ignore_ratio before mean'''
        obj_train = th.stack(self.obj_train[ignore_num:], dim=0).detach().cpu().numpy()
        avg_train = obj_train.mean(axis=1)
        obj_valid = th.stack(self.obj_valid[ignore_num:], dim=0).detach().cpu().numpy()
        avg_valid = obj_valid.mean(axis=1)

        '''plot subplots'''
        plt = import_matplotlib_in_server()

        fig, axs = plt.subplots(3)
        fig.set_size_inches(12, 20)
        fig.suptitle('Loss Curve', y=0.98)
        alpha = 0.8
        tl_color, tl_style, tl_width = 'black', '-', 3  # train line
        vl_color, vl_style, vl_width = 'black', '-.', 3  # valid line

        xs = step_idx
        ys_train = avg_train
        ys_valid = avg_valid

        # 制表并保存
        res_df = pd.DataFrame(np.array([ys_train, ys_valid]).T, index=xs, columns=['train_avg_loss', 'valid_avg_loss'])
        res_df.to_csv(os.path.join(os.path.dirname(figure_path), 'loss_df.csv'))

        ax0 = axs[0]
        ax0.plot(xs, ys_train, color=tl_color, linestyle=tl_style, label='TrainAvg')
        ax0.plot(xs, ys_valid, color=vl_color, linestyle=vl_style, label='ValidAvg')
        ax0.legend()
        ax0.grid()

        ax1 = axs[1]
        ax1.plot(xs, ys_train, color=tl_color, linestyle=tl_style, linewidth=tl_width, label='TrainAvg')
        for label_i in range(obj_train.shape[1]):
            ax1.plot(xs, obj_train[:, label_i], alpha=alpha, label=f'Lab-{label_i}')
        ax1.legend()
        ax1.grid()

        ax2 = axs[2]
        ax2.plot(xs, ys_valid, color=vl_color, linestyle=vl_style, linewidth=vl_width, label='ValidAvg')
        for label_i in range(obj_valid.shape[1]):
            ax2.plot(xs, obj_valid[:, label_i], alpha=alpha, label=f'Lab-{label_i}')
        ax2.legend()
        ax2.grid()

        plt.subplots_adjust(left=0.05, right=0.98, top=0.98, bottom=0.05, hspace=0.05, wspace=0.05)
        plt.savefig(figure_path, dpi=200)
        plt.close('all')  # avoiding warning about too many open figures, rcParam `figure.max_open_warning`
        # plt.show()  # if use `mpl.use('Agg')` to draw figures without GUI, then plt can't plt.show()

    def close(self, gpu_id: int):
        device = th.device(f"cuda:{gpu_id}" if (th.cuda.is_available() and (gpu_id >= 0)) else "cpu")

        if th.cuda.is_available():
            max_memo = th.cuda.max_memory_allocated(device=device)
            dev_memo = th.cuda.get_device_properties(gpu_id).total_memory
        else:
            max_memo = 0.0
            dev_memo = 0.0

        self.draw_train_valid_loss_curve(gpu_id=gpu_id)
        print(f"GPU(GB)    {max_memo / 2 ** 30:.2f}    "
              f"GPU(ratio) {max_memo / dev_memo:.2f}    "
              f"TimeUsed {int(time.time() - self.start_time)}")


class Validator:  # need to ask Ding1 Yi3Hen2 before open source
    def __init__(self, out_dir: str, if_report: bool):
        self.thresh_ary = th.tensor(np.sort(np.append(np.linspace(0, 1, 64)[1:-1], .25)))
        self.thresh_ary = th.tensor(np.append(np.linspace(0, 1, 64)[1:-1], .25))
        self.idx_25 = th.where(self.thresh_ary.eq(0.25))[0]  # 计算threshold为0.25时的accuracy
        self.accuracy_list = []
        self.tpr_list = []
        self.fpr_list = []
        self.out_list, self.lab_list = [], []

        self.out_dir = out_dir
        self.if_report = if_report

        os.makedirs(out_dir, exist_ok=True)

    @skip_method_if_report_disabled
    def reset_list(self):
        self.accuracy_list = []
        self.tpr_list = []
        self.fpr_list = []
        self.out_list, self.lab_list = [], []

    @skip_method_if_report_disabled
    def record_accuracy_tpr_fpr(self, out, lab):
        if not self.if_report:
            return

        accuracy_list = []
        tpr_list = []
        fpr_list = []
        self.out_list.append(out), self.lab_list.append(lab)
        for thresh in self.thresh_ary:
            lab3 = th.zeros_like(lab)
            lab3[lab.ge(+0.5)] = +1
            lab3[lab.le(-0.5)] = -1

            out3 = th.zeros_like(out)
            out3[out.ge(+thresh)] = +1
            out3[out.le(-thresh)] = -1

            # 计算准确率
            accuracy = lab3.eq(out3).float().mean(dim=(0, 1))
            accuracy_list.append(accuracy.cpu().data.numpy())

            # 计算 confusion matrix 中的相关值
            t_positive = th.sum((lab3.ne(0)) & (out3.ne(0)), dim=(0, 1))
            f_positive = th.sum((lab3.eq(0)) & (out3.ne(0)), dim=(0, 1))
            t_negative = th.sum((lab3.eq(0)) & (out3.eq(0)), dim=(0, 1))
            f_negative = th.sum((lab3.ne(0)) & (out3.eq(0)), dim=(0, 1))

            # 计算 tpr 和 fpr
            tpr = t_positive / (t_positive + f_negative)
            fpr = f_positive / (f_positive + t_negative)

            tpr_list.append(tpr.cpu().data.numpy())
            fpr_list.append(fpr.cpu().data.numpy())

        accuracy_list = np.stack(accuracy_list)
        tpr_list = np.stack(tpr_list)
        fpr_list = np.stack(fpr_list)

        self.accuracy_list.append(accuracy_list)
        self.tpr_list.append(tpr_list)
        self.fpr_list.append(fpr_list)

    def _plot_accuracy_curve(self, accuracy_list):
        plt = import_matplotlib_in_server()

        num_labels = accuracy_list.shape[1]
        for i in range(num_labels):
            accuracy = accuracy_list[:, i]

            line_style = '--' if i < 5 else '-'
            plt.plot(self.thresh_ary, accuracy, linestyle=line_style, label=f'label {i}')

        plt.xlabel('Threshold (Thresh)')
        plt.ylabel('Accuracy')
        plt.title('Accuracy vs. Threshold')
        plt.grid()
        plt.legend()
        plt.grid(True)

    @staticmethod
    def _plot_roc_curve(fpr_list, tpr_list):
        plt = import_matplotlib_in_server()

        num_labels = fpr_list.shape[1]
        for i in range(num_labels):
            fpr = fpr_list[:, i]
            tpr = tpr_list[:, i]

            line_style = '--' if i < 5 else '-'
            plt.plot(fpr, tpr, linestyle=line_style, label=f"label {i}")

        plt.plot([0, 1], [0, 1], color='black', linewidth=2, label='Random')
        plt.xlabel('False Positive Rate (FPR)')
        plt.ylabel('True Positive Rate (TPR)')
        plt.title('ROC Curves')
        plt.grid()
        plt.legend()
        plt.grid(True)

    @skip_method_if_report_disabled
    def draw_roc_curve_and_accuracy_curve(self, gpu_id: int = -1, figure_path='', step_idx=0):
        figure_path = figure_path if figure_path else \
            f"{self.out_dir}/a_figure_ROC_accuracy_curve_{gpu_id}_{step_idx:06}.jpg"

        plt = import_matplotlib_in_server()
        plt.figure(figsize=(6, 12))

        plt.subplot(2, 1, 1)
        accuracy_list = np.stack(self.accuracy_list).mean(axis=0)
        self._plot_accuracy_curve(accuracy_list)

        plt.subplot(2, 1, 2)
        tpr_list = np.stack(self.tpr_list).mean(axis=0)
        fpr_list = np.stack(self.fpr_list).mean(axis=0)
        self._plot_roc_curve(fpr_list, tpr_list)

        plt.tight_layout()  # 调整布局，防止重叠
        plt.savefig(figure_path, dpi=200)
        plt.close('all')  # avoiding warning about too many open figures, rcParam `figure.max_open_warning`
        # plt.show()  # if use `mpl.use('Agg')` to draw figures without GUI, then plt can't plt.show()
        # auc_values = [auc(fpr_list[i], tpr_list[i]) for i in range(fpr_list.shape[1])]
        # auc_df = pd.DataFrame([auc_values], columns=[f'label-{i}' for i in range(fpr_list.shape[1])])
        # auc_df.to_csv(os.path.join(os.path.dirname(figure_path), 'auc.csv'))

    def _validate_calculate(self, outs: TEN, labs: TEN) -> pd.DataFrame:
        res_list = []
        mae_loss = th.nn.L1Loss(reduction='none')
        mse_loss = th.nn.MSELoss(reduction='none')
        accuracy_list = np.stack(self.accuracy_list).mean(axis=0)
        idx_25 = int(self.idx_25)

        mse_list = th.mean(mse_loss(outs[:, :, :], labs[:, :, :]), dim=(0, 1)).cpu().detach().numpy()
        mae_list = th.mean(mae_loss(outs[:, :, :], labs[:, :, :]), dim=(0, 1)).cpu().detach().numpy()

        num_labels = labs.shape[2]
        df0 = pd.DataFrame({'label': [f'label-{i}' for i in range(num_labels)]})
        df0['MSE'] = mse_list
        df0['MAE'] = mae_list
        df0['ACC'] = (accuracy_list[idx_25, :]).tolist()
        res_list.append(df0)

        mse_i = np.mean(mse_list)
        mae_i = np.mean(mae_list)
        res_list.append(pd.DataFrame([['price', mse_i, mae_i, np.mean(accuracy_list[idx_25, 0:5])]],
                                     columns=['label', 'MSE', 'MAE', 'ACC']))

        mse_i = np.mean(mse_list)
        mae_i = np.mean(mae_list)
        res_list.append(pd.DataFrame([['all', mse_i, mae_i, np.mean(accuracy_list[idx_25, :])]],
                                     columns=['label', 'MSE', 'MAE', 'ACC']))

        return pd.concat(res_list, axis=0)

    @skip_method_if_report_disabled
    def validate_save(self, save_path: str):
        if not self.if_report:
            return

        outs, labs = th.cat(self.out_list, dim=1), th.cat(self.lab_list, dim=1)
        res = self._validate_calculate(outs, labs)
        res.to_csv(os.path.join(save_path))
