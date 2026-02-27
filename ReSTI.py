import torch
import torch.nn as nn
from activation_function import Swish,SELU,Mish,HardTanhActivation,CReLU,Tanh_up,Tanh,DyT
import copy

# af = Tanh()
af = HardTanhActivation()
# af = DyT()

from activation_function import Swish,SELU,Mish,HardTanhActivation,CReLU,Tanh_up,Tanh,DyT
import random
from RepODCCB import Rep_1d_block

class Conv_activation_fun(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, kernel_size=3, bias=False, stride=1, padding=1):
        super(Conv_activation_fun, self).__init__()

        self.conv1d = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                stride=stride, padding=padding, padding_mode="zeros", bias=bias)

        # self.nonlinearity = nn.Tanh()
        self.nonlinearity = HardTanhActivation()
    def forward(self, inputs):
        out = self.conv1d(inputs)
        out += inputs

        return self.nonlinearity(out)



class Model(nn.Module):
    def __init__(self, leftlen=96,rightlen=24,pred_len=96,period_len=24,branch_num=1, aside_kernel_size_list=[7],
                 conv_with_activation_function=False, class_conv="normal", conv_with_bias=False,
                 linear_with_bias=False,is_use_bidirectional_flow=True,is_convolution_first=False):
        super(Model, self).__init__()

        # get parameters
        self.seq_len = leftlen + rightlen
        self.pred_len = pred_len
        self.enc_in = 1
        self.period_len = period_len
        self.d_model = 128
        self.model_type = 'linear'
        # self.nonlinearity = nn.ReLU()
        self.is_use_total_mean = True
        self.is_convolution_first = is_convolution_first
        self.is_use_bidirectional_flow = is_use_bidirectional_flow
        self.bn1 = nn.BatchNorm1d(1)
        self.bn2 = nn.BatchNorm1d(self.pred_len // self.period_len)
        self.bias = nn.Parameter(torch.tensor(1.0))
        # self.nonlinearity = HardTanhActivation()
        self.nonlinearity = af

        self.conv_with_activation_function = conv_with_activation_function
        self.class_conv = class_conv
        self.conv_with_bias = conv_with_bias
        self.linear_with_bias = linear_with_bias

        assert self.model_type in ['linear', 'mlp']

        self.seg_num_x = self.seq_len // self.period_len
        self.seg_num_y = self.pred_len // self.period_len

        if self.class_conv == "normal":
            if self.conv_with_bias:
                self.conv1d = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=1 + 2 * (self.period_len // 2),
                                    stride=1, padding=self.period_len // 2, bias=True)
            else:
                self.conv1d = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=1 + 2 * (self.period_len // 2),
                                    stride=1, padding=self.period_len // 2, bias=True)
        elif self.class_conv == "rep_conv":
            if self.conv_with_bias:
                self.conv1d = Rep_1d_block(in_channels=1, out_channels=1,
                    prime_kernel_size=1 + 2 * (self.period_len // 2), stride=1,
                    num_of_aside_branches=branch_num,aside_kernel_size_list=aside_kernel_size_list,bias=True)
            else:
                self.conv1d = Rep_1d_block(in_channels=1, out_channels=1,
                                           prime_kernel_size=1 + 2 * (self.period_len // 2), stride=1,
                                           num_of_aside_branches=branch_num,
                                           aside_kernel_size_list=aside_kernel_size_list, bias=False)

        # self.linear_2_conv1d = nn.Conv1d(in_channels=1, out_channels=self.seg_num_y,
        #                                  kernel_size=self.seg_num_x, dilation=self.period_len, bias=True)

        # self.conv1d = Conv_activation_fun(in_channels=1, out_channels=1, kernel_size=1 + 2 * (self.period_len // 2),
        #                         stride=1, padding=self.period_len // 2, bias=False)
        if self.model_type == 'linear':
            if self.linear_with_bias:
                self.linear = nn.Linear(self.seg_num_x, self.seg_num_y, bias=True)
            else:
                self.linear = nn.Linear(self.seg_num_x, self.seg_num_y, bias=False)
        elif self.model_type == 'mlp':
            self.mlp = nn.Sequential(
                nn.Linear(self.seg_num_x, self.d_model),
                nn.ReLU(),
                nn.Linear(self.d_model, self.seg_num_y)
            )

    def forward(self, x):
        batch_size = x.shape[0]
        x_len = x.shape[1]

        seq_mean = torch.mean(x, dim=1).unsqueeze(1)
        miu = seq_mean.repeat(1, 1, x.shape[1])

        left_mean = torch.mean(x[:,:(x_len//2)], dim=1).unsqueeze(1)
        right_mean = torch.mean(x[:,-(x_len//2):], dim=1).unsqueeze(1)


        x_left = (x[:, :(x_len // 2)] - seq_mean).permute(0, 2, 1)
        x_right = (x[:, -(x_len // 2):] - seq_mean).permute(0, 2, 1)


        if self.is_convolution_first:
            if isinstance(self.conv1d, nn.Conv1d):
                x_left = self.conv1d(x_left.reshape(-1, 1, self.seq_len //2)).reshape(-1, self.enc_in, self.seq_len//2) + x_left
                x_right = self.conv1d(x_right.reshape(-1, 1, self.seq_len//2)).reshape(-1, self.enc_in, self.seq_len//2) + x_right
                x = torch.cat([x_left, x_right], dim=2)
            # elif isinstance(self.conv1d, Conv_activation_fun):
            #     x_left = self.conv1d(x_left.reshape(-1, 1, self.seq_len)).reshape(-1, self.enc_in, self.seq_len//2)
            #     x_right = self.conv1d(x_right.reshape(-1, 1, self.seq_len)).reshape(-1, self.enc_in, self.seq_len//2)
            #     x = torch.cat([x_left, x_right], dim=2)
            elif isinstance(self.conv1d, Rep_1d_block):
                x_left = self.conv1d(x_left.reshape(-1, 1, self.seq_len//2)).reshape(-1, self.enc_in, self.seq_len // 2)
                x_right = self.conv1d(x_right.reshape(-1, 1, self.seq_len//2)).reshape(-1, self.enc_in, self.seq_len // 2)
                x = torch.cat([x_left, x_right], dim=2)
            if self.conv_with_activation_function:
                x = self.nonlinearity(x)

        else:
            x = (x - seq_mean).permute(0, 2, 1)
            if isinstance(self.conv1d, nn.Conv1d):
                x = self.conv1d(x.reshape(-1, 1, self.seq_len)).reshape(-1, self.enc_in, self.seq_len) + x
            # elif isinstance(self.conv1d, Conv_activation_fun):
            #     x = self.conv1d(x.reshape(-1, 1, self.seq_len)).reshape(-1, self.enc_in, self.seq_len)
            elif isinstance(self.conv1d, Rep_1d_block):
                x = self.conv1d(x.reshape(-1, 1, self.seq_len)).reshape(-1, self.enc_in, self.seq_len)
            if self.conv_with_activation_function:
                x = self.nonlinearity(x)


        # downsampling: b,c,s -> bc,n,w -> bc,w,n
        x = x.reshape(-1, self.seg_num_x, self.period_len).permute(0, 2, 1)
        miu = self.conv1d(miu)
        miu = miu.reshape(-1, self.seg_num_x, self.period_len).permute(0, 2, 1)

        # sparse forecasting
        if self.model_type == 'linear':
            y = self.linear(x)  # bc,w,m
        elif self.model_type == 'mlp':
            y = self.mlp(x)

        # upsampling: bc,w,m -> bc,m,w -> b,c,s
        y = y.permute(0, 2, 1).reshape(batch_size, self.enc_in, self.pred_len)


        if not self.is_convolution_first:
            # permute and denorm

            y = y.permute(0, 2, 1) + seq_mean

        else:
            y_left = y[:,:,(self.pred_len//2):].permute(0, 2, 1) + left_mean
            y_right = y[:,:,-(self.pred_len//2):].permute(0, 2, 1) + right_mean
            y = torch.cat([y_left, y_right], dim=1)

        return y
        # return y, -(self.linear(miu) + self.bias)

def insert_zeros(tensor_left, tensor_right, n_zeros, position=5):
    """
    在指定位置插入n个0
    Args:
        tensor: 输入1D张量 (length=10)
        n_zeros: 要插入的0的数量
        position: 插入位置（默认在第5个元素后）
    """

    middle_zeros = torch.zeros(n_zeros, dtype=tensor_left.dtype, device=tensor_left.device)
    left_zeros = torch.zeros(n_zeros, dtype=tensor_left.dtype, device=tensor_left.device)
    right_zeros = torch.zeros(n_zeros, dtype=tensor_left.dtype, device=tensor_left.device)

    return torch.cat([left_zeros, tensor_left, middle_zeros, tensor_right, right_zeros])



def repmtcn_model_convert(model:torch.nn.Module, save_path=None, do_copy=True):
    if do_copy:
        model = copy.deepcopy(model)
    for module in model.modules():
        if hasattr(module, 'switch_to_deploy'):
            module.switch_to_deploy()
    if save_path is not None:
        torch.save(model.state_dict(), save_path)
    return model




def generate_odd_numbers():
    # 随机生成一个1到5之间的整数
    a = random.randint(1, 5)
    # 生成a个范围为1到23之间的奇数
    branch_list = random.sample(range(1, 24, 2), a)
    # 返回a和branch_list
    return a, branch_list









def complex_mse_loss(pred_freq, true_freq):
    # pred_freq和true_freq为复数张量，形状为(batch_size, seq_len)
    real_loss = torch.mean((pred_freq.real - true_freq.real).abs())
    imag_loss = torch.mean((pred_freq.imag - true_freq.imag).abs())
    return real_loss + imag_loss
