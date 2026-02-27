import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
from numpy import random
from activation_function import Swish,SELU,Mish,HardTanhActivation,CReLU,Tanh_up

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
# setup_seed(222)

class Conv1xk_bn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, bias=False, stride=1, is_prime_block=False):
        super(Conv1xk_bn, self).__init__()
        self.conv1xk = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, bias=bias, stride=stride)
        self.is_prime_block = is_prime_block  # This parameter is used to assign the BN layer to the main convolution kernel.


    def forward(self, inputs):
        out = self.conv1xk(inputs)

        return out

class Rep_1d_block(nn.Module):
    def __init__(self, in_channels, out_channels, num_of_aside_branches=1, aside_kernel_size_list=[12], prime_kernel_size=3,
                 deploy=False, bias=True, stride=1):
        super(Rep_1d_block, self).__init__()
        assert len(aside_kernel_size_list) == num_of_aside_branches
        self.deploy = deploy
        self.prime_kernel_size = prime_kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.num_of_aside_branches = num_of_aside_branches
        self.aside_kernel_size_list = aside_kernel_size_list
        self.padding_size = []
        self.final_bn = nn.BatchNorm1d(num_features=out_channels)

        self.nonlinearity = HardTanhActivation()
        self.bias = bias


        for kernel_size in self.aside_kernel_size_list:
            padding = (kernel_size - 1) # Standard padding for non-dilated convolution
            self.padding_size.append(padding)

        padding = (self.prime_kernel_size - 1) # Standard padding for non-dilated convolution
        self.padding_size.append(padding)

        names = self.__dict__['_modules']
        if self.deploy:
            self.branch_reparam = nn.Conv1d(in_channels, out_channels, kernel_size=prime_kernel_size, bias=self.bias, stride=stride)
        else:
            for i in range(num_of_aside_branches):
                names['side_branch' + str(i)] = Conv1xk_bn(in_channels, out_channels, kernel_size=aside_kernel_size_list[i], stride=stride, is_prime_block=False)
            self.prime_conv_1xk = Conv1xk_bn(in_channels, out_channels, kernel_size=self.prime_kernel_size, bias=self.bias, stride=stride, is_prime_block=True)

    def forward(self, inputs):
        padding_inputs_list = []
        padding_inputs_list_ = []
        for padding in self.padding_size:
            # padding_inputs_list.append(F.pad(inputs, (padding, padding)))  # Symmetric padding for standard convolution
            padding_inputs_list.append(F.pad(inputs, (padding // 2, padding //2)))  # Symmetric padding for standard convolution
            padding_inputs_list_.append(F.pad(inputs, (padding // 2, padding // 2)).squeeze(0))  # Symmetric padding for standard convolution

        if hasattr(self, 'branch_reparam'):
            return self.branch_reparam(padding_inputs_list[-1])
            # return self.nonlinearity(self.branch_reparam(padding_inputs_list[-1]))
        else:
            out = self.__dict__['_modules']['side_branch' + str(0)](padding_inputs_list[0])
            for i in range(self.num_of_aside_branches - 1):
                new_out = self.__dict__['_modules']['side_branch' + str(i + 1)](padding_inputs_list[i+1])
                out += new_out
            new_out = self.prime_conv_1xk(padding_inputs_list[-1])
            out += new_out
            if self.in_channels == self.out_channels and self.stride == 1:
                out += inputs


            return out


    def get_equivalent_kernel_bias(self):
        if self.bias:
            kernel_weight, kernel_bias = self._fuse()
            return kernel_weight, kernel_bias
        else:
            kernel_weight = self._fuse()
            return kernel_weight

    def _fuse(self):
        kernel_list = []
        bias_list = []
        prime_kernel, prime_bias = self.prime_conv_1xk.conv1xk.weight, self.prime_conv_1xk.conv1xk.bias
        kernel_list.append(prime_kernel)
        bias_list.append(prime_bias)

        # Get weights from side branches
        for i in range(self.num_of_aside_branches):
            kernel = self.__dict__['_modules']['side_branch' + str(i)].conv1xk.weight
            kernel = F.pad(kernel, ((self.prime_kernel_size - self.aside_kernel_size_list[i])//2, (self.prime_kernel_size - self.aside_kernel_size_list[i])//2, 0, 0))  # Adjust kernel size
            kernel_list.append(kernel)

        total_kernel = 0
        for kernel in kernel_list:
            total_kernel += kernel

        if self.in_channels == self.out_channels and self.stride == 1:
            kernel_value = np.zeros((self.in_channels, self.out_channels, 1), dtype=np.float32)
            for i in range(self.in_channels):
                kernel_value[i, i, 0] = 1
            id_kernel = torch.from_numpy(kernel_value)
            id_kernel = F.pad(id_kernel , ((self.prime_kernel_size - 1)//2, (self.prime_kernel_size - 1)//2, 0, 0))
            total_kernel += id_kernel

        if prime_bias == None:
            return total_kernel
        else:
            total_bias = 0
            for bias in bias_list:
                total_bias += bias
            return total_kernel, total_bias


    def _fuse_1xk_bn_tensor(self, branch):
        kernel = branch.conv1xk.weight
        running_mean = branch.bn.running_mean
        running_var = branch.bn.running_var
        gamma = branch.bn.weight
        beta = branch.bn.bias
        eps = branch.bn.eps

        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def switch_to_deploy(self):
        if hasattr(self, 'branch_reparam'):
            return
        if self.bias:
            kernel, bias = self.get_equivalent_kernel_bias()

            self.branch_reparam = nn.Conv1d(in_channels=self.prime_conv_1xk.conv1xk.in_channels,
                                            out_channels=self.prime_conv_1xk.conv1xk.out_channels,
                                            kernel_size=self.prime_conv_1xk.conv1xk.kernel_size,
                                            bias=True,
                                            stride=self.prime_conv_1xk.conv1xk.stride)
            self.branch_reparam.weight.data = kernel
            self.branch_reparam.bias.data = bias

            for para in self.parameters():
                para.detach_()

            for i in range(self.num_of_aside_branches):
                self.__delattr__('side_branch' + str(i))
            self.__delattr__('prime_conv_1xk')
            self.__delattr__('final_bn')
            self.deploy = True

        else:
            kernel = self.get_equivalent_kernel_bias()

            self.branch_reparam = nn.Conv1d(in_channels=self.prime_conv_1xk.conv1xk.in_channels,
                                            out_channels=self.prime_conv_1xk.conv1xk.out_channels,
                                            kernel_size=self.prime_conv_1xk.conv1xk.kernel_size,
                                            bias=self.bias,
                                            stride=self.prime_conv_1xk.conv1xk.stride)
            self.branch_reparam.weight.data = kernel


            for para in self.parameters():
                para.detach_()

            for i in range(self.num_of_aside_branches):
                self.__delattr__('side_branch' + str(i))
            self.__delattr__('prime_conv_1xk')
            self.__delattr__('final_bn')
            self.deploy = True

# #==================以下代码是测试模块转换前后等效性的代码，如果想要测试请将注释取消直接运行即可====================================#


if __name__ == "__main__":
    # 初始化block
    inputs = torch.randn(1, 1, 100)
    block = Rep_1d_block(1, 1, 2, [13,7 ], prime_kernel_size=25)
    
    for module in block.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            nn.init.uniform_(module.running_mean, 0, 0.1)
            nn.init.uniform_(module.running_var, 0, 0.1)
            nn.init.uniform_(module.weight, 0, 0.1)
            nn.init.uniform_(module.bias, 0, 0.1)
    #打印block结构
    print(block)
    
    #验证转换前后模型的输出结果是否相同
    block.eval()
    # outputs1 = branch(inputs, padding_mode='normal')
    outputs1 = block(inputs)
    print(block)
    
    block.switch_to_deploy()
    print(block)
    # outputs2 = branch(inputs,padding_mode='normal')
    outputs2 = block(inputs)
    
    print("转换前后输出的差值：",end=' ')
    print((outputs2 - outputs1).sum().item() ** 2)
    print(outputs1.shape)
    
    print(outputs1.max())
    print(outputs2.max())
    
    print(outputs1.min())
    print(outputs2.min())
    
    print("done!")
