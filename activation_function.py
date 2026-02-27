import torch
import torch.nn as nn

# 1. SELU (Scaled Exponential Linear Unit)
class SELU(nn.Module):
    def __init__(self):
        super(SELU, self).__init__()
        self.activation = nn.SELU()

    def forward(self, x):
        return self.activation(x)

# 2. Swish
class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)

# 3. Mish
class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        return x * torch.tanh(torch.nn.functional.softplus(x))



# 4. HardTanh
class HardTanhActivation(nn.Module):
    def __init__(self):
        super(HardTanhActivation, self).__init__()
        self.activation = nn.Hardtanh()

    def forward(self, x):
        return self.activation(x)

# 5. CReLU (Concatenated ReLU)
class CReLU(nn.Module):
    def __init__(self):
        super(CReLU, self).__init__()

    def forward(self, x):
        return torch.cat([torch.relu(x), torch.relu(-x)], dim=1)

class Tanh_up(torch.nn.Module):
    def __init__(self):
        super(Tanh_up, self).__init__()

    def forward(self, x):
        # Tanh implementation
        return (torch.exp(x) - torch.exp(-x)) / (torch.exp(x) + torch.exp(-x)) + 0.2

class Tanh(torch.nn.Module):
    def __init__(self):
        super(Tanh, self).__init__()

    def forward(self, x):
        # Tanh implementation
        return (torch.exp(x) - torch.exp(-x)) / (torch.exp(x) + torch.exp(-x))

# input x has the shape of [B, T, C]
# B: batch size, T: tokens, C: dimension
class DyT(nn.Module):
    def __init__(self, C=192, init_alpha=0.5):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1) * init_alpha)
        self.gama = nn.Parameter(torch.ones(C))
        self.beta = nn.Parameter(torch.zeros(C))
    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        return self.gama * x + self.beta







# # 使用示例
# if __name__ == "__main__":
#     # 示例输入
#     x = torch.linspace(-5, 5, steps=100).unsqueeze(0)
#
#     # 实例化并使用激活函数
#     selu = SELU()
#     swish = Swish()
#     mish = Mish()
#     hardtanh = HardTanhActivation()
#     crelu = CReLU()
#
#     # 前向传播
#     selu_output = selu(x)
#     swish_output = swish(x)
#     mish_output = mish(x)
#     hardtanh_output = hardtanh(x)
#     crelu_output = crelu(x)
#
#     # 打印输出
#     print("SELU Output:", selu_output)
#     print("Swish Output:", swish_output)
#     print("Mish Output:", mish_output)
#     print("HardTanh Output:", hardtanh_output)
#     print("CReLU Output:", crelu_output)
