# -*- coding: utf-8 -*-
# Author: 小狼狗

import torch
x = torch.Tensor([1, 2, 3, 4])
# x = torch.unsqueeze(x, 0)
# print(x)
# print(x.size())
x = torch.unsqueeze(x, 1)
print(x)
print(x.size())
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt

x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)  # x data (tensor), shape=(100, 1)
y = x.pow(2) + 0.2*torch.rand(x.size())                 # noisy y data (tensor), shape=(100, 1)

# 用 Variable 来修饰这些数据 tensor
x, y = torch.autograd.Variable(x), Variable(y)

# 画图
plt.scatter(x.data.numpy(), y.data.numpy())
plt.show()
