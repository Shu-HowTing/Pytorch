# -*- coding: utf-8 -*-
# Author: 小狼狗

'''
numpy 和 torch之间的转换
'''
import  numpy as np
import torch

# 将numpy转换为torch的张量
np_data = np.arange(6).reshape((3,2))
torch_data = torch.from_numpy(np_data)
print(np_data)
print(torch_data)

#将torch转换为numpy
torch2array = torch_data.numpy()
print(torch2array)

#一些简单的运算
array = [-2,3,-4,5,]
tensor = torch.FloatTensor(array)
print(torch.abs(tensor))
print(torch.sin(tensor))

#矩阵的乘法
data = [[1,2],[3,4]]
tensor_data = torch.FloatTensor(data)
np_data = np.array(data)
print(np.dot(data,data))
print(tensor_data.mm(tensor_data)) #等价于torch.mm(tensor_data,tensor_data)
# 7  10
# 15  22
print(tensor_data.dot(tensor_data))
# 30.0









