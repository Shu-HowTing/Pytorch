# -*- coding: utf-8 -*-
# Author: 小狼狗
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt

#fake.data
x = torch.linspace(-5,5,200)
x = Variable(x)
x_np = x.data.numpy()

#relu
y_relu = F.relu(x).data.numpy()

#sigmoid
y_sig = F.sigmoid(x).data.numpy()

#tanh
y_tanh = F.tanh(x).data.numpy()

#soft_plus
y_soft = F.softplus(x).data.numpy()

plt.figure(1,figsize=(8,6))
plt.subplot(221)
plt.plot(x_np,y_relu,c='r',label='relu')
plt.ylim(-1,5)
plt.legend(loc = 1)

plt.subplot(222)
plt.plot(x_np,y_sig,c='r',label='sigmoid')
plt.ylim(-0,1.1)
plt.legend(loc = 1)

plt.subplot(223)
plt.plot(x_np,y_tanh,c='r',label='tanh')
plt.ylim(-2,2)
plt.legend(loc = 1)

plt.subplot(224)
plt.plot(x_np,y_soft,c='r',label='softplus')
plt.ylim(-1,5)
plt.legend(loc = 1)

plt.show()