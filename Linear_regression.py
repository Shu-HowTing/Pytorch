# -*- coding: utf-8 -*-
# Author: 小狼狗

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

x = torch.unsqueeze(torch.linspace(-2,2,200),dim=1) #unsqueeze将一维变成二维
print(x.size())  #(200,1)
y = x.pow(2) + 0.5*torch.rand(x.size())
x,y = Variable(x),Variable(y)

class Net(torch.nn.Module):
    def __init__(self,n_feature,n_hidden,n_out):
        super(Net,self).__init__()
        #super(LinearRegression, self).__init__()
        self.hidden = torch.nn.Linear(n_feature,n_hidden)  #linear类
        self.predict = torch.nn.Linear(n_hidden,n_out)
    def forward(self, x):
        # self.hidden和self.predict是Linear的实例对象，以下调用__call__方法
        hidden = F.relu(self.hidden(x))
        y_ = self.predict(hidden)
        return y_
model = Net(1,10,1)
#print(net)

optimizer = torch.optim.SGD(model.parameters(), lr=0.2)
criterion = torch.nn.MSELoss()
#可视化
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(x.data.numpy(),y.data.numpy(),c = 'b')
plt.ion() #  打开交互模式
#plt.show()

num_epochs = 1000
for epoch in range(num_epochs):
    # forward
    #通过调用__call__函数调用forward方法
    out = model(x)
    loss = criterion(out, y)
    # backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 20 == 0:
        print('Epoch[{}/{}], loss:  {:.6f}'
              .format(epoch+1, num_epochs, loss.data[0]))
        line=ax.plot(x.data.numpy(), out.data.numpy(), 'r')
        plt.pause(0.1)
        ax.lines.remove(line[0])