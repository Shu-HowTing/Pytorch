# -*- coding: utf-8 -*-
# Author: 小狼狗

'''
比较采用不同的梯度优化策略，loss的下降曲线
'''
import torch
import torch.utils.data as Data
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt

LR = 0.01
BATCH_SIZE = 32
EPOCH = 12

#数据
x = torch.unsqueeze(torch.linspace(-1,1,1000), dim=1)
y = x.pow(2) + 0.1*torch.normal(torch.zeros(*x.size()))

# plt.scatter(x.numpy(), y.numpy())
# plt.show()
# put dateset into torch dataset
torch_dataset = Data.TensorDataset(x, y)
print(type(torch_dataset))     #<class 'torch.utils.data.dataset.TensorDataset'>
loader = Data.DataLoader(
                    dataset = torch_dataset,
                    batch_size = BATCH_SIZE,
                    shuffle = True,
                    num_workers = 2)
class Net(torch.nn.Module):
    def __init__(self,n_input,n_hidden,n_out):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_input,n_hidden)
        self.out = torch.nn.Linear(n_hidden,n_out)

    def forward(self, x):
        h = F.relu(self.hidden(x))  # activation function for hidden layer
        out = self.out(h)
        return out

if __name__ == '__main__':
    #每个优化器构造一个网络
    net_SGD = Net(1,20,1)
    net_Momentum = Net(1,20,1)
    net_RMSprop = Net(1,20,1)
    net_Adam = Net(1,20,1)

    nets = [net_SGD, net_Momentum, net_RMSprop, net_Adam]

    opt_SGD     = torch.optim.SGD(net_SGD.parameters(),lr=LR)
    opt_Moment  = torch.optim.SGD(net_Momentum.parameters(),lr=LR,momentum=0.8)
    opt_RMSprop = torch.optim.RMSprop(net_RMSprop.parameters(),lr=LR,alpha=0.9)
    opt_Adam    = torch.optim.Adam(net_Adam.parameters(),lr=LR,betas=(0.9,0.99))
    optimizer = [opt_SGD,opt_Moment,opt_RMSprop,opt_Adam]

    loss_func = torch.nn.MSELoss()
    loss_his = [[],[],[],[]]

    #训练
    for epoch in range(EPOCH):
        print('Epoch:',epoch)
        for step,(batch_x,batch_y) in enumerate(loader):
            b_x = Variable(batch_x)
            b_y = Variable(batch_y)

            for net,opt,l_his in zip(nets,optimizer,loss_his):
                out = net(b_x)
                loss = loss_func(out, b_y)
                opt.zero_grad()
                loss.backward()
                opt.step()
                l_his.append(loss.data[0])
    #画图
    labels = ['SGD', 'Momentum', 'RMSprop', 'Adam']
    for i, l_his in enumerate(loss_his):
        plt.plot(l_his, label=labels[i])
    plt.legend(loc='best')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.ylim((0, 0.2))
    plt.show()
