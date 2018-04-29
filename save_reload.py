# -*- coding: utf-8 -*-
# Author: 小狼狗
import torch
from torch.autograd import  Variable
import matplotlib.pyplot as plt

torch.manual_seed(1)
x = torch.unsqueeze(torch.linspace(-2,2,200),dim=1)
y = x.pow(2) + 0.2*torch.rand(x.size())

x,y = Variable(x,requires_grad=False),Variable(y,requires_grad=False)

def save():
    net1 = torch.nn.Sequential(
                 torch.nn.Linear(1,10),
                 torch.nn.ReLU(),
                 torch.nn.Linear(10,1)
                 )

    optimizer = torch.optim.SGD(net1.parameters(),lr=0.2)
    loss_func = torch.nn.MSELoss()
    for t in range(100):
        prediction = net1(x)
        loss = loss_func(prediction,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # plot result
    plt.figure(1, figsize=(10, 3))
    plt.subplot(131)
    plt.title('Net1')
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
    #两种保存方式
    torch.save(net1, 'net.pkl')                   #保存整个神经网络
    torch.save(net1.state_dict(), 'net_params.pkl') #只保留参数

def restore_net():
    net2 = torch.load('net.pkl')
    prediction = net2(x)
    plt.subplot(132)
    plt.title('Net2')
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)


def restore_params():
    net3 = torch.nn.Sequential(
        torch.nn.Linear(1,10),
        torch.nn.ReLU(),
        torch.nn.Linear(10,1)
    )
    net3.load_state_dict(torch.load('net_params.pkl'))
    prediction = net3(x)
    plt.subplot(133)
    plt.title('Net3')
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
    plt.show()


if __name__ == '__main__':
    # save net1
    save()

    # restore entire net (may slow)
    restore_net()

    # restore only the net parameters
    restore_params()