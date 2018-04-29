# -*- coding: utf-8 -*-
# Author: 小狼狗
'''
批训练
'''
import torch
import torch.utils.data as Data

BATCH_SIZE = 5

x = torch.linspace(1,10,10)  # x data (torch tensor)
y = torch.linspace(10,1,10)

torch_dataset = Data.TensorDataset(data_tensor=x, target_tensor=y)  #训练的数据
loader = Data.DataLoader(
    dataset     = torch_dataset,    #数据
    batch_size  = BATCH_SIZE,       #Batch的大小
    shuffle     = True,             #是否打乱顺序
    num_workers = 2                 #线程
)
if __name__ == '__main__':
    for epoch in range(3):
        for step, (batch_x,batch_y)  in enumerate(loader):
            #training..
            print("Epoch:",epoch,'| Step:',step,'| batch x:',
                batch_x.numpy(),'| batch y:',batch_y.numpy())
