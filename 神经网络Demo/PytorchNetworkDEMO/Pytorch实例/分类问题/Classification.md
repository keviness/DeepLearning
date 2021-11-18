# Pytorch搭建简单神经网络（二）:分类问题

Pytorch搭建简单神经网络（一）简单介绍了回归，本文将简单和大家分享一下搭建一个**简单的分类网络**

## **导包**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.autograd import Variable
```

## 生成数据

```python
n_data = torch.ones(100,2)
x0 = torch.normal(2*n_data,1)
y0 = torch.zeros(100)
# print(x0)
x1 = torch.normal(-2*n_data,1)
y1 = torch.ones(100)


x = torch.cat((x0,x1),0).type(torch.FloatTensor)
y = torch.cat((y0,y1)).type(torch.LongTensor)
# print(y)
x,y = Variable(x),Variable(y)
```

![](https://pic2.zhimg.com/80/v2-cc40750a9a878a96a7198af9b53560a9_1440w.jpg)
生成的数据散点图

## 搭建简易神经网络

```python
class Net(torch.nn.Module):
    def __init__(self,n_input,n_hidden,n_output):
        super(Net,self).__init__()
        self.hidden1 = torch.nn.Linear(n_input,n_hidden)
        self.hidden2 = torch.nn.Linear(n_hidden,n_hidden)
        self.predict = torch.nn.Linear(n_hidden,n_output)

    def forward(self, input):
        out = self.hidden1(input)
        out = F.sigmoid(out)
        out = self.hidden2(out)
        out = F.sigmoid(out)
        out = self.predict(out)
        return out
```

![](https://pic2.zhimg.com/80/v2-ea7dd2977908c4684fa6ff37cdb09751_1440w.jpg)
网络结构如图所示

网络有两个隐藏层，中间层为两个全连接层，其中激活函数为sigmod激活函数

```python3
net = Net(2,20,2)
print(net)
```

![](https://pic2.zhimg.com/80/v2-5b496042ec595c2a1ab317e7205bce9d_1440w.jpg)
可以输出网络查看网络的结构

注意此处的网络的输入为2，输出也为2，因为输入的x是一个二维的数据，输出的也是一个二维的数据，输出的结果经过sofrmax函数激活后得到了最后的预测结果。

## 训练回归

```python3
optimizer = torch.optim.SGD(net.parameters(),lr=0.02)
loss_func = torch.nn.CrossEntropyLoss()

for t in range(100):
    prediction = net(x)
    # print(prediction)
    loss = loss_func(prediction,y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

## 结果可视化

```python3
    if t%2==0:
        plt.cla()
        # 过了一道 softmax 的激励函数后的最大概率才是预测值
        print(F.softmax(prediction))
        prediction = torch.max(F.softmax(prediction),1)[1]
        pred_y = prediction.data.numpy().squeeze()
        target_y = y.data.numpy()
        plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=pred_y, s=100, lw=0, cmap='RdYlGn')
        accuracy = sum(pred_y == target_y) / 200.  # 预测中有多少和真实值一样
        plt.text(1.5, -4, 'Accuracy=%.2f' % accuracy, fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.1)
```

## 训练结果

![](https://pic3.zhimg.com/80/v2-a76f30c264875e7f327294705604602a_1440w.jpg)
训练后的分类效果

---

## 附完整代码

```python3
import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt

n_data = torch.ones(100,2)
x0 = torch.normal(2*n_data,1)
y0 = torch.zeros(100)
# print(x0)
x1 = torch.normal(-2*n_data,1)
y1 = torch.ones(100)


x = torch.cat((x0,x1),0).type(torch.FloatTensor)
y = torch.cat((y0,y1)).type(torch.LongTensor)
# print(y)
x,y = Variable(x),Variable(y)

# plt.scatter(x[:,0],x[:,1],c=y.data.numpy(), s=100, lw=0, cmap='RdYlGn')
# plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=y.data.numpy(), s=100, lw=0, cmap='RdYlGn')
# plt.show()

class Net(torch.nn.Module):
    def __init__(self,n_input,n_hidden,n_output):
        super(Net,self).__init__()
        self.hidden1 = torch.nn.Linear(n_input,n_hidden)
        self.hidden2 = torch.nn.Linear(n_hidden,n_hidden)
        self.predict = torch.nn.Linear(n_hidden,n_output)

    def forward(self, input):
        out = self.hidden1(input)
        out = F.sigmoid(out)
        out = self.hidden2(out)
        out = F.sigmoid(out)
        out = self.predict(out)
        # out = F.softmax(out)
        return out

net = Net(2,20,2)
print(net)

optimizer = torch.optim.SGD(net.parameters(),lr=0.02)
loss_func = torch.nn.CrossEntropyLoss()

plt.ion()
plt.show()

for t in range(100):
    out = net(x)
    # print(prediction)
    loss = loss_func(out,y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if t%2==0:
        plt.cla()
        # 过了一道 softmax 的激励函数后的最大概率才是预测值
        # print(F.softmax(prediction))
        prediction = torch.max(F.softmax(out),1)[1]
        pred_y = prediction.data.numpy().squeeze()
        target_y = y.data.numpy()
        plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=pred_y, s=100, lw=0, cmap='RdYlGn')
        accuracy = sum(pred_y == target_y) / 200.  # 预测中有多少和真实值一样
        plt.text(1.5, -4, 'Accuracy=%.2f' % accuracy, fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.1)

plt.ioff()  # 停止画图
plt.show()
```
