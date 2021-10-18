# Pytorch搭建简单神经网络（一）:回归问题

Pytorch是一个开源的Python机器学习库，基于Torch

神经网络主要分为两种类型，分类和回归，下面就自己学习用Pytorch搭建**简易回归网络**进行分享

## 首先导入需要用的一些包

```python3
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.autograd import Variable
```

## 随机生成一组数据

并加上随机噪声增加数据的复杂性

```text
x = torch.unsqueeze(torch.linspace(-1,1,100),dim=1)
y = x.pow(3)+0.1*torch.randn(x.size())
```

将数据转化成Variable的类型用于输入神经网络

为了更好的看出生成的数据类型，我们采用将生成的数据plot出来

```text
x , y =(Variable(x),Variable(y))

plt.scatter(x.data,y.data)
# 或者采用如下的方式也可以输出x,y
# plt.scatter(x.data.numpy(),y.data.numpy())
plt.show()
```

![](https://pic1.zhimg.com/80/v2-e5fdb71504f19161ce6683d09e2699e8_1440w.jpg)
生成的数据散点图

这里由于x,y都是Variable的类型，需要调用data将其输出出来,直接输出也可以

## 开始搭建神经网络

以下作为搭建网络的模板，定义类，然后继承nn.Module，再继承自己的超类。

```text
class Net(nn.Module):
    def __init__(self):
        super(self).__init__()
        pass
    def forward(self):
        pass
```

不多说直接搭建网络

为了增加网络的复杂性，网络设置为由两个全连接层组成的隐藏层

```text
class Net(nn.Module):
    def __init__(self,n_input,n_hidden,n_output):
        super(Net,self).__init__()
        self.hidden1 = nn.Linear(n_input,n_hidden)
        self.hidden2 = nn.Linear(n_hidden,n_hidden)
        self.predict = nn.Linear(n_hidden,n_output)
    def forward(self,input):
        out = self.hidden1(input)
        out = F.relu(out)
        out = self.hidden2(out)
        out = F.sigmoid(out)
        out =self.predict(out)

        return out
```

为了方便理解，我来画出这个网络的结构

![](https://pic4.zhimg.com/80/v2-10867e40ac432b61d1aee8fff226617f_1440w.jpg)
隐藏层由全连接层构成

```text
net = Net(1,20,1)
print(net)
```

简单的网络就搭建好了，通过调用和print可以输出网络的结构

![](https://pic1.zhimg.com/80/v2-046f90bfb8a761506055cfc5eeac5134_1440w.jpg)
输出网络的结构

## 构建优化目标及损失函数

`torch.optim`是一个实现了各种优化算法的库。大部分常用的方法得到支持，并且接口具备足够的通用性。为了使用 `torch.optim`，你需要构建一个optimizer对象。这个对象能够保持当前参数状态并基于计算得到的梯度进行参数更新。

为了构建一个 `Optimizer`，你需要给它一个包含了需要优化的参数（必须都是 `Variable`对象）的iterable。然后，你可以设置optimizer的参 数选项，比如学习率，权重衰减，等等。^[[1]](https://zhuanlan.zhihu.com/p/114980874#ref_1)^

```text
optimizer = torch.optim.SGD(net.parameters(),lr = 0.1)
loss_func = torch.nn.MSELoss()
```

采用随机梯度下降进行训练，损失函数采用常用的均方损失函数，设置学习率为0.1，可以根据需要进行设置，原则上越小学习越慢，但是精度也越高，然后进行迭代训练（这里设置为5000次）

```text
for t in range(5000):
    prediction = net(x)
    loss = loss_func(prediction,y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

**optimizer.zero_grad()** 意思是把梯度置零，也就是把loss关于weight的导数变成0，即将梯度初始化为零（因为一个batch的loss关于weight的导数是所有sample的loss关于weight的导数的累加和）；**loss.backward() **对loss进行反向传播，  **optimizer.step()** 再对梯度进行优化，更新所有参数。

## 动态显示学习过程

```text
    if t%5 ==0:
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
        plt.text(0.5, 0, 'Loss = %.4f' % loss.data, fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.05)
```

附训练开始及结果图

![](https://pic3.zhimg.com/80/v2-c4fc8500c7c0f1c7cbc3b84209438432_1440w.jpg)
开始训练的图形

![](https://pic4.zhimg.com/80/v2-01dbb3868c819773c4ef56082c1cef93_1440w.jpg)
收敛后的结果（结果不是很好）

我将网络中的一个激活函数从sigmod激活改成relu激活，将学习率改成了0.01后效果改善了很多

![](https://pic3.zhimg.com/80/v2-69508900fb082aeb2df8ef95a6db0d3e_1440w.jpg)
改善后的效果图

---

## 附完整代码

```text
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.autograd import Variable

x = torch.unsqueeze(torch.linspace(-1,1,100),dim=1)
y = x.pow(3)+0.1*torch.randn(x.size())

x , y =(Variable(x),Variable(y))

# plt.scatter(x,y)
# plt.scatter(x.data,y.data)
# plt.scatter(x.data.numpy(),y.data.numpy())
# plt.show()

class Net(nn.Module):
    def __init__(self,n_input,n_hidden,n_output):
        super(Net,self).__init__()
        self.hidden1 = nn.Linear(n_input,n_hidden)
        self.hidden2 = nn.Linear(n_hidden,n_hidden)
        self.predict = nn.Linear(n_hidden,n_output)
    def forward(self,input):
        out = self.hidden1(input)
        out = F.relu(out)
        out = self.hidden2(out)
        out = F.sigmoid(out)
        out =self.predict(out)

        return out

net = Net(1,20,1)
print(net)

optimizer = torch.optim.SGD(net.parameters(),lr = 0.1)
loss_func = torch.nn.MSELoss()

plt.ion()
plt.show()

for t in range(5000):
    prediction = net(x)
    loss = loss_func(prediction,y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if t%5 ==0:
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
        plt.text(0.5, 0, 'Loss = %.4f' % loss.data, fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.05)

plt.ioff()
plt.show()
```
