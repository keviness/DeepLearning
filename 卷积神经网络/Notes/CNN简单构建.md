# （pytorch实战）使用Pytorch实现小型卷积神经网络网络

关注他

﻿卷积层 卷积神经网络中每层卷积层（Convolutional layer）由若干卷积单元组成，每个卷积单元的参数都是通过反向传播算法最佳化得到的。卷积运算的目的是提取输入的不同特征，第一层卷积层可能只能提取一些低级的特征如边缘、线条和角等层级，更多层的网路能从低级特征中迭代提取更复杂的特征。 ·

**pytorch的卷积层：**

```text
class torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
```

一维卷积层，输入的尺度是(N, C_in,L)，输出尺度（ N,C_out,L_out）的计算方式：

```text
out(N_i, C_{out_j})=bias(C {out_j})+\sum^{C{in}-1}{k=0}weight(C{out_j},k)\bigotimes input(N_i,k)
bigotimes: 表示相关系数计算
stride: 控制相关系数的计算步长
dilation: 用于控制内核点之间的距离，详细描述在[这里](https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md)
groups: 控制输入和输出之间的连接， group=1，输出是所有的输入的卷积；group=2，此时相当于有并排的两个卷积层，每个卷积层计算输入通道的一半，并且产生的输出是输出通道的一半，随后将这两个输出连接起来。
```

参数说明如下：

```text
Parameters：

in_channels(int) – 输入信号的通道
out_channels(int) – 卷积产生的通道
kerner_size(int or tuple) - 卷积核的尺寸
stride(int or tuple, optional) - 卷积步长
padding (int or tuple, optional)- 输入的每一条边补充0的层数
dilation(int or tuple, `optional``) – 卷积核元素之间的间距
groups(int, optional) – 从输入通道到输出通道的阻塞连接数
bias(bool, optional) - 如果bias=True，添加偏置
```

**举例：**

```text
m = nn.Conv1d(16, 33, 3, stride=2)
input = Variable(torch.randn(20, 16, 50))
output = m(input)
print(output.size())
#torch.Size([20, 33, 24])
```

**二维卷积层：**

```text
class torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
```

**例子：**

```text
>>> # With square kernels and equal stride
>>> m = nn.Conv2d(16, 33, 3, stride=2)
>>> # non-square kernels and unequal stride and with padding
>>> m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
>>> # non-square kernels and unequal stride and with padding and dilation
>>> m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1))
>>> input = autograd.Variable(torch.randn(20, 16, 50, 100))
>>> output = m(input)
```

## 池化层

```text
class torch.nn.MaxPool1d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
```

对于输入信号的输入通道，提供1维最大池化（max pooling）操作

```text
参数：

kernel_size(int or tuple) - max pooling的窗口大小
stride(int or tuple, optional) - max pooling的窗口移动的步长。默认值是kernel_size
padding(int or tuple, optional) - 输入的每一条边补充0的层数
dilation(int or tuple, optional) – 一个控制窗口中元素步幅的参数
return_indices - 如果等于True，会返回输出最大值的序号，对于上采样操作会有帮助
ceil_mode - 如果等于True，计算输出信号大小的时候，会使用向上取整，代替默认的向下取整的操作
```

**例子：**

```text
>>> # pool of size=3, stride=2
>>> m = nn.MaxPool1d(3, stride=2)
>>> input = autograd.Variable(torch.randn(20, 16, 50))
>>> output = m(input)
```

---

## 全连接层

```text
class torch.nn.Linear(in_features, out_features, bias=True)
参数：
in_features - 每个输入样本的大小
out_features - 每个输出样本的大小
bias - 若设置为False，这层不会学习偏置。默认值：True
```

---

## 卷积神经网络

卷积神经网络（Convolutional Neural Network,CNN）是一种前馈神经网络，它的人工神经元可以响应一部分覆盖范围内的周围单元，对于大型图像处理有出色表现。

---

## pytorch实现ConvNet(注释详解)

```text
import torch
from torch.autograd import Variable
#torch.autograd提供了类和函数用来对任意标量函数进行求导。
import torch.nn as nn
import torch.nn.functional as F
class MNISTConvNet(nn.Module):
    def __init__(self):
        super(MNISTConvNet, self).__init__()
        '''
这是对继承自父类的属性进行初始化。而且是用父类的初始化方法来初始化继承的属性。
也就是说，子类继承了父类的所有属性和方法，父类属性自然会用父类方法来进行初始化。
        '''
#定义网络结构
        self.conv1 = nn.Conv2d(1, 10, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(10, 20, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
    def forward(self, input):
        x = self.pool1(F.relu(self.conv1(input)))
        x = self.pool2(F.relu(self.conv2(x))).view(320)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

net = MNISTConvNet()
print(net)
input = Variable(torch.randn(1, 1, 28, 28))
out = net(input)
print(out.size())
```

---

## pytorch卷积层与池化层输出的尺寸的计算公式详解

要设计卷积神经网络的结构，必须匹配层与层之间的输入与输出的尺寸，这就需要较好的计算输出尺寸 **我在[这里](https://link.zhihu.com/?target=https%3A//blog.csdn.net/qq_42255269/article/details/108297401)详细讲了如何计算尺寸，请浏览**

---

## torch.nn.functional详解

**· Convolution 函数**

```text
torch.nn.functional.conv1d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1)
```

对几个输入平面组成的输入信号应用1D卷积。

```text
参数： 
-  -input – 输入张量的形状 (minibatch x in_channels x iW)
-  - weight – 过滤器的形状 (out_channels, in_channels, kW) 
- - bias – 可选偏置的形状 (out_channels) 
- - stride – 卷积核的步长，默认为1
```

例子：

```text
>>> filters = autograd.Variable(torch.randn(33, 16, 3))
>>> inputs = autograd.Variable(torch.randn(20, 16, 50))
>>> F.conv1d(inputs, filters)
```

**· Pooling 函数**

```text
torch.nn.functional.avg_pool1d(input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True)
```

对由几个输入平面组成的输入信号进行一维平均池化。

```text
参数： 
- kernel_size – 窗口的大小 
- - stride – 窗口的步长。默认值为kernel_size 
- - padding – 在两边添加隐式零填充 
- - ceil_mode – 当为True时，将使用ceil代替floor来计算输出形状 
- - count_include_pad – 当为True时，这将在平均计算时包括补零
```

例子：

```text
>>> # pool of square window of size=3, stride=2
>>> input = Variable(torch.Tensor([[[1,2,3,4,5,6,7]]]))
>>> F.avg_pool1d(input, kernel_size=3, stride=2)
Variable containing:
(0 ,.,.) =
  2  4  6
[torch.FloatTensor of size 1x1x3]
```

**· 非线性激活函数**

```text
torch.nn.functional.relu(input, inplace=False)
```

**· Normalization 函数**

```text
torch.nn.functional.batch_norm(input, running_mean, running_var, weight=None, bias=None, training=False, momentum=0.1, eps=1e-05)
```

**· 线性函数**

```text
torch.nn.functional.linear(input, weight, bias=None)
```

**· Dropout 函数**

```text
torch.nn.functional.dropout(input, p=0.5, training=False, inplace=False)
```

**· 距离函数（Distance functions）**

```text
torch.nn.functional.pairwise_distance(x1, x2, p=2, eps=1e-06)
```

计算向量v1、v2之间的距离

```text
x1:第一个输入的张量
x2:第二个输入的张量
p:矩阵范数的维度。默认值是2，即二范数。
```

例子：

```text
>>> input1 = autograd.Variable(torch.randn(100, 128))
>>> input2 = autograd.Variable(torch.randn(100, 128))
>>> output = F.pairwise_distance(input1, input2, p=2)
>>> output.backward()
```

**· 损失函数（Loss functions）**

```text
torch.nn.functional.nll_loss(input, target, weight=None, size_average=True)
```

负的log likelihood损失函数.

```text
参数：
 - input - (N,C) C 是类别的个数 
 - - target - (N) 其大小是 0 <= targets[i] <= C-1 
 - - weight (Variable, optional) – 一个可手动指定每个类别的权重。如果给定的话，必须是大小为nclasses的Variable 
 - - size_average (bool, optional) – 默认情况下，是mini-batchloss的平均值，然而，如果size_average=False，则是mini-batchloss的总和。
```

发布于 2020-08-29 20:16

[神经网络](https://www.zhihu.com/topic/19607065)

[卷积](https://www.zhihu.com/topic/19678959)

赞同添加评论

分享

喜欢收藏
