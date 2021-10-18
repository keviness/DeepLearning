# 用pytorch搭建一个简单的神经网络

# 一、任务

首先说下我们要搭建的网络要完成的学习任务： 让我们的神经网络学会逻辑异或运算，异或运算也就是俗称的“相同取0，不同取1” 。再把我们的需求说的简单一点，也就是我们需要搭建这样一个神经网络，让我们在输入（1,1）时输出0，输入（1,0）时输出1（相同取0，不同取1），以此类推。

# 二、实现思路

因为我们的需求需要有两个输入，一个输出，所以我们需要在输入层设置两个输入节点，输出层设置一个输出节点。因为问题比较简单，所以隐含层我们只需要设置10个节点就可以达到不错的效果了，隐含层的激活函数我们采用ReLU函数，输出层我们用Sigmoid函数，让输出保持在0到1的一个范围，如果输出大于0.5，即可让输出结果为1，小于0.5，让输出结果为0.

# 三、实现过程

搭建网络时，我们使用简单的快速搭建法，这种方法就像搭积木一样可以让你快速高效地搭建起一个神经网络来，具体实现如下：

## 第一步：引入必要的库

代码如下：

```
import torch
import torch.nn as nn
import numpy as np
```

用pytorch当然要引入torch包，然后为了写代码方便将torch包里的nn用nn来代替，nn这个包就是neural network的缩写，专门用来搭神经网络的一个包。引入numpy是为了创建矩阵作为输入。

## 第二步：创建输入集

代码如下：

```
# 构建输入集
x = np.mat('0 0;'
           '0 1;'
           '1 0;'
           '1 1')
x = torch.tensor(x).float()
y = np.mat('1;'
           '0;'
           '0;'
           '1')
y = torch.tensor(y).float()
```

我个人比较喜欢用np.mat这种方式构建矩阵，感觉写法比较简单，当然你也可以用其他的方法。但是构建完矩阵一定要有这一步torch.tensor(x).float()，必须要把你所创建的输入转换成tensor变量。

什么是tensor呢？你可以简单地理解他就是pytorch中用的一种变量，你想用pytorch这个框架就必须先把你的变量转换成tensor变量。而我们这个神经网络会要求你的输入和输出必须是float浮点型的，指的是tensor变量中的浮点型，而你用np.mat创建的输入是int型的，转换成tensor也会自动地转换成itensor的int型，所以要在后面加个.float（）转换成浮点型。

这样我们就构建完成了输入和输出（分别是x矩阵和y矩阵），x是四行二列的一个矩阵，他的每一行是一个输入，一次输入两个值，这里我们把所有的输入情况都列了出来。输出y是一个四行一列的矩阵，每一行都是一个输出，对应x矩阵每一行的输入。

## 第三步：搭建网络

代码如下：

```
# 搭建网络
myNet = nn.Sequential(
    nn.Linear(2, 10),
    nn.ReLU(),
    nn.Linear(10, 1),
    nn.Sigmoid()
)
print(myNet)
```

我们使用nn包中的Sequential搭建网络，这个函数就是那个可以让我们像搭积木一样搭神经网络的一个东西。

nn.Linear(2,10)的意思搭建输入层，里面的2代表输入节点个数，10代表输出节点个数。Linear也就是英文的线性，意思也就是这层不包括任何其它的激活函数，你输入了啥他就给你输出了啥 。nn.ReLU（）这个就代表把一个激活函数层，把你刚才的输入扔到了ReLU函数中去。 接着又来了一个Linear，最后再扔到Sigmoid函数中去。 2,10,1就分别代表了三个层的个数，简单明了。

## 第四步：设置优化器

代码如下：

```
# 设置优化器
optimzer = torch.optim.SGD(myNet.parameters(), lr=0.05)
loss_func = nn.MSELoss()
```

对这一步的理解就是，你需要有一个优化的方法来训练你的网络，所以这步设置了我们所要采用的优化方法。

torch.optim.SGD的意思就是采用SGD方法训练，你只需要吧你网络的参数和学习率传进去就可以了，分别是myNet.paramets和lr。 loss_func这句设置了代价函数，因为我们的这个问题比较简单，所以采用了MSE，也就是均方误差代价函数。

## 第五步：训练网络

代码如下：

```
for epoch in range(5000):
    out = myNet(x)
    loss = loss_func(out, y)  # 计算误差
    optimzer.zero_grad()  # 清除梯度
    loss.backward()
    optimzer.step()
```

我这里设置了一个5000次的循环（可能不需要这么多次），让这个训练的动作迭代5000次。每一次的输出直接用myNet（x），把输入扔进你的网络就得到了输出out（就是这么简单粗暴！），然后用代价函数和你的标准输出y求误差。 清除梯度的那一步是为了每一次重新迭代时清除上一次所求出的梯度，你就把这一步记住就行，初学不用理解太深。 loss.backward（）当然就是让误差反向传播，接着optimzer.step（）也就是让我们刚刚设置的优化器开始工作。

## 第六步：测试

代码如下：

```
print(myNet(x).data)
```

输出结果：

```
tensor([[0.9424],
        [0.0406],
        [0.0400],
        [0.9590]])
```

可以看到这个结果已经非常接近我们期待的结果了，当然你也可以换个数据测试，结果也会是相似的。这里简单解释下为什么我们的代码末尾加上了一个.data，因为我们的tensor变量其实是包含两个部分的，一部分是tensor数据，另一部分是tensor的自动求导参数，我们加上.data意思是输出取tensor中的数据，如果不加的话会输出下面这样：

```
tensor([[0.9492],
        [0.0502],
        [0.0757],
        [0.9351]], grad_fn=<SigmoidBackward>)
```

今天的分享就到这里，本人第一次发博客，写得不妥当的地方还望大家批评指正，不知道有没有帮到初学pytorch的你，完整代码如下：

```
import torch
import torch.nn as nn
import numpy as np

# 构建输入集
x = np.mat('0 0;'
           '0 1;'
           '1 0;'
           '1 1')
x = torch.tensor(x).float()
y = np.mat('1;'
           '0;'
           '0;'
           '1')
y = torch.tensor(y).float()

# 搭建网络
myNet = nn.Sequential(
    nn.Linear(2, 10),
    nn.ReLU(),
    nn.Linear(10, 1),
    nn.Sigmoid()
)
print(myNet)

# 设置优化器
optimzer = torch.optim.SGD(myNet.parameters(), lr=0.05)
loss_func = nn.MSELoss()

for epoch in range(5000):
    out = myNet(x)
    loss = loss_func(out, y)  # 计算误差
    optimzer.zero_grad()  # 清除梯度
    loss.backward()
    optimzer.step()


print(myNet(x).data)
```
