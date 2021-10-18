# Pytorch搭建简单神经网络（三):网络快速搭建、保存与提取

但是如何快速搭建一个简单的神经网络而不是定义一个类再去调用，以及我们定义了一个网络并训练好，该如何在日后去调用这个网络去实现相应的功能。

## 其他的相关代码

关于导包以及生成数据的代码在这里不过多赘述，直接复制粘贴就好了

```python3
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.autograd import Variable

x = torch.unsqueeze(torch.linspace(-1,1,100),dim=1)
y = x.pow(3)+0.1*torch.randn(x.size())

x , y =(Variable(x),Variable(y))
```

之前用类定义的网络

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
        out = F.relu(out)
        out =self.predict(out)

        return out
```

输出网络可以看出定义的网络如下

```text
net = Net(1,20,1)
print(net)
```

## ![img](https://pic2.zhimg.com/80/v2-e25c1b50173f7b69e55e3a92015beaa9_1440w.jpg)一、快速搭建

接下来用快速搭建的方式来定义网络

需要使用到 torch.nn.Sequential(* args)，一个时序容器。`Modules`会以他们传入的顺序被添加到容器中。当然，也可以传入一个 `OrderedDict`^[[1]](https://zhuanlan.zhihu.com/p/115251842#ref_1)^

```text
 net1 = torch.nn.Sequential(
    nn.Linear(1,20),
    torch.nn.ReLU(),
    nn.Linear(20,20),
    torch.nn.ReLU(),
    nn.Linear(20,1)
)
```

输出net后可以看到

```text
print(net1)
```

![img](https://pic2.zhimg.com/80/v2-1895bd5c1eb4968d6cc64cab011b6fb5_1440w.jpg)可以看出和上面搭建的网络的类型是一样的，自然通过训练后的功能是一样的，关于训练的 代码直接贴在这里，具体的不多赘述，可以参考

```text
optimizer = torch.optim.SGD(net.parameters(),lr = 0.05)
loss_func = torch.nn.MSELoss()

plt.ion()
plt.show()

for t in range(2000):
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
```

---

那么对于第一部分中已经训练好的网络，需要做的就是对训练好的网络进行保存，保存的方式有两种，一种是直接对网络的所有都进行保存 ^[[2]](https://zhuanlan.zhihu.com/p/115251842#ref_2)^ ，一种对网络中的参数进行保存，保存优化选项默认字典^[[3]](https://zhuanlan.zhihu.com/p/115251842#ref_3)^

## 二、网络的保存

在训练后的网络中直接进行两种不同的保存方式

```text
torch.save(net,'net.pkl')    #保存所有的网络参数
torch.save(net.state_dict(),'net_parameter.pkl')    #保存优化选项默认字典，不保存网络结构
```

运行后在当前目录生成指定pkl文件

![](https://pic2.zhimg.com/80/v2-c3f5e3a9be6a0c9d8ede00ca9ef91be9_1440w.jpg)---

## 三、网络的提取

那么成功保存了，咱们得后续的去提取它

## 1、提取整个网络的方法

直接调用torch.load来提取整个网络

torch.load从磁盘文件中读取一个通过 `torch.save()`保存的对象。`torch.load()`可通过参数 `map_location`动态地进行内存重映射，使其能从不动设备中读取文件。一般调用时，需两个参数: storage 和 location tag. 返回不同地址中的storage，或着返回None (此时地址可以通过默认方法进行解析). 如果这个参数是字典的话，意味着其是从文件的地址标记到当前系统的地址标记的映射。^[[4]](https://zhuanlan.zhihu.com/p/115251842#ref_4)^

```text
net1 = torch.load('net.pkl')
```

## 2、提取网络中的参数的方法

对于提取网络中的参数的方式，必须先完整的建立和需要提取的网络一样的结构的网络，再去提取参数进而恢复网络

```text
net2 = torch.nn.Sequential(
    nn.Linear(1,20),
    torch.nn.ReLU(),
    nn.Linear(20,20),
    torch.nn.ReLU(),
    nn.Linear(20,1)
)
net2.load_state_dict(torch.load('net_parameter.pkl'))
```

以上两种方法恢复的网络是一样的，有的朋友肯定会问，既然都一样的，为什么我们不选择直接恢复而是选择先建立一样的网络再去恢复参数。因为在大型神经网络中，网络的结构很复杂网络的参数也很发杂，所以直接保存整个网络会占用很大的磁盘资源，就本次实验的一个例子就可以看出，保存网络参数和保存网络结构对磁盘的占用是完全不同的，所以在大型神经网络中更倾向于用保存参数的方式去保存真个网络。

![](https://pic2.zhimg.com/80/v2-f8e811ff105ebdd524cda512d4b52a0d_1440w.png)接下里咱们来看看整个网络恢复后的功能吧

```text
prediction1 = net1(x)
prediction2 = net2(x)

#可视化的部分
plt.figure(1,figsize=(10,3))
plt.subplot(121)
plt.title('net1')
plt.scatter(x.data.numpy(), y.data.numpy())
plt.plot(x.data.numpy(), prediction1.data.numpy(), 'r-', lw=5)
# plt.show()

plt.figure(1,figsize=(10,3))
plt.subplot(122)
plt.title('net2')
plt.scatter(x.data.numpy(), y.data.numpy())
plt.plot(x.data.numpy(), prediction2.data.numpy(), 'r-', lw=5)
plt.show()
```

![](https://pic1.zhimg.com/80/v2-245489d0b9c6b526b5112f57d180d60c_1440w.jpg)
复原后网络的结果是完全一样的

---

## 附整个程序的所有代码

```text
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.autograd import Variable

x = torch.unsqueeze(torch.linspace(-1,1,100),dim=1)
y = x.pow(3)+0.1*torch.randn(x.size())

x , y =(Variable(x),Variable(y))

'''
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
        out = F.relu(out)
        out =self.predict(out)

        return out
'''

'''
net = torch.nn.Sequential(
    nn.Linear(1,20),
    torch.nn.ReLU(),
    nn.Linear(20,20),
    torch.nn.ReLU(),
    nn.Linear(20,1)
)

# net = Net(1,20,1)
print(net)
'''
'''
optimizer = torch.optim.SGD(net.parameters(),lr = 0.05)
loss_func = torch.nn.MSELoss()

plt.ion()
plt.show()

for t in range(2000):
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

torch.save(net,'net.pkl')
torch.save(net.state_dict(),'net_parameter.pkl')
'''

net1 = torch.load('net.pkl')
net2 = torch.nn.Sequential(
    nn.Linear(1,20),
    torch.nn.ReLU(),
    nn.Linear(20,20),
    torch.nn.ReLU(),
    nn.Linear(20,1)
)
net2.load_state_dict(torch.load('net_parameter.pkl'))

prediction1 = net1(x)
prediction2 = net2(x)

plt.figure(1,figsize=(10,3))
plt.subplot(121)
plt.title('net1')
plt.scatter(x.data.numpy(), y.data.numpy())
plt.plot(x.data.numpy(), prediction1.data.numpy(), 'r-', lw=5)
# plt.show()

plt.figure(1,figsize=(10,3))
plt.subplot(122)
plt.title('net2')
plt.scatter(x.data.numpy(), y.data.numpy())
plt.plot(x.data.numpy(), prediction2.data.numpy(), 'r-', lw=5)
plt.show()
```

## 参考

1. [^](https://zhuanlan.zhihu.com/p/115251842#ref_1_0)torch.nn.Sequential中文文档 [https://pytorch-cn.readthedocs.io/zh/latest/package_references/torch-nn/#class-torchnnsequential-args](https://pytorch-cn.readthedocs.io/zh/latest/package_references/torch-nn/#class-torchnnsequential-args)
2. [^](https://zhuanlan.zhihu.com/p/115251842#ref_2_0)保存整个网络 [https://pytorch-cn.readthedocs.io/zh/latest/package_references/torch/#torchsavessource](https://pytorch-cn.readthedocs.io/zh/latest/package_references/torch/#torchsavessource)
3. [^](https://zhuanlan.zhihu.com/p/115251842#ref_3_0)保存优化选项默认字典 [https://pytorch-cn.readthedocs.io/zh/latest/package_references/torch-optim/#state_dict-source](https://pytorch-cn.readthedocs.io/zh/latest/package_references/torch-optim/#state_dict-source)
4. [^](https://zhuanlan.zhihu.com/p/115251842#ref_4_0)torch.load [https://pytorch-cn.readthedocs.io/zh/latest/package_references/torch/#torchloadsource](https://pytorch-cn.readthedocs.io/zh/latest/package_references/torch/#torchloadsource)
