**PyTorch的神经网络保存和提取**

在学习和研究深度学习的时候，当我们通过一定时间的训练，得到了一个比较好的模型的时候，我们当然希望将这个模型及模型参数保存下来，以备后用，所以神经网络的保存和模型参数提取重载是很有必要的。

首先，我们需要在需要保存网路结构及其模型参数的神经网络的定义、训练部分之后通过torch.save()实现对网络结构和模型参数的保存。有两种保存方式：一是保存年整个神经网络的的结构信息和模型参数信息，save的对象是网络net；二是只保存神经网络的训练模型参数，save的对象是net.state_dict()，保存结果都以.pkl文件形式存储。

对应上面两种保存方式，重载方式也有两种。对应第一种完整网络结构信息，重载的时候通过torch.load(‘.pkl')直接初始化新的神经网络对象即可。对应第二种只保存模型参数信息，需要首先搭建相同的神经网络结构，通过net.load_state_dict(torch.load('.pkl'))完成模型参数的重载。在网络比较大的时候，第一种方法会花费较多的时间。

代码实现：

import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt

torch.manual_seed(1) # 设定随机数种子

# #创建数据

x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
y = x.pow(2) + 0.2*torch.rand(x.size())
x, y = Variable(x, requires_grad=False), Variable(y, requires_grad=False)

# #将待保存的神经网络定义在一个函数中

def save():

net1 = torch.nn.Sequential(
torch.nn.Linear(1, 10),
torch.nn.ReLU(),
torch.nn.Linear(10, 1),
)
optimizer = torch.optim.SGD(net1.parameters(), lr=0.5)
loss_function = torch.nn.MSELoss()

# #训练部分

for i in range(300):
prediction = net1(x)
loss = loss_function(prediction, y)
optimizer.zero_grad()
loss.backward()
optimizer.step()

# #绘图部分

plt.figure(1, figsize=(10, 3))
plt.subplot(131)
plt.title('net1')
plt.scatter(x.data.numpy(), y.data.numpy())
plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)

# #保存神经网络

torch.save(net1, '7-net.pkl')           # 保存整个神经网络的结构和模型参数
torch.save(net1.state_dict(), '7-net_params.pkl') # 只保存神经网络的模型参数

# #载入整个神经网络的结构及其模型参数

def reload_net():
net2 = torch.load('7-net.pkl')
prediction = net2(x)

plt.subplot(132)
plt.title('net2')
plt.scatter(x.data.numpy(), y.data.numpy())
plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)

# 只载入神经网络的模型参数，神经网络的结构需要与保存的神经网络相同的结构

def reload_params():

# #首先搭建相同的神经网络结构

net3 = torch.nn.Sequential(
torch.nn.Linear(1, 10),
torch.nn.ReLU(),
torch.nn.Linear(10, 1),
)

# #载入神经网络的模型参数

net3.load_state_dict(torch.load('7-net_params.pkl'))
prediction = net3(x)

plt.subplot(133)
plt.title('net3')
plt.scatter(x.data.numpy(), y.data.numpy())
plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)

# #运行测试

save()
reload_net()
reload_params()
