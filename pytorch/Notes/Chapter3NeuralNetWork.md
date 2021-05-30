## Chapter3 神经网络
> 神经网络可以通过torch.nn包来构建。
>神经网络是基于自动梯度(autograd)来定义一些模型。一个nn.Module包括层和一个方法forward(input)它会返回输出(output)。

### 一，一个典型的神经网络训练过程
1，定义包含一些可学习参数(或者叫权重）的神经网络
2，在输入数据集上迭代
3，通过网络处理输入
4，计算损失(loss)(输出和正确答案的距离）
5，将梯度反向传播给网络的参数
6，更新网络的权重，一般使用一个简单的规则：weight = weight - learning_rate * gradient

### 二，定义神经网络
~~~py
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

net = Net()
print(net)
#输出：
Net(
  (conv1): Conv2d(1, 6, kernel_size=(5, 5), stride=(1, 1))
  (conv2): Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1))
  (fc1): Linear(in_features=400, out_features=120, bias=True)
  (fc2): Linear(in_features=120, out_features=84, bias=True)
  (fc3): Linear(in_features=84, out_features=10, bias=True)
)
#注意：我们只需要定义 forward 函数，backward函数会在使用autograd时自动定义，backward函数用来计算导数。我们可以在 forward 函数中使用任何针对张量的操作和计算。
~~~

#### （一）net.parameters()
>net.parameters()返回一个模型的可学习参数
~~~py
params = list(net.parameters())
print(len(params))
print(params[0].size())  # conv1's .weight
#输出：
10
torch.Size([6, 1, 5, 5])
~~~

#### （二）backward()
>backward()随机梯度的反向传播
~~~py
#让我们尝试一个随机的 32x32 的输入。注意:这个网络 (LeNet）的期待输入是 32x32 的张量。
#如果使用 MNIST 数据集来训练这个网络，要把图片大小重新调整到 32x32。
input = torch.randn(1, 1, 32, 32)
out = net(input)
print(out)
#输出：
tensor([[ 0.0399, -0.0856,  0.0668,  0.0915,  0.0453, -0.0680, -0.1024,  0.0493,
         -0.1043, -0.1267]], grad_fn=<AddmmBackward>)
#清零所有参数的梯度缓存，然后进行随机梯度的反向传播：
net.zero_grad()
out.backward(torch.randn(1, 10))
#注意：torch.nn只支持小批量处理 (mini-batches）。
#整个 torch.nn 包只支持小批量样本的输入，不支持单个样本的输入。比如，nn.Conv2d 接受一个4维的张量，即nSamples x nChannels x Height x Width 如果是一个单独的样本，只需要使用input.unsqueeze(0) 来添加一个“假的”批大小维度。
~~~