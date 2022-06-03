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
