from sklearn import datasets
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as Fun

dataset = datasets.load_iris()
data = dataset['data']
iris_type = dataset['target']
print(data)
print(iris_type)

input = torch.FloatTensor(dataset['data'])
print(input)
label = torch.LongTensor(dataset['target'])
print(label)

# 定义BP神经网络
#方法一
class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.out = torch.nn.Linear(n_hidden, n_output)
      
    def forward(self,x):
        x = Fun.relu(self.hidden(x))
        x = self.out(x)
        return x
#方法二
'''
net2 = torch.nn.Sequential(
    torch.nn.Linear(2, 12),
    torch.nn.ReLU(),    #激活函数
    torch.nn.Linear(12, 1),
    torch.nn.Softplus()
    )
'''

net = Net(n_feature=4, n_hidden=20, n_output=3)
optimizer = torch.optim.SGD(net.parameters(), lr=0.05)
# SGD:随机梯度下降法
loss_func = torch.nn.CrossEntropyLoss
# 设定损失函数

for i in range(100):
    out = net(input)
    # 输出与label对比
    loss = loss_func(out, label)
    #初始化,必须在反向传播前先清零。
    optimizer.zero_grad()

    # 反向传播，计算各参数对于损失loss的梯度
    loss.backward()
  
    # 根据刚刚反向传播得到的梯度更新模型参数
    optimizer.step()

out = net(input)
# out是一个计算矩阵
prediction = torch.max(out, 1)[1]
pred_y = prediction.numpy()
# 预测y输出数列
target_y = label.data.numpy()
# 实际y输出数据