import torch
import torch.nn as nn
import numpy as np

# 构建输入集
x = np.mat('0  0;'
           '0  1;'
           '1  0;'
           '1  1')
print("x:\n", x)
x = torch.tensor(x).float()
y = np.mat('1;'
           '0;'
           '0;'
           '1')
print("y:\n", y)
y = torch.tensor(y).float()

# 搭建网络
myNet = nn.Sequential(
    nn.Linear(2, 100),
    nn.ReLU(),
    nn.Linear(100, 2),
    nn.Sigmoid()
)
print(myNet)

# 设置优化器
optimzer = torch.optim.SGD(myNet.parameters(), lr=0.05)
loss_func = nn.MSELoss()

for epoch in range(100):
    out = myNet(x)
    loss = loss_func(out, y)  # 计算误差
    optimzer.zero_grad()  # 清除梯度
    loss.backward()
    optimzer.step()
    predictLabel = torch.max(out.data, 1)[1]
    accuracy = sum(predictLabel==y.flatten())/len(predictLabel)
    if epoch%50 == 0:
        print(f'{accuracy}:{loss}')

torch.save(myNet, '/Users/kevin/Desktop/program files/DeepLearning/神经网络Demo/PytorchNetworkDEMO/Trys/Data/model/myNet.pkl')

model = torch.load('/Users/kevin/Desktop/program files/DeepLearning/神经网络Demo/PytorchNetworkDEMO/Trys/Data/model/myNet.pkl')
out = model(x)
print('out:\n',out.data)
print('myNet(x).data:\n', myNet(x).data)
predict = torch.max(out.data, 1)[1]
accuracy = sum(predict==y.flatten())/len(predict)
print('predict:\n', predict)
print("accuracy:",accuracy)