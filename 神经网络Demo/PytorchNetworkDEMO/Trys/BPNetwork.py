from sklearn import datasets
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as Fun
#from sklearn.datasets import load_iris
import pandas as pd

'''
#data = datasets.load_iris()
data = datasets.load_breast_cancer()
outputfile = "/Users/kevin/Desktop/program files/DeepLearning/神经网络Demo/PytorchNetworkDEMO/Trys/Data/breast_cancer.xls"  # 保存文件路径名
column = list(data['feature_names'])
dd = pd.DataFrame(data.data, index=range(569), columns=column)
dt = pd.DataFrame(data.target, index=range(569), columns=['outcome'])

jj = dd.join(dt, how='outer')  # 用到DataFrame的合并方法，将data.data数据与data.target数据合并
jj.to_excel(outputfile)  # 将数据保存到outputfile文件中
'''

dataset = datasets.load_iris()
#dataset = datasets.load_digits()
data = dataset['data']
iris_type = dataset['target']
print(data.shape)
print(iris_type)

inputData = torch.FloatTensor(dataset['data'])
#print(input)
label = torch.LongTensor(dataset['target'])
#print(label)


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
loss_func = torch.nn.CrossEntropyLoss()
# 设定损失函数

for i in range(1000):
    out = net(inputData)
    # 输出与label对比
    loss = loss_func(out, label)
    #初始化,必须在反向传播前先清零。
    optimizer.zero_grad()

    # 反向传播，计算各参数对于损失loss的梯度
    loss.backward()
  
    # 根据刚刚反向传播得到的梯度更新模型参数
    optimizer.step()

torch.save(net,'/Users/kevin/Desktop/program files/DeepLearning/神经网络Demo/PytorchNetworkDEMO/Trys/Data/model/net.pkl')  

model = torch.load('/Users/kevin/Desktop/program files/DeepLearning/神经网络Demo/PytorchNetworkDEMO/Trys/Data/model/net.pkl')
testData = np.array([[5.2, 3.2, 4.5, 0.2],[5.3, 3.5, 4.0, 0.1]])
inputTest = torch.FloatTensor(testData)

out =  model(inputTest)
print("out:\n", out)
# out是一个计算矩阵

prediction = torch.max(out, 1)[1]
print("prediction:\n", prediction)

# 预测y输出数列
pred_y = prediction.numpy()
print('pred_y:\n', pred_y)

# 实际y输出数据
target_y = label.data.numpy()
print('target_y:\n', target_y)
