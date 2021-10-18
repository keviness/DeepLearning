# 使用pytorch实现鸢尾花的分类——BP神经网络

上图构建的输入层+2个隐藏层+输出层，共计4层结构的神经网络。

因此是4->layer1->layer2->3的三分类问题。考虑可以使用多种算法进行分析，本文先介绍使用BP神经网络进行分析。

先读取数据，并将数据分类：

```
from sklearn import datasets
dataset = datasets.load_iris()
data = dataset['data']
iris_type = dataset['target']
print(data)
print(iris_type)
```

这里将data（输出5行）和iris_type输出一下

```javascript
[[5.1 3.5 1.4 0.2]
 [4.9 3.  1.4 0.2]
 [4.7 3.2 1.3 0.2]
 [4.6 3.1 1.5 0.2]
 [5.  3.6 1.4 0.2]]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2
 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
 2 2]
```

为计算需求 这里要将数据转换为Tensor模式

```
input = torch.FloatTensor(dataset['data'])
print(input)
label = torch.LongTensor(dataset['target'])
print(label)
```

分别输出为

```javascript
tensor([[5.1000, 3.5000, 1.4000, 0.2000],
        [4.9000, 3.0000, 1.4000, 0.2000],
        [4.7000, 3.2000, 1.3000, 0.2000],
        [4.6000, 3.1000, 1.5000, 0.2000],
        [5.0000, 3.6000, 1.4000, 0.2000]])
tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,        0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,        1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,        2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,        2, 2, 2, 2, 2, 2])
```

引入pytorch工具包 构建BP网络

```javascript
import torch.nn.functional as Fun
# 定义BP神经网络
class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.out = torch.nn.Linear(n_hidden, n_output)
      
    def forward(self,x):
        x = Fun.relu(self.hidden(x))
        x = self.out(x)
        return x
```

选定网络、优化器和损失函数

```
net = Net(n_feature=4, n_hidden=20, n_output=3)
optimizer = torch.optim.SGD(net.parameters(), lr=0.05)
# SGD:随机梯度下降法
loss_func = torch.nn.CrossEntropyLoss
# 设定损失函数
```

开始训练数据

```javascript
for i in range(1000):
    out = net(input)
    loss = loss_func(out, label)
    # 输出与label对比
    optimizer.zero_grad()
    # 初始化
    loss.backward()
    optimizer.step()
```

开始输出数据

```
out = net(input)
# out是一个计算矩阵
prediction = torch.max(out, 1)[1]
pred_y = prediction.numpy()
# 预测y输出数列
target_y = label.data.numpy()
# 实际y输出数据
```

输出正确率

```
正确率为: 0.98
```

该正确率明显高于前面的机器学习预测的正确率。
