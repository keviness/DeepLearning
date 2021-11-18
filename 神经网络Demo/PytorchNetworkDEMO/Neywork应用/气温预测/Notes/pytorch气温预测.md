# 搭建 PyTorch 神经网络进行气温预测 (回归任务)

搭建 PyTorch 神经网络进行气温预测 (回归任务)

### 数据处理

```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline
```

```
features = pd.read_csv('temps.cvs')

# 看看数据长什么样子
features.head()
```

```
   year  month  day  week  temp_2  temp_1  average  actual  friend
0  2016      1    1   Fri      45      45     45.6      45      29
1  2016      1    2   Sat      44      45     45.7      44      61
2  2016      1    3   Sun      45      44     45.8      41      56
3  2016      1    4   Mon      44      41     45.9      40      53
4  2016      1    5  Tues      41      40     46.0      44      41
```

数据表中

* year, moth, day, week 分别表示的具体的时间
* temp_2：前天的最高温度值
* temp_1：昨天的最高温度值
* average：在历史中，每年这一天的平均最高温度值
* actual：这就是我们的标签值了，当天的真实最高温度
* friend：这一列可能是凑热闹的，你的朋友猜测的可能值，咱们不管它就好了

```
print('数据维度:', features.shape)
```

```
数据维度: (348, 9)
```

处理时间数据, 便于后续可视化展示

```
import datetime

# 分别得到年, 月, 日
years = features['year']
months = features['month']
days = features['day']

# datetime 格式
dates = [str(int(year)) + '-' + str(int(month)) + '-' + str(int(day)) for year, month, day in zip(years, months, days)]
dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in dates]
```

```
dates[:5]
```

```
[datetime.datetime(2016, 1, 1, 0, 0), 
 datetime.datetime(2016, 1, 2, 0, 0), 
 datetime.datetime(2016, 1, 3, 0, 0), 
 datetime.datetime(2016, 1, 4, 0, 0), 
 datetime.datetime(2016, 1, 5, 0, 0)]
```

准备画图

```
# 准备画图
# 指定默认风格
plt.style.use('fivethirtyeight')

# 设置布局
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize = (10,10))
fig.autofmt_xdate(rotation = 45)

# 标签值
ax1.plot(dates, features['actual'])
ax1.set_xlabel(''); ax1.set_ylabel('Temperature'); ax1.set_title('Max Temp')

# 昨天
ax2.plot(dates, features['temp_1'])
ax2.set_xlabel(''); ax2.set_ylabel('Temperature'); ax2.set_title('Previous Max Temp')

# 前天
ax3.plot(dates, features['temp_2'])
ax3.set_xlabel('Date'); ax3.set_ylabel('Temperature'); ax3.set_title('Two Days Prior Max Temp')

# 我的逗逼朋友
ax4.plot(dates, features['friend'])
ax4.set_xlabel('Date'); ax4.set_ylabel('Temperature'); ax4.set_title('Friend Estimate')

plt.tight_layout(pad=2)
```

![]()

对特征 (week) 进行 one hot 编码, 转换为数值型数据

```
# one hot 编码
# get_dummies 会自动对传进来的数据做判断, 对需要处理的数据做 one hot 编码
features = pd.get_dummies(features)
features.head()
```

```
   year  month  day  temp_2  ...  week_Sun  week_Thurs  week_Tues  week_Wed
0  2016      1    1      45  ...         0           0          0         0
1  2016      1    2      44  ...         0           0          0         0
2  2016      1    3      45  ...         1           0          0         0
3  2016      1    4      44  ...         0           0          0         0
4  2016      1    5      41  ...         0           0          1         0

[5 rows x 15 columns]
```

标签是我们的期望值, 不能和输入数据混合

```
# 标签
labels = np.array(features['actual'])

# 在特征中去掉标签
features= features.drop('actual', axis = 1)

# 名字单独保存一下，以备后患
feature_list = list(features.columns)

# 转换成合适的格式
features = np.array(features)
```

```
features.shape
```

```
(348, 14)
```

数据标准化 (将数据浮动范围减小)

一般做完标准化之后, 数据收敛的速度会更快, 收敛的损失值也会更小

```
from sklearn import preprocessing
input_features = preprocessing.StandardScaler().fit_transform(features)
```

```
input_features[0]
```

```
array([ 0.        , -1.5678393 , -1.65682171, -1.48452388, -1.49443549,
       -1.3470703 , -1.98891668,  2.44131112, -0.40482045, -0.40961596,
       -0.40482045, -0.40482045, -0.41913682, -0.40482045])
```

### 构建网络模型

```
# 将 numpy 的 ndarray 格式转换为 pytorch 支持的 tensor 格式
x = torch.tensor(input_features, dtype = float)
y = torch.tensor(labels, dtype = float)
```

构建网络的第一步: 需要设计网络的结构

y = wx + b

我们的输入 x (348, 14) 有 14 个特征, 按照神经网络的做法, 我们需要将输入的特征转换为隐层的特征, 这里我们先用 128 个特征来表示隐层的特征 ([348, 14] x [14, 128]). 这里我们定义完了 w1.

偏置参数该如何定义呢? 对于 w1 来说我们得到了 128 个神经元特征, 偏置参数的目的是对网络进行微调.

在这里大家记住一点就行了, 对于偏置参数来说, 它的 shape 值或者说它的大小永远是跟你得到的结果是一致的, 我们的结果经过该隐层后得到了 128 个特征, 所以偏置参数也需要有 128 个. 表示我们需要对隐层中的 128 个特征做微调. 这里我们定义完了 b1.

对于回归任务来说, 我们需要得到的是一个实际的值, 所以我们需要将 128 转换为 1.

[348, 14] x [14, 128] x [128, 1]

所以对于 w2 来说, 它的形状需要设计为 [128, 1] 的矩阵; 同理 b2 应该设计为 1 个特征.

设计完网络结构后, 下一步我们需要对权重进行初始化操作.

这里我们用标准正态分布来初始化权重.

```
## 权重参数初始化 ([348,14] [14, 128] [128, 1])
# 将当前输入的特征 (这里有14个特征) 转换为隐藏的特征 (这里设计为 128 个神经元来表示)
weights = torch.randn((14, 128), dtype = float, requires_grad = True) 
# 偏置参数的 shape 与结果一致 (上面输出 128 个隐藏的特征), 故这里设置为 128, 即对这 128 个隐藏的特征都进行微调
biases = torch.randn(128, dtype = float, requires_grad = True) 
# 因为我们做的是回归任务, 需要得到一个实际的值, 即将这 128 个特征转换为一个值
weights2 = torch.randn((128, 1), dtype = float, requires_grad = True) 
# 同上, 取 1
biases2 = torch.randn(1, dtype = float, requires_grad = True) 

learning_rate = 0.001 
losses = []

for i in range(1000):
    # 计算隐层
    hidden = x.mm(weights) + biases
    # 加入激活函数
    hidden = torch.relu(hidden)
    # 得到预测结果
    predictions = hidden.mm(weights2) + biases2
    # 计算损失值 (均方误差)
    loss = torch.mean((predictions - y) ** 2) 
    losses.append(loss.data.numpy())

    # 打印损失值
    if i % 100 == 0:
        print('loss:', loss)
    #返向传播计算
    loss.backward()

    # 更新参数 (梯度下降)
    weights.data.add_(- learning_rate * weights.grad.data)  
    biases.data.add_(- learning_rate * biases.grad.data)
    weights2.data.add_(- learning_rate * weights2.grad.data)
    biases2.data.add_(- learning_rate * biases2.grad.data)

    # 每次迭代都得记得将梯度清空, 防止累加
    weights.grad.data.zero_()
    biases.grad.data.zero_()
    weights2.grad.data.zero_()
    biases2.grad.data.zero_()
```

```
loss: tensor(7781.1483, dtype=torch.float64, grad_fn=<MeanBackward0>)
loss: tensor(155.0691, dtype=torch.float64, grad_fn=<MeanBackward0>)
loss: tensor(146.8949, dtype=torch.float64, grad_fn=<MeanBackward0>)
loss: tensor(144.4075, dtype=torch.float64, grad_fn=<MeanBackward0>)
loss: tensor(143.0622, dtype=torch.float64, grad_fn=<MeanBackward0>)
loss: tensor(142.1858, dtype=torch.float64, grad_fn=<MeanBackward0>)
loss: tensor(141.5800, dtype=torch.float64, grad_fn=<MeanBackward0>)
loss: tensor(141.1374, dtype=torch.float64, grad_fn=<MeanBackward0>)
loss: tensor(140.7983, dtype=torch.float64, grad_fn=<MeanBackward0>)
loss: tensor(140.5365, dtype=torch.float64, grad_fn=<MeanBackward0>)
```

```
predictions.shape
```

```
torch.Size([348, 1])
```

### 更简单的构建网络模型

```
input_size = input_features.shape[1] # 14
hidden_size = 128
output_size = 1
batch_size = 16
my_nn = torch.nn.Sequential(
    # 第一层: 全连接层
    torch.nn.Linear(input_size, hidden_size),
    # 激活函数
    torch.nn.Sigmoid(),
    # 第二层: 全连接层
    torch.nn.Linear(hidden_size, output_size),
)
# 损失函数: 均方误差
cost = torch.nn.MSELoss(reduction='mean')
# 优化器 (动态调整学习率)
optimizer = torch.optim.Adam(my_nn.parameters(), lr = 0.001)
```

### 训练网络

```
# 训练网络
losses = []
for i in range(1000):
    batch_loss = []
    # MINI-Batch 方法来进行训练
    for start in range(0, len(input_features), batch_size):
        end = start + batch_size if start + batch_size < len(input_features) else len(input_features)
        # 一个 batch 输入数据
        xx = torch.tensor(input_features[start:end], dtype = torch.float, requires_grad = True)
        # 一个 batch 的期望值
        yy = torch.tensor(labels[start:end], dtype = torch.float, requires_grad = True)
        # 前向传播
        prediction = my_nn(xx)
        # 计算损失值
        loss = cost(prediction, yy)
        # 优化并对梯度做清零
        optimizer.zero_grad()
        # 反向传播
        loss.backward(retain_graph=True)
        # 更新参数
        optimizer.step()
        batch_loss.append(loss.data.numpy())

    # 打印损失
    if i % 100==0:
        losses.append(np.mean(batch_loss))
        print(i, np.mean(batch_loss))
```

```
0 3950.7627
100 37.9201
200 35.654438
300 35.278366
400 35.116814
500 34.986076
600 34.868954
700 34.75414
800 34.637356
900 34.516705
```

### 预测训练结果

```
x = torch.tensor(input_features, dtype = torch.float)
predict = my_nn(x).data.numpy()
```

```
# 转换日期格式
dates = [str(int(year)) + '-' + str(int(month)) + '-' + str(int(day)) for year, month, day in zip(years, months, days)]
dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in dates]

# 创建一个表格来存日期和其对应的标签数值
true_data = pd.DataFrame(data = {'date': dates, 'actual': labels})

# 同理，再创建一个来存日期和其对应的模型预测值
months = features[:, feature_list.index('month')]
days = features[:, feature_list.index('day')]
years = features[:, feature_list.index('year')]

test_dates = [str(int(year)) + '-' + str(int(month)) + '-' + str(int(day)) for year, month, day in zip(years, months, days)]

test_dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in test_dates]

predictions_data = pd.DataFrame(data = {'date': test_dates, 'prediction': predict.reshape(-1)}) 
```

```
# 真实值
plt.plot(true_data['date'], true_data['actual'], 'b-', label = 'actual')

# 预测值
plt.plot(predictions_data['date'], predictions_data['prediction'], 'ro', label = 'prediction')
plt.xticks(rotation = '60'); 
plt.legend()

# 图名
plt.xlabel('Date'); plt.ylabel('Maximum Temperature (F)'); plt.title('Actual and Predicted Values');
```

![]()
