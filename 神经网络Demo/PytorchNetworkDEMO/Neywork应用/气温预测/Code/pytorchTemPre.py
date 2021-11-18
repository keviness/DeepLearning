import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import datetime
from sklearn import preprocessing
import matplotlib
import warnings
 
warnings.filterwarnings("ignore")
 
features = pd.read_csv('temps.csv')
# 看看数据长什么样子，head()默认展示前五个
print('原始数据维度: {0}, 数据: \n{1} '.format(features.shape, features.head()))
 
# 独热编码    将week中的Fri、Sun等编码而不是String格式
features = pd.get_dummies(features)
features.head(5)
 
# 标签    也就要预测的温度的真实值
labels = np.array(features['actual'])
 
# 在特征中去掉标签
features = features.drop('actual', axis=1)
 
# 训练集每列名字单独保存，留备用
feature_list = list(features.columns)
 
# 转换成合适的格式
features = np.array(features)
 
input_features = preprocessing.StandardScaler().fit_transform(features)
 
print("\n标准化原始数据，维度：{0} 具体数据：\n{1}".format(input_features.shape, input_features))
 
# 构建网络模型
input_size = input_features.shape[1]
hidden_size = 128
output_size = 1
batch_size = 16
my_nn = torch.nn.Sequential(
    torch.nn.Linear(input_size, hidden_size),
    torch.nn.Sigmoid(),
    torch.nn.Linear(hidden_size, output_size),
)
cost = torch.nn.MSELoss(reduction='mean')  # 计算损失函数（均方误差)
optimizer = torch.optim.Adam(my_nn.parameters(), lr=0.001)  # 优化器
 
# 训练网络
losses = []
for i in range(500):
    batch_loss = []
    # MINI-Batch方法来进行训练
    for start in range(0, len(input_features), batch_size):
        end = start + batch_size if start + batch_size < len(input_features) else len(input_features)
        xx = torch.tensor(input_features[start:end], dtype=torch.float, requires_grad=True)
        yy = torch.tensor(labels[start:end], dtype=torch.float, requires_grad=True)
        prediction = my_nn(xx)
        loss = cost(prediction, yy)
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        # 所有optimizer都实现了step()方法，它会更新所有的参数。
        # 一旦梯度被如backward()之类的函数计算好后，我们就可以调用这个函数。
        optimizer.step()
        batch_loss.append(loss.data.numpy())
 
    # 打印损失  每100轮打印一次
    if i % 100 == 0:
        losses.append(np.mean(batch_loss))
        print(i, np.mean(batch_loss), batch_loss)
 
# 预测训练结果
x = torch.tensor(input_features, dtype=torch.float)
predict = my_nn(x).data.numpy()
 
# 转换日期格式
months = features[:, feature_list.index('month')]
days = features[:, feature_list.index('day')]
years = features[:, feature_list.index('year')]
dates = [str(int(year)) + '-' + str(int(month)) + '-' + str(int(day)) for year, month, day in zip(years, months, days)]
dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in dates]
# 创建一个表格来存日期和其对应的标签数值
true_data = pd.DataFrame(data={'date': dates, 'actual': labels})
 
# 同理，再创建一个来存日期和其对应的模型预测值
test_dates = [str(int(year)) + '-' + str(int(month)) + '-' + str(int(day)) for year, month, day in
              zip(years, months, days)]
test_dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in test_dates]
predictions_data = pd.DataFrame(data={'date': test_dates, 'prediction': predict.reshape(-1)})
 
# 开始画图
# matplotlib添加本地的支持中文的字体库，默认是英文的无法显示中文
matplotlib.rc("font", family='Songti SC')
# 真实值
plt.plot(true_data['date'], true_data['actual'], 'b+', label='真实值')
# 预测值
plt.plot(predictions_data['date'], predictions_data['prediction'], 'r+', label='预测值')
plt.xticks(rotation='60')
plt.legend()
 
# 图名
plt.xlabel('日期')
plt.ylabel('最高温度 (F：华氏)')
plt.title('真实温度和预测温度')
plt.show()