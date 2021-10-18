## [机器学习实战—搭建BP神经网络实现手写数字识别](https://www.cnblogs.com/ynqwer/p/14756872.html)

看了几天的BP神经网络，总算是对它有一点点的理解了。今天就用python搭建了一个模型来实现手写数字的识别。

### 一、BP神经网络简介

BP(back propagation)神经网络是一种按照误差逆向传播算法训练的多层前馈神经网络，是应用最广泛的一种神经网络。BP神经网络算法的基本思想是学习过程由信号正向传播和误差反向传播两个过程组成。
正向传播时，把样本的特征从输入层进行输入，信号经过各个隐藏层逐层处理之后，由输出层传出，对于网络的输出值与样本真实标签之间的误差，从最后一层逐层往前反向传播，计算出各层的学习信号，再根据学习信号来调整各层的权值参数。这种信号的正向传播和误差的反向传播是反复进行的，网络中权值调整的过程也就是模型训练的过程，反复训练模型，直到模型的代价函数小于某个预先设定的值，或者训练次数达到预先设置的最大训练次数为止。

### 二、手写数字数据集介绍

我用的手写数字数据集是 `sklearn.datasets`中的一个数据集，使用 `load_digits()`命令就可以载入数据集，数据集包含了1797个样本，也就是有1797张手写数字的图片，每个样本包含了64个特征，实际上每个样本就是一张8x8的图片，对应着0-9中的一个数字。看一下第一个样本长什么样子:

```python
from matplotlib import pyplot as plt
from sklearn.datasets import load_digits

# 载入数据集
digits = load_digits()
# 展示第一张图片，
plt.imshow(digits.images[0])
plt.show()

```

结果如下图:

![](https://img2020.cnblogs.com/blog/2136035/202105/2136035-20210511200212402-1802794488.png)

从结果也可以看出，是一张8x8的图片，这张图片显实的应该是数字0。

### 三、网络的介绍以及搭建

##### 1、网络的介绍

我搭建的是一个2层的神经网络，包含一个输入层（ *注意：输入层一般不计入网络的层数里面* ），一个隐藏层和一个输出层。由于每个样本包含64个特征，所以输入层设置了64个神经元，输出层设置了10个神经元，因为我将标签进行了独热化处理（ **样本有10种标签，独热化处理就会将每种标签转化成一个只包含0和1，长度为10的数组，例如：数字0的标签就为[1,0,0,0,0,0,0,0,0,0],数字1的标签为[0,1,0,0,0,0,0,0,0,0]，数字2的标签为[0,0,1,0,0,0,0,0,0,0]，以此类推** ），隐藏层的神经元数量可以随便设置，我设置的是100个神经元。对于神经网络的输出，也是一个长度为10的数组，只需要取出数组中最大数字对应的索引，即为预测的结果（ **例如：输出为[1,0,0,0,0,0,0,0,0,0]，最大数字的索引为0，即预测结果为0；输出为[0,0,0,0,0,0,0,0,0,1]，最大数字对应的索引为9，即预测结果为9** ）。网络中使用的激活函数为sigmoid函数。

##### 2、网络搭建

```python
import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import load_digits
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


class NeuralNetwork:
    def __init__(self, layers):
        # 初始化隐藏层权值
        self.w1 = np.random.random([layers[0], layers[1]]) * 2 - 1
        # 初始化输出层权值
        self.w2 = np.random.random([layers[1], layers[2]]) * 2 - 1
        # 初始化隐藏层的偏置值
        self.b1 = np.zeros([layers[1]])
        # 初始化输出层的偏置值
        self.b2 = np.zeros([layers[2]])

    # 定义激活函数
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # 定义激活函数的导函数
    @staticmethod
    def dsigmoid(x):
        return x * (1 - x)

    def train(self, x_data, y_data, lr=0.1, batch=50):
        """
        模型的训练函数
        :param x_data: 训练数据的特征
        :param y_data: 训练数据的标签
        :param lr: 学习率
        :param batch: 每次要训练的样本数量
        :return:
        """
        # 随机选择一定批次的数据进行训练
        index = np.random.randint(0, x_data.shape[0], batch)
        x = x_data[index]
        t = y_data[index]
        # 计算隐藏层的输出
        l1 = self.sigmoid(np.dot(x, self.w1) + self.b1)
        # 计算输出层的输出
        l2 = self.sigmoid(np.dot(l1, self.w2) + self.b2)
        # 计算输出层的学习信号
        delta_l2 = (t - l2) * self.dsigmoid(l2)
        # 计算隐藏层的学习信号
        delta_l1 = delta_l2.dot(self.w2.T) * self.dsigmoid(l1)
        # 计算隐藏层的权值变化
        delta_w1 = lr * x.T.dot(delta_l1) / x.shape[0]
        # 计算输出层的权值变化
        delta_w2 = lr * l1.T.dot(delta_l2) / x.shape[0]
        # 改变权值
        self.w1 += delta_w1
        self.w2 += delta_w2
        # 改变偏置值
        self.b1 += lr * np.mean(delta_l1, axis=0)
        self.b2 += lr * np.mean(delta_l2, axis=0)

    def predict(self, x):
        """
        模型的预测函数
        :param x: 测试数据的特征
        :return: 返回一个包含10个0-1之间数字的numpy.array对象
        """
        l1 = self.sigmoid(np.dot(x, self.w1) + self.b1)
        l2 = self.sigmoid(np.dot(l1, self.w2) + self.b2)
        return l2


# 载入数据集
digits = load_digits()
X = digits.data
T = digits.target
# 数据归一化
X = (X - X.min()) / (X.max() - X.min())
# 将数据拆分成训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(X, T)
# 将训练数据标签化为独热编码
labels = LabelBinarizer().fit_transform(y_train)
# 定义一个2层的网络模型：64-100-10
nn = NeuralNetwork([64, 100, 10])
# 训练周期
epoch = 20001
# 测试周期
test = 400
# 用来保存测试时产生的代价函数的值
loss = []
# 用来保存测试过程中的准确率
accuracy = []
for n in range(epoch):
    nn.train(x_train, labels)
    # 每训练一定的次数后，进行一次测试
    if n % test == 0:
        # 用测试集测试模型，返回结果为独热编码的标签
        predictions = nn.predict(x_test)
        # 取返回结果最大值的索引，即为预测数据
        y2 = np.argmax(predictions, axis=1)
        # np.equal用来比较数据是否相等，相等返回True，不相等返回False
        # 比较的结果求平均值，即为模型的准确率
        acc = np.mean(np.equal(y_test, y2))
        # 计算代价函数
        cost = np.mean(np.square(y_test - y2) / 2)
        # 将准确率添加到列表
        accuracy.append(acc)
        # 将代价函数添加到列表
        loss.append(cost)
        print('epoch:', n, 'accuracy:', acc, 'loss:', ls)

# 训练完成之后，使用测试数据对模型进行测试
pred = nn.predict(x_test)
y_pred = np.argmax(pred, axis=1)
# 查看模型预测结果与真实标签之间的报告
print(classification_report(y_test, y_pred))
# 查看模型预测结果与真实标签之间的混淆矩阵
print(confusion_matrix(y_test, y_pred))

plt.subplot(2, 1, 1)
plt.plot(range(0, epoch, test), loss)
plt.ylabel('loss')
plt.subplot(2, 1, 2)
plt.plot(range(0, epoch, test), accuracy)
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.show()

```

执行以上代码，可以看到代价函数和预测准确率随着模型的训练周期的变化，随着模型训练次数的增加，代价函数逐渐减小，然后趋于稳定，而准确率则是逐渐的增加，最后稳定在95%左右，画出图像如下图所示：

![](https://img2020.cnblogs.com/blog/2136035/202105/2136035-20210511215442840-1193468587.png)

经过200001次的训练之后，模型的准确率会稳定在95%左右，对于这个数据集来说，应该可以算是还不错的模型了。
