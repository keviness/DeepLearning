# PyTorch实战: 使用卷积神经网络对照片进行分类

**本文任务**

 我们接下来需要用CIFAR-10数据集进行分类，步骤如下：

1. 使用torchvision 加载并预处理CIFAR-10数据集
2. 定义网络
3. 定义损失函数和优化器
4. 训练网络并更新网络参数
5. 测试网络

对卷积不了解的同学建议先阅读

10分钟理解深度学习中的卷积

conv2d处理的数据是什么样的？

注意：文章末尾含有项目jupyter notebook实战教程下载可供大家课后实战操作

**一、CIFAR-10数据加载及预处理**

 CIFAR-10 是一个常用的彩色图片数据集，它有 10 个类别，分别是 airplane、automobile、bird、cat、deer、dog、frog、horse、ship和 truck。每张图片都是 3*32*32 ,也就是 三通道彩色图片，分辨率 32*32。

第一次运行torchvision会自动下载CIFAR-10数据集，大约163M。这里我将数据直接放到项目 data文件夹 中。

运行结果

注意，数据集中的照片数据是以 PIL.Image.Image类 形式存储的，在我们加载数据时，要注意将其转化为 Tensor类。

运行结果

DataLoader是一个可迭代的对象，它将dataset返回的每一条数据样本拼接成一个batch，并提供多线程加速优化和数据打乱等操作。当程序对 cirfar_dataset 的所有数据遍历完一遍， 对Dataloader也完成了一次迭代。

**二、定义网络**

 最早的卷积神经网络LeNet为例，学习卷积神经网络。

![](https://fjjwhjwd3p.feishu.cn/space/api/box/stream/download/asynccode/?code=NTJlMzY3MDc2NDY1OTBiNzc1NGJkNzBhNjJiMmFkMzhfZTB6UnFjQzZDek9mdXhKYWQ5clBsWHVnZ0xscU1qMURfVG9rZW46Ym94Y24zRElTUTRNeUlpZHdvSGp6a1E4eTRnXzE2NTM1ODE3NjY6MTY1MzU4NTM2Nl9WNA)

**2.1 第一个convolutions层**

 图中显示是单通道照片，但是由于我们的数据集中的照片是三通道照片。所以

该层输入的是 三通道图片，图片长宽均为32，那么通过kernel_size=5的卷积核卷积后的尺寸为（32-5+1）=28

同时要注意，第一个convolution中，图片由 三通道变为6通道， 所以在此卷积过程中，in_channels=3, out_channels=6

**2.2 第一subsampling层**

 该层输入数据是6通道，输出还为6通道，但是图片的长宽从28变为14，我们可以使用池化层来实现尺寸缩小一倍。这里我们使用MaxPool2d(2, 2)

**2.3 第二个convolutions层**

 该层输入的是6通道数据，输出为16通道数据，且图片长宽从14变为10。这里我们使用

nn.Conv2d(in_channels=6,

 out_channels=16,

 kernel_size=5)

**2.4 全连接层作用**

 在此之前的卷积层和池化层都属于特征工程层，用于从数据中抽取特征。而之后的多个全连接层，功能类似于机器学习中的模型，用于学习特征数据中的规律，并输出预测结果。

**2.5 第一全连接层full connection**

 第二个convolutions层输出的 数据形状为 (16, 5, 5) 的数组,是一个三维数据。

而在全连接层中，我们需要将其 展平为一个一维数据（样子类似于列表，长度为16*5*5）

nn.Linear(in_features=16*5*5,

 out_features=120) #根据图中，该输出为120

 2.6 第二全连接层

 该层的输入是一维数组，长度为120，输出为一维数组，长度为84.

nn.Linear(in_features=120,

 out_features=84) #根据图中，该输出为84

**2.7 第三全连接层**

 该层的输入是一维数组，长度为84，输出为一维数组，长度为10，该层网络定义如下

nn.Linear(in_features=84,

 out_features=10) #根据图中，该输出为10

**注意：**

 这里的长度10的列表，可以看做输出的label序列。例如理想情况下

output = [1, 0, 0, 0, 0, 0, 0 ,0, 0 ,0]

 该output表示 input数据 经过该神经网络运算得到的 预测结果 显示的类别是 第一类

同理，理想情况下

output2 = [0, 1, 0, 0, 0, 0, 0 ,0, 0 ,0]

 该output2表示 input数据 经过该神经网络运算得到的 预测结果 显示的类别是 第二类

根据前面对LeNet网络的解读，现在我们用pytorch来定义LeNet网络结构

实例化神经网络LeNet

net = LeNet()

 net

 运行结果

我们随机传入一批照片（batch_size=4） ，将其输入给net，看输出的结果是什么情况。

**注意：**

运行结果

**t.max(input, dim)**

* input：传入的tensor
* dim: tensor的方向。dim=1表示按照行方向计算最大值

t.max(outputs, dim=1)

 运行结果

(tensor([0.1963, 0.2260, 0.2168, 0.2025], grad_fn=`<MaxBackward0>`),

 tensor([0, 0, 0, 0]))

 上述的操作，找到了outputs中四个最大的值，及其对应的index（该index可以理解为label）

**三、定义损失函数和优化器**

 神经网络强大之处就在于 反向传播，通过比较 预测结果 与 真实结果， 修整网络参数。

这里的 比较 就是 损失函数，而 修整网络参数 就是 优化器。

这样充分利用了每个训练数据，使得网络的拟合和预测能力大大提高。

from torch import optim

#定义交叉熵损失函数

 criterion = nn.CrossEntropyLoss()

#随机梯度下降SGD优化器

 optimizer = optim.SGD(params = net.parameters(),

 lr = 0.001)

**四、训练网络**

 所有网络的训练的流程都是类似的，不断执行（轮）：

* 给网络输入数据
* 前向传播+反向传播
* 更新网络参数

遍历完一遍数据集称为一个epoch，这里我们进行 2个epoch 轮次的训练。

运行结果

**五、测试网络**

**5.1 打印误差曲线**

![](https://fjjwhjwd3p.feishu.cn/space/api/box/stream/download/asynccode/?code=MWIwZWVhNmJjNWNiZTQzMjNlZjUwZGFhOTEwODBmOGVfWlRtVG9kUmxZWlpjb0J2VERhQk1BdUhCdmZGT0dRd1pfVG9rZW46Ym94Y255UEd3NWRaVnlxRURKa3FQNGFPcXVjXzE2NTM1ODE3NjY6MTY1MzU4NTM2Nl9WNA)

**5.2 查看训练的准确率**

 我们使用测试集检验训练的神经网络的性能。

运行结果

10000张测试集中准确率为： 57%

 数据集一共有10种照片，且每种照片数量相等。所以理论上，我们猜测对每一张照片的概率为10%。

而通过我们神经网络LeNet预测的准确率达到 57%，证明网络确实学习到了规律。
