# textCNN原理一览与基于Pytorch的文本分类案例

> 原文链接：

[【深度学习】textCNN论文与原理**mp.weixin.qq.com/s/866DT6aNywKfxKK-wbGSOw**![](https://pic2.zhimg.com/v2-59ecb09d3bc021b194f242c6ab964e29_180x120.jpg)](https://link.zhihu.com/?target=https%3A//mp.weixin.qq.com/s/866DT6aNywKfxKK-wbGSOw)

[【深度学习】textCNN论文与原理——短文本分类(基于pytorch)**mp.weixin.qq.com/s/NX-5HnT9iXptoPG6VJHWmw**![](https://pic2.zhimg.com/v2-59ecb09d3bc021b194f242c6ab964e29_180x120.jpg)](https://link.zhihu.com/?target=https%3A//mp.weixin.qq.com/s/NX-5HnT9iXptoPG6VJHWmw)

## **前言**

文本分类是自然语言处理中一个比较基础与常见的任务。咱也不谈传统的文本分类方法，就看看这个使用CNN是如何进行文本分类了，听说效果还不错。如果CNN不是很了解的话，可以看看我之前的文章： **卷积神经网络入门与基于Pytorch图像分类案例[1]** ，当然既然是一种深度学习方法进行文本分类，跑不了使用词向量相关内容，所以读者也是需要有一定词向量(也就是词语的一种分布式表示而已)的概念。对于使用CNN进行文本的原论文地址如下：**[https://**arxiv.org/abs/1408.5882**](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1408.5882)[2]** 感兴趣的话，可自行下载原文。

现在介绍一下如何使用textcnn进行文本分类，该部分内容参考了: **Pytorch-textCNN（不调用torchtext与调用torchtext）[3]** 。当然原文写的也挺好的，不过感觉不够工程化。现在我们就来看看如何使用pytorch和cnn来进行文本分类吧。

那么废话不多说，我们来一起看看论文主要内容与模型设计和一个小案例吧。

## **1 textcnn网络结构**

文中给出的网络模型如下：

![](https://pic4.zhimg.com/80/v2-4d61026684bf6ded401dc0e50f76d0c7_1440w.jpg)

从图中可以看出，模型还是很简单的。如果上图看不懂（涉及多个通道），我们也可以看看网上流行的另一张图（来自于论文：**[A Sensitivity Analysis of (and Practitioners’ Guide to) Convolutional Neural Networks for Sentence Classification](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/1510.03820.pdf)** ）：

![](https://pic4.zhimg.com/80/v2-e3e8b908058ab1e205c90789f6f0991b_1440w.jpg)

这里我们就按照上图去讲解这个模型。模型包含的模块如下：

* Embedding层：一个7x5的矩阵，其中每行是一个单词的向量，即词向量，词向量的维度为5
* Convolution层：这里是一维卷积，卷积核宽度固定：即为词向量维度，卷积核的高度即一个窗口中包含的单词的个数，这里的kernel_size=(2,3,4)即为一个窗口中分别可以包含2，3，4个单词。然后每个kernel_size输出2个通道，实际上是每类(包含单词个数)卷积核的个数为2而已，再将卷积输出的结果经过激活函数输出
* MaxPooling层：每类卷积核输出两个通道，然后再取各个通道结果的最大值(MaxPooling)，于是就得到了6个值
* FullConnection层：将MaxPooling层的输出结果进行拼接构成一个全连接层的输入，然后再根据分类类别数接上一个softmax层就可以得到分类结果了。

当然原文还使用了dropout，dropout的作用也可想而知，这里就不做介绍了。

**补充：**

 **通道（Channels）** ：图像中可以利用 (R, G, B) 作为不同channel；文本的输入的channel通常是不同方式的embedding方式（比如 word2vec或Glove），实践中也有利用静态词向量和fine-tunning词向量作为不同channel的做法。

 **一维卷积（conv-1d）** ：图像是二维数据；文本是一维数据，因此在TextCNN卷积用的是一维卷积（在word-level上是一维卷积；虽然文本经过词向量表达后是二维数据，但是在embedding-level上的二维卷积没有意义）。一维卷积带来的问题是需要通过设计不同 kernel_size 的 filter 获取不同宽度的视野，也可以理解为 **n-gram** 。

## **2 再看textcnn网络结构**

如果上面的图你不懂，可再看看这个计算过程吧，再不懂的话，我也没法了。

![](https://pic2.zhimg.com/80/v2-8ae6a546ad41bfb165427881a966a5f9_1440w.jpg)

其中模型还对比了输入的词向量的embedding层，

CNN-rand：作为一个基础模型，Embedding layer所有words被随机初始化，然后模型整体进行训练。

CNN-static：模型使用预训练的word2vec初始化Embedding layer，对于那些在预训练的word2vec没有的单词，随机初始化。然后固定Embedding layer，fine-tune整个网络。

CNN-non-static：同（2），只是训练的时候，Embedding layer跟随整个网络一起训练。

CNN-multichannel： Embedding layer有两个channel，一个channel为static，一个为non-static。然后整个网络fine-tune时只有一个channel更新参数。两个channel都是使用预训练的word2vec初始化的。

其实模型的主要内容到这里就完了，下面就是需要动手去实现他了，至于论文给出的实验结果，可以了解一下。

**接下来我找一些开源语料使用Pytorch来实现该模型了。**

## **3 实验语料介绍与预处理**

本文进行的任务本质是一个情感二分类的任务，语料内容为英文，其格式如下：

![](https://pic3.zhimg.com/80/v2-e13b0b12b3ccc178c60908e4122482de_1440w.png)

一行文本即实际的一个样本，样本数据分别在neg.txt和pos.txt文件中。在进行数据预处理之前，先介绍一下本任务可能用到的一些参数，这些参数我放在了一个config.py的文件中，内容如下：

```text
#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author: juzipi
@file: config.py
@time:2020/12/06
@description: 配置文件
"""
LARGE_SENTENCE_SIZE = 50  # 句子最大长度
BATCH_SIZE = 128          # 语料批次大小
LEARNING_RATE = 1e-3      # 学习率大小
EMBEDDING_SIZE = 200      # 词向量维度
KERNEL_LIST = [3, 4, 5]   # 卷积核长度
FILTER_NUM = 100          # 每种卷积核输出通道数
DROPOUT = 0.5             # dropout概率
EPOCH = 20                # 训练轮次
```

下面就是数据预处理过程啦，先把代码堆上来：

```text
import numpy as np
from collections import Counter
import random
import torch
from sklearn.model_selection import train_test_split
import config

random.seed(1000)
np.random.seed(1000)
torch.manual_seed(1000)


def read_data(filename):
    """
    数据读取
    :param filename: 文件路径
    :return: 数据读取内容（整个文档的字符串）
    """
    with open(filename, "r", encoding="utf8") as reader:
        content = reader.read()
    return content


def get_attrs():
    """
    获取语料相关参数
    :return: vob_size, pos_text, neg_text, total_text, index2word, word2index
    """
    pos_text, neg_text = read_data("corpus/pos.txt"), read_data("corpus/neg.txt")
    total_text = pos_text + '\n' + neg_text

    text = total_text.split()
    vocab = [w for w, f in Counter(text).most_common() if f > 1]
    vocab = ['<pad>', '<unk>'] + vocab

    index2word = {i: word for i, word in enumerate(vocab)}
    word2index = {word: i for i, word in enumerate(vocab)}

    return len(word2index), pos_text, neg_text, total_text, index2word, word2index


def convert_text2index(sentence, word2index, max_length=config.LARGE_SENTENCE_SIZE):
    """
    将语料转成数字化数据
    :param sentence: 单条文本
    :param word2index: 词语-索引的字典
    :param max_length: text_cnn需要的文本最大长度
    :return: 对语句进行截断和填充的数字化后的结果
    """
    unk_id = word2index['<unk>']
    pad_id = word2index['<pad>']
    # 对句子进行数字化转换，对于未在词典中出现过的词用unk的index填充
    indexes = [word2index.get(word, unk_id) for word in sentence.split()]
    if len(indexes) < max_length:
        indexes.extend([pad_id] * (max_length - len(indexes)))
    else:
        indexes = indexes[:max_length]
    return indexes


def number_sentence(pos_text, neg_text, word2index):
    """
    语句数字化处理
    :param pos_text: 正例全部文本
    :param neg_text: 负例全部文本
    :param word2index: 词到数字的字典
    :return: 经过训练集和测试集划分的结果X_train, X_test, y_train, y_test
    """
    pos_indexes = [convert_text2index(sentence, word2index) for sentence in pos_text.split('\n')]
    neg_indexes = [convert_text2index(sentence, word2index) for sentence in neg_text.split('\n')]

    # 为了方便处理，转化为numpy格式
    pos_indexes = np.array(pos_indexes)
    neg_indexes = np.array(neg_indexes)

    total_indexes = np.concatenate((pos_indexes, neg_indexes), axis=0)

    pos_targets = np.ones((pos_indexes.shape[0]))  # 正例设置为1
    neg_targets = np.zeros((neg_indexes.shape[0]))  # 负例设置为0

    total_targets = np.concatenate((pos_targets, neg_targets), axis=0).reshape(-1, 1)

    return train_test_split(total_indexes, total_targets, test_size=0.2)


def get_batch(x, y, batch_size=config.BATCH_SIZE, shuffle=True):
    """
    构建迭代器，获取批次数据
    :param x: 需要划分全部特征数据的数据集
    :param y: 需要划分全部标签数据的数据集
    :param batch_size: 批次大小
    :param shuffle: 是否打乱
    :return: 以迭代器的方式返回数据
    """
    assert x.shape[0] == y.shape[0], "error shape!"
    if shuffle:
        # 该函数是对[0, x.shape[0])进行随机排序
        shuffled_index = np.random.permutation(range(x.shape[0]))
        # 使用随机排序后的索引获取新的数据集结果
        x = x[shuffled_index]
        y = y[shuffled_index]

    n_batches = int(x.shape[0] / batch_size)  # 统计共几个完整的batch
    for i in range(n_batches - 1):
        x_batch = x[i*batch_size: (i + 1)*batch_size]
        y_batch = y[i*batch_size: (i + 1)*batch_size]
        yield x_batch, y_batch

```

其中各个函数怎么使用以及相关参数已经在函数的说明中了，这里再赘述就耽误观众姥爷的时间了，哈哈。这些代码我放在了一个dataloader.py的python文件中了，相信你会合理的使用它，如果有啥不明白的可以留言交流哦。

## **4 textcnn模型构建**

我依然先把代码堆出来，不是网传那么一句话嘛：“talk is cheap, show me code”，客官，代码来咯：

```text
#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author: juzipi
@file: model.py
@time:2020/12/06
@description:
"""
import numpy as np
import torch
from torch import nn, optim
import matplotlib.pyplot as plt
import config
import dataloader
import utils


class TextCNN(nn.Module):
    # output_size为输出类别（2个类别，0和1）,三种kernel，size分别是3,4，5，每种kernel有100个
    def __init__(self, vocab_size, embedding_dim, output_size, filter_num=100, kernel_list=(3, 4, 5), dropout=0.5):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # 1表示channel_num，filter_num即输出数据通道数，卷积核大小为(kernel, embedding_dim)
        self.convs = nn.ModuleList([
            nn.Sequential(nn.Conv2d(1, filter_num, (kernel, embedding_dim)),
                          nn.LeakyReLU(),
                          nn.MaxPool2d((config.LARGE_SENTENCE_SIZE - kernel + 1, 1)))
            for kernel in kernel_list
        ])
        self.fc = nn.Linear(filter_num * len(kernel_list), output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.embedding(x)  # [128, 50, 200] (batch, seq_len, embedding_dim)
        x = x.unsqueeze(1)     # [128, 1, 50, 200] 即(batch, channel_num, seq_len, embedding_dim)
        out = [conv(x) for conv in self.convs]
        out = torch.cat(out, dim=1)   # [128, 300, 1, 1]，各通道的数据拼接在一起
        out = out.view(x.size(0), -1)  # 展平
        out = self.dropout(out)        # 构建dropout层
        logits = self.fc(out)          # 结果输出[128, 2]
        return logits


# 数据获取
VOB_SIZE, pos_text, neg_text, total_text, index2word, word2index = dataloader.get_attrs()
# 数据处理
X_train, X_test, y_train, y_test = dataloader.number_sentence(pos_text, neg_text, word2index)
# 模型构建
cnn = TextCNN(VOB_SIZE, config.EMBEDDING_SIZE, 2)
# print(cnn)
# 优化器选择
optimizer = optim.Adam(cnn.parameters(), lr=config.LEARNING_RATE)
# 损失函数选择
criterion = nn.CrossEntropyLoss()


def train(model, opt, loss_function):
    """
    训练函数
    :param model: 模型
    :param opt: 优化器
    :param loss_function: 使用的损失函数
    :return: 该轮训练模型的损失值
    """
    avg_acc = []
    model.train()  # 模型处于训练模式
    # 批次训练
    for x_batch, y_batch in dataloader.get_batch(X_train, y_train):
        x_batch = torch.LongTensor(x_batch)  # 需要是Long类型
        y_batch = torch.tensor(y_batch).long()
        y_batch = y_batch.squeeze()  # 数据压缩到1维
        pred = model(x_batch)        # 模型预测
        # 获取批次预测结果最大值，max返回最大值和最大索引（已经默认索引为0的为负类，1为为正类）
        acc = utils.binary_acc(torch.max(pred, dim=1)[1], y_batch)
        avg_acc.append(acc)  # 记录该批次正确率
        # 使用损失函数计算损失值，预测值要放在前
        loss = loss_function(pred, y_batch)
        # 清楚之前的梯度值
        opt.zero_grad()
        # 反向传播
        loss.backward()
        # 参数更新
        opt.step()
    # 所有批次数据的正确率计算
    avg_acc = np.array(avg_acc).mean()
    return avg_acc


def evaluate(model):
    """
    模型评估
    :param model: 使用的模型
    :return: 返回当前训练的模型在测试集上的结果
    """
    avg_acc = []
    model.eval()  # 打开模型评估状态
    with torch.no_grad():
        for x_batch, y_batch in dataloader.get_batch(X_test, y_test):
            x_batch = torch.LongTensor(x_batch)
            y_batch = torch.tensor(y_batch).long().squeeze()
            pred = model(x_batch)
            acc = utils.binary_acc(torch.max(pred, dim=1)[1], y_batch)
            avg_acc.append(acc)
    avg_acc = np.array(avg_acc).mean()
    return avg_acc


# 记录模型训练过程中模型在训练集和测试集上模型预测正确率表现
cnn_train_acc, cnn_test_acc = [], []
# 模型迭代训练
for epoch in range(config.EPOCH):
    # 模型训练
    train_acc = train(cnn, optimizer, criterion)
    print('epoch={},训练准确率={}'.format(epoch, train_acc))
    # 模型测试
    test_acc = evaluate(cnn)
    print("epoch={},测试准确率={}".format(epoch, test_acc))
    cnn_train_acc.append(train_acc)
    cnn_test_acc.append(test_acc)


# 模型训练过程结果展示
plt.plot(cnn_train_acc)
plt.plot(cnn_test_acc)

plt.ylim(ymin=0.5, ymax=1.01)
plt.title("The accuracy of textCNN model")
plt.legend(["train", 'test'])
plt.show()

```

多说无益程序都在这，相关原理已经介绍了，各位读者慢慢品尝，有事call me。 对了，程序最后运行的结果如下：

![](https://pic1.zhimg.com/80/v2-f99d0dbd77ee1682bda7ee5b0e024230_1440w.jpg)

## **6 结果的一个简要分析**

其中随着模型的训练，模型倒是在训练集上效果倒好（毕竟模型在训练集上调整参数嘛），测试集上的结果也慢慢上升最后还略有下降，可见开始过拟合咯。本任务没有使用一些预训练的词向量以及语料介绍，总体也就1万多条，在测试集达到了这个效果也是差强人意了。主要想说明如何使用pytorch构建textcnn模型，实际中的任务可能更复杂，对语料的处理也更麻烦（数据决定模型的上限嘛）。或许看完这个文章后，你对损失函数、优化器、数据批次处理等还有一些未解之谜和改进的期待，我尽力在工作之余书写相关文章以飨读者，敬请关注哦。至于本文的全部代码和语料，我都上传到github上了: **[https://**github.com/Htring/NLP_A**pplications](https://link.zhihu.com/?target=https%3A//github.com/Htring/NLP_Applications)[4]** ，后续其他相关应用代码也会陆续更新，也欢迎star，指点哦。

### **参考资料**

[1] 卷积神经网络入门与基于Pytorch图像分类案例: *[https://**zhuanlan.zhihu.com/p/33**9777736](https://zhuanlan.zhihu.com/p/339777736)*

[2] [https://**arxiv.org/abs/1408.5882**:](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1408.5882%3A) *[https://**arxiv.org/abs/1408.5882**](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1408.5882)*

[3] Pytorch-textCNN（不调用torchtext与调用torchtext）: *[https://www.**cnblogs.com/cxq1126/p/1**3466998.html](https://link.zhihu.com/?target=https%3A//www.cnblogs.com/cxq1126/p/13466998.html)*

[4] [https://**github.com/Htring/NLP_A**pplications:](https://link.zhihu.com/?target=https%3A//github.com/Htring/NLP_Applications%3A) *[https://**github.com/Htring/NLP_A**pplications](https://link.zhihu.com/?target=https%3A//github.com/Htring/NLP_Applications)*
