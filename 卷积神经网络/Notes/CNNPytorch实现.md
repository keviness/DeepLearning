# PyTorch入门实战教程笔记（二十三）：卷积神经网络实现

## [PyTorch](https://so.csdn.net/so/search?q=PyTorch&spm=1001.2101.3001.7020)入门实战教程笔记（二十三）：卷积神经网络实现 1：Lenet5实现CIFAR10

### CIFAR10[数据集](https://so.csdn.net/so/search?q=数据集&spm=1001.2101.3001.7020)介绍

关于CIFAR-10数据集，可以访问它的官网进行下载：

[http://www.cs.toronto.edu/~kriz/cifar.html](http://www.cs.toronto.edu/~kriz/cifar.html)。

 CIFAR包含常见的10类物体的照片，照片的size 为32×32，每一类照片有6000张，所以一共6000万张照片，我们把6万张照片随机选出5万张照片作为training，剩余的1万张作为test.

![](https://fjjwhjwd3p.feishu.cn/space/api/box/stream/download/asynccode/?code=NzAxMGExMjBkODRmYTM1MmU5OGQxZmM0NDU5ZjNhNGZfS3NrTVE2UmdxaEZpZlQ3Y1VodWFaWmF5U2xrbGI1SUtfVG9rZW46Ym94Y25pZWRFbHZDbjVwUWR2R3llNW1aSGZmXzE2NTM1ODIxOTA6MTY1MzU4NTc5MF9WNA)

### CIFAR10代码实战准备

1. **数据集的加载与使用**，加载数据要用到的函数类：DataLoader、datasets、transforms，从对应的包中导入。过iter方法把DataLoader迭代器先得到，使用迭代器.next()方法得到一个batch，来验证数据的shape和label的shape，得到最终结果：x: torch.Size([32, 3, 32, 32]) label: torch.Size([32])。详细代码：

```Python
import  torch
from    torch.utils.data import DataLoader
from    torchvision import datasets
from    torchvision import transforms

def main():
    batchsz = 32

    #当前目录下新建文件夹'cifar'，train = True，transform对数据进行变换，download=True自动下载数据集
    cifar_train = datasets.CIFAR10('cifar', True, transform=transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]), download=True)
    #DataLoader方便一次加载多个，第一个参数为数据集cifar_train，第二个参数batch_size为每次批处理数量，
    #根据显卡设置batch_size，不要太小。第三个参数shuffle为打乱，设置成True。
    cifar_train = DataLoader(cifar_train, batch_size=batchsz, shuffle=True)

    cifar_test = datasets.CIFAR10('cifar', False, transform=transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]), download=True)
    cifar_test = DataLoader(cifar_test, batch_size=batchsz, shuffle=True)

    #通过iter方法把DataLoader迭代器先得到，使用迭代器.next()方法得到一个batch。
    x, label = iter(cifar_train).next()
    print('x:', x.shape, 'label:', label.shape)

if __name__ == '__main__':
    main()
```

2. **新建一个类lenet5**，所有的pytorch的神经结构类都要继承自nn.Module这个类，使用from torch import nn，将其导入。新建类的初始化方法，调用super(Lenet5, self).**init**() ，调用类的初始化方法类初始化父类。接下来参考下图来写网络层。

![](https://fjjwhjwd3p.feishu.cn/space/api/box/stream/download/asynccode/?code=M2VmNmNkMTlmM2ZlMzU4MDkyMDM2MjgyYTAyOTgxZDZfc2t6WGlQbmFSS3dQR3M5SjdZYjJKbmtWTGJhcFVDNXVfVG9rZW46Ym94Y241d3FYUW1GcHlzamRZNloydmI5ajdjXzE2NTM1ODIxOTA6MTY1MzU4NTc5MF9WNA)

 我们使用nn.Sequential(),将网络结构包在里面，使用nn.Conv2d()新建一个卷积层。Subsampling可通过nn.MaxPool2d/nn.AvgPool2d均可。写完卷积层后是全连接，需要先打平，但是pytorch没有打平这个类，我们需要重新在建一个类单元来实现。之后我们随机一个tmp = torch.randn(2, 3, 32, 32)当作图片，通过out =self.conv_unit(tmp),来查看卷积层的输出。所有的网络机构，都是有一个forward代表前向流程，并且能够自动的往回走一遍，所以不需要写backward，与from torch.nn import functional as F的F函数不同的是，nn.xxxx需要先初始化类，把参数先给它，然后方便后面调用，而F是函数，可以直接使用。此外，输出的y还没有给出，所欲对于loss函数，我们放在类外面做（下面代码已注释掉）。构建lenet5函数代码（及测试）如下：（也是完整的 lenet5.py）

```Python
import  torch
from    torch import nn
from    torch.nn import functional as F

class Lenet5(nn.Module):
    """
    for cifar10 dataset.
    """
    def __init__(self):
        super(Lenet5, self).__init__()

        self.conv_unit = nn.Sequential(
            # x: [b, 3, 32, 32] => [b, 6, ]
            #第一个参数为输入的channel，第二个参数为输出的channel，...
            nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            #
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            #
        )
        # flatten
        # fc unit
        self.fc_unit = nn.Sequential(
            nn.Linear(16*5*5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10)
        )


        # [b, 3, 32, 32]
        tmp = torch.randn(2, 3, 32, 32)
        out = self.conv_unit(tmp)
        # [b, 16, 5, 5]
        print('conv out:', out.shape)

        # # use Cross Entropy Loss
        #self.criteon = nn.CrossEntropyLoss()

    def forward(self, x):
        """
        :param x: [b, 3, 32, 32]
        :return:
        """
        batchsz = x.size(0)
        # [b, 3, 32, 32] => [b, 16, 5, 5]
        x = self.conv_unit(x)
        # [b, 16, 5, 5] => [b, 16*5*5]
        x = x.view(batchsz, 16*5*5)   #16*5*5也可写成-1
        # [b, 16*5*5] => [b, 10]
        logits = self.fc_unit(x)

        # # [b, 10]
        #pred = F.softmax(logits, dim=1)
        #nn.CrossEntropyLoss()包含softmax操作，所以不需要再写
        #loss = self.criteon(logits, y)

        return logits

def main():

    net = Lenet5()

    tmp = torch.randn(2, 3, 32, 32)
    out = net(tmp)
    print('lenet out:', out.shape)

if __name__ == '__main__':
    main()
```

### lenet5 训练cifar10实战

1. **前期准备**：我们需要优化器optim，所以from torch import nn, optim，并且将上述的lenet5网络导入到主文件，from lenet5 import Lenet5。接下来配置：将需要运算的通过.to(device)装换到GPU上去，并且使用nn.CrossEntropyLoss().to(device)的loss，

 和优化器：optimizer = optim.Adam(model.parameters(), lr=1e-3)，如下：

```Python
device = torch.device('cuda')
    model = Lenet5().to(device)

    criteon = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    print(model)
```

2. **训练代码**：通过for batchidx, (x, label) in enumerate(cifar_train)来对一个batch迭代一次(一次batch 32张图片)。并且将（x,label）都加载到GPU上，执行logits = model(x)，将数据送入模型，然后计算loss，在backward之前一定要将梯度清零，调用optimizer.step()，进行梯度更新。

```Python
for epoch in range(1):   #1改为1000

        model.train()
        for batchidx, (x, label) in enumerate(cifar_train):
            # 这里是对一个batch迭代一次，一次batch 32张图片
            # [b, 3, 32, 32], [b]
            x, label = x.to(device), label.to(device)

            logits = model(x)
            # logits: [b, 10], label: [b], loss: tensor scalar
            loss = criteon(logits, label)

            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 使用 .item()将最后一个标量loss转换成Numpy打印出来
        print(epoch, 'loss:', loss.item())
```

3. **测试代码**：因为测试过程不需要梯度更新，为了保险起见，使用with torch.no_grad()，通过for x, label in cifar_test，来加载测试数据，将x传入模型：logits = model(x)，然后将预测值最高的序列号作为预测结果，通过eq函数与真实label对比，将batch中正确的相加和，再最终累加，通过total_correct / total_num求得精度。

```Python
model.eval()
        with torch.no_grad():
            # test
            total_correct = 0
            total_num = 0
            for x, label in cifar_test:
                # [b, 3, 32, 32], [b]
                x, label = x.to(device), label.to(device)

                # [b, 10]
                logits = model(x)
                # [b]
                pred = logits.argmax(dim=1)
                # [b] vs [b] => scalar tensor
                correct = torch.eq(pred, label).float().sum().item()
                total_correct += correct
                total_num += x.size(0)
                # print(correct)

            acc = total_correct / total_num
            print(epoch, 'test acc:', acc)
```

4. 将上述的完整的完整的 lenet5.py和下面完整的 main.py放入一个工程下，运行main.py，即可实现数据加载、训练、测试全过程。完整的 main.py代码如下：

```Python
import  torch
from    torch.utils.data import DataLoader
from    torchvision import datasets
from    torchvision import transforms
from    torch import nn, optim

from    lenet5 import Lenet5

def main():
    batchsz = 32

    #当前目录下新建文件夹'cifar'，train = True，transform对数据进行变换，download=True自动下载数据集
    cifar_train = datasets.CIFAR10('cifar', True, transform=transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]), download=True)
    #DataLoader方便一次加载多个，第一个参数为数据集cifar_train，第二个参数batch_size为每次批处理数量，
    #根据显卡设置，不要太小。第三个参数shuffle为打乱，设置成True。
    cifar_train = DataLoader(cifar_train, batch_size=batchsz, shuffle=True)

    cifar_test = datasets.CIFAR10('cifar', False, transform=transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]), download=True)
    cifar_test = DataLoader(cifar_test, batch_size=batchsz, shuffle=True)

    #通过iter方法把DataLoader迭代器先得到，使用迭代器.next()方法得到一个batch。
    x, label = iter(cifar_train).next()
    print('x:', x.shape, 'label:', label.shape)


    device = torch.device('cuda')
    model = Lenet5().to(device)

    criteon = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    print(model)

    for epoch in range(1000):   

        model.train()
        for batchidx, (x, label) in enumerate(cifar_train):
            # 这里是对一个batch迭代一次，一次batch 32张图片
            # [b, 3, 32, 32], [b]
            x, label = x.to(device), label.to(device)

            logits = model(x)
            # logits: [b, 10], label: [b], loss: tensor scalar
            loss = criteon(logits, label)

            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 使用 .item()将最后一个标量loss转换成Numpy打印出来
        print(epoch, 'loss:', loss.item())


        model.eval()
        with torch.no_grad():
            # test
            total_correct = 0
            total_num = 0
            for x, label in cifar_test:
                # [b, 3, 32, 32], [b]
                x, label = x.to(device), label.to(device)

                # [b, 10]
                logits = model(x)
                # [b]
                pred = logits.argmax(dim=1)
                # [b] vs [b] => scalar tensor
                correct = torch.eq(pred, label).float().sum().item()
                total_correct += correct
                total_num += x.size(0)
                # print(correct)

            acc = total_correct / total_num
            print(epoch, 'test acc:', acc)



if __name__ == '__main__':
    main()
```
