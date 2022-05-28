# 手写数字识别的卷积神经网络实现-pytorch

## [数据集](https://so.csdn.net/so/search?q=数据集&spm=1001.2101.3001.7020)下载

卷积神经网络经常被应用于手写数字的数据集[mnist](https://so.csdn.net/so/search?q=mnist&spm=1001.2101.3001.7020)的识别，若数据集下载出现异常，可以到【[MNIST数据集](http://yann.lecun.com/exdb/mnist/)】进行数据集下载。

```Python
train_data = torchvision.datasets.MNIST(
        root='./mnist',         #保存或者提取位置
        train=True,             #如果为True则为训练集，如果为False则为测试集
        transform=torchvision.transforms.ToTensor(),    #将图片转化成取值[0,1]的Tensor用于网络处理
        download=False           #是否下载数据集
    )
```

## 训练数据获取

```Python
def dataLoader():
    # 获取Mnist手写数字数据集
    train_data = torchvision.datasets.MNIST(
        root='./mnist',         #保存或者提取位置
        train=True,             #如果为True则为训练集，如果为False则为测试集
        transform=torchvision.transforms.ToTensor(),    #将图片转化成取值[0,1]的Tensor用于网络处理
        download=False           #是否下载数据集
    )
    # plt.imshow(train_data.train_data[0].numpy(),cmap='gray')
    # plt.title('%i'%train_data.train_labels[0])
    # plt.show()
    loader = Data.DataLoader(
        dataset=train_data,
        batch_size=50,      #最小训练批量
        shuffle=True,       #是否对数据进行随机打乱
        num_workers=2,      #多线程来读数据
    )

    return loader
```

## 测试数据获取

```Python
def dataTest():
    #获取测试数据
    test_data = torchvision.datasets.MNIST(root='./mnist/', train=False)
    #测试前2000个数据
    test_x = torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor)[:2000]/255.   to (2000, 1, 28, 28), value in range(0,1)
    test_y = test_data.test_labels[:2000]
    return test_x,test_y
```

## 网络搭建

```Python
#搭建网络
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(         # input shape (1, 28, 28)
            nn.Conv2d(
                in_channels=1,              # input height
                out_channels=16,            # n_filters
                kernel_size=5,              # filter size
                stride=1,                   # filter movement/step
                padding=2,                  # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
            ),                              # output shape (16, 28, 28)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(kernel_size=2),    # choose max value in 2x2 area, output shape (16, 14, 14)
        )
        self.conv2 = nn.Sequential(         # input shape (16, 14, 14)
            nn.Conv2d(16, 32, 5, 1, 2),     # output shape (32, 14, 14)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(2),                # output shape (32, 7, 7)
        )
        self.out = nn.Linear(32 * 7 * 7, 10)   # fully connected layer, output 10 classes

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)           # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        output = self.out(x)
        return output, x    # return x for visualization
```

![](https://fjjwhjwd3p.feishu.cn/space/api/box/stream/download/asynccode/?code=NWU0MzZjMTFmMWVhM2VhNTVjM2U5ZTdhYzQyNjBlOTZfZUdyOEFSOFFvUDE5bjVlT1NDMjJLczAzcnhPR1dyOUxfVG9rZW46Ym94Y244dzVlMlI0dU53bU5PZGIxbVF1RDFmXzE2NTM1ODM0ODU6MTY1MzU4NzA4NV9WNA)

## 训练过程

```Python
if __name__=="__main__":
    #模拟数据
    loader=dataLoader()
    test_x,test_y=dataTest()
    net=Net()
    print(net)
    #定义优化器
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    #定义误差函数
    loss_fun=nn.CrossEntropyLoss()

    plt.ion()
    #迭代训练
    for epoch in range(1):
        for step, (batch_x, batch_y) in enumerate(loader):
            #预测
            prediction=net(batch_x)[0]
            #计算误差
            loss=loss_fun(prediction,batch_y)
            #梯度降为0
            optimizer.zero_grad()
            #反向传递
            loss.backward()
            #优化梯度
            optimizer.step()

            if step % 50 == 0:
                test_output, last_layer = net(test_x)
                pred_y = torch.max(test_output, 1)[1].data.numpy()
                accuracy = float((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
                print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)
```

## 训练结果

![](https://fjjwhjwd3p.feishu.cn/space/api/box/stream/download/asynccode/?code=MTQyMWRmZjVmOThkOTgyNWZlMDk2NTk1N2Y3ODU2N2NfRDMwTzNFRFpESk5jajVUUGN1bXFTb2NETGRkcGhIZkxfVG9rZW46Ym94Y256UzRPME5nTFdMTTQ3NWIxUFFrcXVkXzE2NTM1ODM0ODU6MTY1MzU4NzA4NV9WNA)
