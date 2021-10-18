# Pytorch搭建简易神经网络（四）:批处理

这里和大家简单介绍一下批处理以及优化器optimizer

## 批处理

在处理数据的过程中，为了使得整个网络有着更好的学习效果并且不会有过多的资源的浪费，所以有批处理的概念，具体的原理不多说，直接上代码

## 1、导包

```python3
import torch
import torch.utils.data as Data
```

我们设置BATCH_SIZE = 5，在不同的训练任务中可以根据自己的需求或者硬件的需求进行设置，较为常见的为8，16等

```text
BATCH_SIZE = 5
```

随机生成两组数据，为了直观给画出来

```text
x = torch.linspace(1,10,10)
y = torch.linspace(10,1,10)
# plt.scatter(x,y)
# plt.show()
```

![](https://pic2.zhimg.com/80/v2-ebe5d729f1b4cd5ca423b3519c312e35_1440w.jpg)

* **classtorch** . **utils** . **data** .**Dataset**表示Dataset的抽象类。^[[1]](https://zhuanlan.zhihu.com/p/115363495#ref_1)^

所有其他数据集都应该进行子类化。所有子类应该override `__len__`和 `__getitem__`，前者提供了数据集的大小，后者支持整数索引，范围从0到len(self)。

**class**  **torch** . **utils** . **data** . **TensorDataset** (data_tensor, target_tensor)

包装数据和目标张量的数据集。

通过沿着第一个维度索引两个张量来恢复每个样本。

**参数：**

* **data_tensor** ( *Tensor* ) －　包含样本数据
* **target_tensor** ( *Tensor* ) －　包含样本目标（标签）
* **class** **torch** . **utils** . **data** .**DataLoader**数据加载器。组合数据集和采样器，并在数据集上提供单进程或多进程迭代器。^[[2]](https://zhuanlan.zhihu.com/p/115363495#ref_2)^

**参数：**

* **dataset** ( *Dataset* ) – 加载数据的数据集。
* **batch_size** ( *int* , optional) – 每个batch加载多少个样本(默认: 1)。
* **shuffle** ( *bool* , optional) – 设置为 `True`时会在每个epoch重新打乱数据(默认: False).
* **sampler** ( *Sampler* , optional) – 定义从数据集中提取样本的策略。如果指定，则忽略 `shuffle`参数。
* **num_workers** ( *int* , optional) – 用多少个子进程加载数据。0表示数据将在主进程中加载(默认: 0)
* **collate_fn** ( *callable* , optional) –
* **pin_memory** ( *bool* , optional) –
* **drop_last** ( *bool* , optional) – 如果数据集大小不能被batch size整除，则设置为True后可删除最后一个不完整的batch。如果设为False并且数据集的大小不能被batch size整除，则最后一个batch将更小。(默认: False)

```text
torch_dataset = Data.TensorDataset(x , y )
loader = Data.DataLoader(
    dataset=torch_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
    )
```

该版本不需要输入data_tensor,target_tensor，否则会报错；同时对于Windows平台下不要使用多个子进程加载数据，否则会报错，Windows平台下多线程有点问题，具体原因不好说。

如果此时输出loader会对应的为loader加载的数据在物理硬件上的存储地址

![](https://pic4.zhimg.com/80/v2-4fe503363491b2ae58b63129d8d41d17_1440w.png)通过批处理，来输出每一批的数据，来达到直观的效果

```text
for epoch in range(3):
    for step ,(batch_x,batch_y) in enumerate(loader):
        print('Epoch:',epoch,'| step:',step,'|batch_x:',batch_x.numpy(),'|batch_y：',batch_y.numpy())
```

![](https://pic4.zhimg.com/80/v2-ea78404d40c63554e583dd1a7795180b_1440w.jpg)---

对于批处理的问题自己的理解也很片面，后续会加强学习来加深理解，有问题也欢迎各位小伙伴和我沟通交流，如有错误也欢迎各位指正

---

## 附完整代码

```text
import torch
import torch.utils.data as Data
import matplotlib.pyplot as plt

BATCH_SIZE = 5

x = torch.linspace(1,10,10)
y = torch.linspace(10,1,10)
# plt.scatter(x,y)
# plt.show()

# torch_dataset = Data.TensorDataset(data_tensor=x , target_tensor = y )
torch_dataset = Data.TensorDataset(x , y )
loader = Data.DataLoader(
    dataset=torch_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
    )
# print(loader)

for epoch in range(3):
    for step ,(batch_x,batch_y) in enumerate(loader):
        print('Epoch:',epoch,'| step:',step,'|batch_x:',batch_x.numpy(),'|batch_y：',batch_y.numpy())
```

## 参考

1. [^](https://zhuanlan.zhihu.com/p/115363495#ref_1_0)TensorDataset [https://pytorch-cn.readthedocs.io/zh/latest/package_references/data/#torchutilsdata](https://pytorch-cn.readthedocs.io/zh/latest/package_references/data/#torchutilsdata)
2. [^](https://zhuanlan.zhihu.com/p/115363495#ref_2_0)DataLoader [https://pytorch-cn.readthedocs.io/zh/latest/package_references/data/#torchutilsdata](https://pytorch-cn.readthedocs.io/zh/latest/package_references/data/#torchutilsdata)
