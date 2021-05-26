## 三，Pytorch自动微分（autograde）
* autograd包是PyTorch中所有神经网络的核心。
* 该autograd软件包为Tensors上的所有操作提供自动微分。
* 它是一个由运行定义的框架，这意味着以代码运行方式定义你的后向传播，并且每次迭代都可以不同。

### （一）torch.Tensor
* 1，torch.Tensor是包的核心类
* * 如果将其属性.requires_grad设置为True，则会开始跟踪针对tensor的所有操作。
* * 完成计算后，您可以调用.backward()来自动计算所有梯度。该张量的梯度将累积到.grad属性中。

* 2，要停止 tensor 历史记录的跟踪，您可以调用 .detach()，它将其与计算历史记录分离，并防止将来的计算被跟踪。

* 3，要停止跟踪历史记录（和使用内存），还可以将代码块使用with torch.no_grad(): 包装起来。在评估模型时，这是特别有用，因为模型在训练阶段具有requires_grad = True的可训练参数有利于调参，但在评估阶段我们不需要梯度。

* 4，还有一个类对于autograd实现非常重要那就是Function。
Tensor和Function互相连接并构建一个非循环图，它保存整个完整的计算过程的历史信息。每个张量都有一个.grad_fn属性保存着创建了张量的 Function 的引用，（如果用户自己创建张量，则grad_fn是None ）。

* 5，如果你想计算导数，你可以调用Tensor.backward()。如果Tensor是标量（即它包含一个元素数据），则不需要指定任何参数backward()，但是如果它有更多元素，则需要指定一个gradient参数来指定张量的形状。
