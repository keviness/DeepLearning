## 三，Pytorch自动微分（autograde）
>在深度学习中，经常需要对函数求梯度（gradient）。PyTorch提供的autograd包能够根据输入和前向传播过程自动构建计算图，并执行反向传播。
* autograd包是PyTorch中所有神经网络的核心。
* 该autograd软件包为Tensors上的所有操作提供自动微分。
* 它是一个由运行定义的框架，这意味着以代码运行方式定义你的后向传播，并且每次迭代都可以不同。

### （一）torch.Tensor
* 1，torch.Tensor是包的核心类
* * 如果将其属性.requires_grad设置为True，则会开始跟踪针对tensor的所有操作。
* * 完成计算后，您可以调用.backward()来自动计算所有梯度。该张量的梯度将累积到.grad属性中。

* 2，要停止tensor历史记录的跟踪，可以调用.detach()，它将其与计算历史记录分离，并防止将来的计算被跟踪。
* * 要停止跟踪历史记录（和使用内存），还可以将代码块使用with torch.no_grad():包装起来。在评估模型时，这是特别有用，因为模型在训练阶段具有requires_grad = True的可训练参数有利于调参，但在评估阶段我们不需要梯度。

* 3，Function是另外一个很重要的类。Tensor和Function互相结合就可以构建一个记录有整个计算过程的有向无环图（DAG）。每个Tensor都有一个.grad_fn属性，该属性即创建该Tensor的Function, 就是说该Tensor是不是通过某些运算得到的，若是，则grad_fn返回一个与这些运算相关的对象，否则是None。

* 4，如果你想计算导数，你可以调用Tensor.backward()。如果Tensor是标量（即它包含一个元素数据），则不需要指定任何参数backward()，但是如果它有更多元素，则需要指定一个gradient参数来指定张量的形状。

### （二）代码示例
#### 1，创建一个张量，设置requires_grad=True
~~~py
x = torch.ones(2, 2, requires_grad=True)
print(x)
#输出：
tensor([[1., 1.],
        [1., 1.]], requires_grad=True)
~~~
#### 2，针对张量做一个操作
~~~py
y = x + 2
print(y)
#输出：
tensor([[3., 3.],
        [3., 3.]], grad_fn=<AddBackward0>)
#y 作为操作的结果被创建，所以它有 grad_fn
print(y.grad_fn)
#输出：
<AddBackward0 object at 0x7fe1db427470>
~~~

#### 3，针对y做更多算术操作
~~~py
z = y * y * 3
out = z.mean()
print(z, out)
#输出：
tensor([[27., 27.],
        [27., 27.]], grad_fn=<MulBackward0>) 
tensor(27., grad_fn=<MeanBackward0>)
~~~

#### 4，.requires_grad_( ... )
* .requires_grad_( ... ) 会改变张量的 requires_grad 标记。
* 输入的标记默认为 False ，如果没有提供相应的参数。
~~~py
a = torch.randn(2, 2)
a = ((a * 3) / (a - 1))
print(a.requires_grad)
a.requires_grad_(True)
print(a.requires_grad)
b = (a * a).sum()
print(b.grad_fn)
#输出：
False
True
<SumBackward0 object at 0x7fe1db427dd8>
~~~

#### 5，梯度
* 我们现在后向传播，因为输出包含了一个标量，out.backward() 等同于out.backward(torch.tensor(1.))。
* out.backward()
* 打印梯度 d(out)/dx
~~~py
print(x.grad)
#输出：
tensor([[4.5000, 4.5000],
        [4.5000, 4.5000]])
~~~

#### 6，雅可比向量积的例子
~~~py
x = torch.randn(3, requires_grad=True)
y = x * 2
while y.data.norm() < 1000:
    y = y * 2
print(y)
#输出：
tensor([ -444.6791,   762.9810, -1690.0941], grad_fn=<MulBackward0>)

#torch.autograd不能够直接计算整个雅可比，但是如果我们只想要雅可比向量积，只需要简单的传递向量给 backward 作为参数。
v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
y.backward(v)
print(x.grad)
#输出：
tensor([1.0240e+02, 1.0240e+03, 1.0240e-01])

#你可以通过将代码包裹在 with torch.no_grad()，来停止对从跟踪历史中 的 .requires_grad=True 的张量自动求导。
print(x.requires_grad)
print((x ** 2).requires_grad)
with torch.no_grad():
    print((x ** 2).requires_grad)
#输出：
True
True
False
~~~