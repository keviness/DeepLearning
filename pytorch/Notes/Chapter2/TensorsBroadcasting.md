## Chapter3: Tensors广播机制(Broadcasting) & Tensor on GPU/CPU
### 一，广播（broadcasting）
>广播（broadcasting）机制：先适当复制元素，使这两个Tensor形状相同后再按元素运算。
~~~py
x = torch.arange(1, 3).view(1, 2)
print(x)
y = torch.arange(1, 4).view(3, 1)
print(y)
print(x + y)
#输出：
tensor([[1, 2]])
tensor([[1],
        [2],
        [3]])
tensor([[2, 3],
        [3, 4],
        [4, 5]])
~~~

### 二，Tensor on GPU/CPU
#### （一）运算的内存开销
* 索引操作是不会开辟新内存的，而像y = x + y这样的运算是会新开内存的，然后将y指向新内存。
* 可以使用Python自带的id函数进行判断：如果两个实例的ID一致，那么它们所对应的内存地址相同；反之则不同。
~~~py
x = torch.tensor([1, 2])
y = torch.tensor([3, 4])
id_before = id(y)
y = y + x
print(id(y) == id_before) # False 
~~~

* 如果想指定结果到原来的y的内存，可以使用索引来进行替换操作。
* 在下面的例子中，把x+y的结果通过[:]写进y对应的内存中。
~~~py
x = torch.tensor([1, 2])
y = torch.tensor([3, 4])
id_before = id(y)
y[:] = y + x
print(id(y) == id_before) # True
~~~

* 可以使用运算符全名函数中的out参数或者自加运算符+=(也即add_())达到上述效果，例如torch.add(x, y, out=y)和y+=x(y.add_(x))。
~~~py
x = torch.tensor([1, 2])
y = torch.tensor([3, 4])
id_before = id(y)
torch.add(x, y, out=y) # y += x, y.add_(x)
print(id(y) == id_before) # True
#注：虽然view返回的Tensor与源Tensor是共享data的，但是依然是一个新的Tensor（因为Tensor除了包含data外还有一些其他属性），二者id（内存地址）并不一致。
~~~

#### （二）Tensor on GPU
>用方法to()可以将Tensor在CPU和GPU（需要硬件支持）之间相互移动。
~~~py
# 以下代码只有在PyTorch GPU版本上才会执行
if torch.cuda.is_available():
    device = torch.device("cuda")          # GPU
    y = torch.ones_like(x, device=device)  # 直接创建一个在GPU上的Tensor
    x = x.to(device)                       # 等价于 .to("cuda")
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))       # to()还可以同时更改数据类型
~~~