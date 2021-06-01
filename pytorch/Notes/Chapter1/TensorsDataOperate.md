## Chapter2: Tensors的数据操作
### 一，创建Tensors (张量)
> Tensors类似于NumPy的ndarrays，同时Tensors可以使用GPU进行计算
>"tensor"这个单词一般可译作“张量”，张量可以看作是一个多维数组。标量可以看作是0维张量，向量可以看作1维张量，矩阵可以看作是二维张量。
#### 1，Pytorch构建矩阵
##### （1）构造一个5x3矩阵，不初始化。
~~~py
x = torch.empty(5, 3)
print(x)
#输出:
tensor(1.00000e-04 *
       [[-0.0000,  0.0000,  1.5135],
        [ 0.0000,  0.0000,  0.0000],
        [ 0.0000,  0.0000,  0.0000],
        [ 0.0000,  0.0000,  0.0000],
        [ 0.0000,  0.0000,  0.0000]])
~~~
##### （2）构造一个随机初始化的矩阵：
~~~py
x = torch.rand(5, 3)
print(x)
#输出:
tensor([[ 0.6291,  0.2581,  0.6414],
        [ 0.9739,  0.8243,  0.2276],
        [ 0.4184,  0.1815,  0.5131],
        [ 0.5533,  0.5440,  0.0718],
        [ 0.2908,  0.1850,  0.5297]])
~~~
##### （3）构造一个矩阵全为 0，而且数据类型是 long.
~~~py
x = torch.zeros(5, 3, dtype=torch.long)
print(x)
#输出:
tensor([[ 0,  0,  0],
        [ 0,  0,  0],
        [ 0,  0,  0],
        [ 0,  0,  0],
        [ 0,  0,  0]])
~~~

#### 2，Pytorch构造一个张量
##### （1）直接使用数据：
~~~py
x = torch.tensor([5.5, 3])
print(x)
#输出:
tensor([ 5.5000,  3.0000])
~~~

##### （2）基于已经存在的tensor创建一个tensor
~~~py
#此方法会默认重用输入Tensor的一些属性，例如数据类型，除非自定义数据类型。
x = x.new_ones(5, 3, dtype=torch.float64)  # 返回的tensor默认具有相同的torch.dtype和torch.device
print(x)
x = torch.randn_like(x, dtype=torch.float) # 指定新的数据类型
print(x) 
#输出:
tensor([[1., 1., 1.],
        [1., 1., 1.],
        [1., 1., 1.],
        [1., 1., 1.],
        [1., 1., 1.]], dtype=torch.float64)
tensor([[ 0.6035,  0.8110, -0.0451],
        [ 0.8797,  1.0482, -0.0445],
        [-0.7229,  2.8663, -0.5655],
        [ 0.1604, -0.0254,  1.0739],
        [ 2.2628, -0.9175, -0.2251]])

#可以通过shape或者size()来获取Tensor的形状:
print(x.size())
print(x.shape)
#输出:
torch.Size([5, 3])
torch.Size([5, 3])
#注意
torch.Size 是一个元组，所以它支持所有的元组操作。
~~~
#### 3，创建Tensor函数总结
>这些创建方法都可以在创建的时候指定数据类型dtype和存放device(cpu/gpu)。

|函数	|  功能  |
|:----:|:------:|
|Tensor(*sizes) | 基础构造函数 |
|tensor(data,)	|类似np.array的构造函数 |
|ones(*sizes)	|全1Tensor|
|zeros(*sizes)	|全0Tensor|
|eye(*sizes)	|对角线为1，其他为0|
|arange(s,e,step) |	从s到e，步长为step|
|linspace(s,e,steps) |	从s到e，均匀切分成steps份|
|rand/randn(*sizes)  |	均匀/标准分布|
|normal(mean,std)/uniform(from,to) |	正态分布/均匀分布|
|randperm(m) |	随机排列 |


## 二，Tensor操作
### 1，Tensors加法操作
#### （1）方式1
~~~py
y = torch.rand(5, 3)
print(x + y)
#输出：
tensor([[-0.1859,  1.3970,  0.5236],
        [ 2.3854,  0.0707,  2.1970],
        [-0.3587,  1.2359,  1.8951],
        [-0.1189, -0.1376,  0.4647],
        [-1.8968,  2.0164,  0.1092]])
~~~
#### （2）方式2
~~~py
print(torch.add(x, y))
#输出：
tensor([[-0.1859,  1.3970,  0.5236],
        [ 2.3854,  0.0707,  2.1970],
        [-0.3587,  1.2359,  1.8951],
        [-0.1189, -0.1376,  0.4647],
        [-1.8968,  2.0164,  0.1092]])
~~~
#### （3）提供一个输出tensor作为参数
~~~py
result = torch.empty(5, 3)
torch.add(x, y, out=result)
print(result)
#输出：
tensor([[-0.1859,  1.3970,  0.5236],
        [ 2.3854,  0.0707,  2.1970],
        [-0.3587,  1.2359,  1.8951],
        [-0.1189, -0.1376,  0.4647],
        [-1.8968,  2.0164,  0.1092]])
~~~

#### （4）加法:in-place（原地修改）
~~~py
# adds x to y
y.add_(x)
print(y)
#输出：
tensor([[-0.1859,  1.3970,  0.5236],
        [ 2.3854,  0.0707,  2.1970],
        [-0.3587,  1.2359,  1.8951],
        [-0.1189, -0.1376,  0.4647],
        [-1.8968,  2.0164,  0.1092]])

#注意
#注：PyTorch操作inplace版本都有后缀_, 例如x.copy_(y), x.t_()
~~~

### 三，与NumPy类似的索引操作
>可以使用类似NumPy的索引操作来访问Tensor的一部分，需要注意的是：索引出来的结果与原数据共享内存，即修改一个，另一个会跟着修改。
~~~py
print(x[:, 1])
#输出
tensor([ 0.4477, -0.0048,  1.0878, -0.2174,  1.3609])

y = x[0, :]
y += 1
print(y)
print(x[0, :]) # 源tensor也被改了
#输出
tensor([1.6035, 1.8110, 0.9549])
tensor([1.6035, 1.8110, 0.9549])
~~~
#### 1，PyTorch其他高级的选择函数:
| 函数  |  功能   |
|:----:|:-------:|
|index_select(input, dim, index)| 在指定维度dim上选取，比如选取某些行、某些列
|masked_select(input, mask)	|例子如上，a[a>0]，使用ByteTensor进行选取
|nonzero(input)	                |非0元素的下标
|gather(input, dim, index)	|根据index，在dim维度上选取数据，输出的size与index一样

### 四，改变形状
> 可以使用torch.view改变Tensor形状
> Tenso形状的改变为原地修改
~~~py
x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)  # the size -1 is inferred from other dimensions
                   # -1所指的维度可以根据其他维度的值推出来
print(x.size(), y.size(), z.size())
#输出
torch.Size([4, 4]) torch.Size([16]) torch.Size([2, 8])

#注意：
#view()返回的新Tensor与源Tensor虽然可能有不同的size，但是是共享data的，即更改其中的一个，另外一个也会跟着改变。(顾名思义，view仅仅是改变了对这个张量的观察角度，内部数据并未改变)
x += 1
print(x)
print(y) # 也加了1
#输出
tensor([[1.6035, 1.8110, 0.9549],
        [1.8797, 2.0482, 0.9555],
        [0.2771, 3.8663, 0.4345],
        [1.1604, 0.9746, 2.0739],
        [3.2628, 0.0825, 0.7749]])
tensor([1.6035, 1.8110, 0.9549, 1.8797, 2.0482, 0.9555, 0.2771, 3.8663, 0.4345,
        1.1604, 0.9746, 2.0739, 3.2628, 0.0825, 0.7749])

#注意：
#1，如果我们想返回一个真正新的副本（即不共享data内存）该怎么办呢？Pytorch还提供了一个reshape()可以改变形状，但是此函数并不能保证返回的是其拷贝，所以不推荐使用。推荐先用clone创造一个副本然后再使用view。
#2，使用clone还有一个好处是会被记录在计算图中，即梯度回传到副本时也会传到源Tensor。
x_cp = x.clone().view(15)
x -= 1
print(x)
print(x_cp)
#输出
tensor([[ 0.6035,  0.8110, -0.0451],
        [ 0.8797,  1.0482, -0.0445],
        [-0.7229,  2.8663, -0.5655],
        [ 0.1604, -0.0254,  1.0739],
        [ 2.2628, -0.9175, -0.2251]])
tensor([1.6035, 1.8110, 0.9549, 1.8797, 2.0482, 0.9555, 0.2771, 3.8663, 0.4345,
        1.1604, 0.9746, 2.0739, 3.2628, 0.0825, 0.7749])

~~~
#### PyTorch线性代数
|函数  | 功能  |
|:----:|:----:|
|trace  |  对角线元素之和(矩阵的迹) |
|diag   |  对角线元素 |
|triu/tril    |	 矩阵的上三角/下三角，可指定偏移量 |
|mm/bmm       |	矩阵乘法，batch的矩阵乘法 |
addmm/addbmm/addmv/addr/baddbmm..  |	矩阵运算|
|t   |	转置  |
|dot/cross    |	 内积/外积|
|inverse	   | 求逆矩阵  |
|svd   |	奇异值分解 | 

### 五，使用.item()来获得元素tensor的value
>可将一个标量Tensor转换成一个Python number。
~~~py
x = torch.randn(1)
print(x)
print(x.item())
#输出
tensor([ 0.9422])  #标量Tensor
0.9422121644020081 #Python number
~~~

### 六，Tensor与Numpy之间的转换
##### （1）Tensor to NumPy array
~~~py
t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")
#Out:
t: tensor([1., 1., 1., 1., 1.])
n: [1. 1. 1. 1. 1.]

#---A change in the tensor reflects in the NumPy array---
t.add_(1)
print(f"t: {t}")
print(f"n: {n}")
#Out:
t: tensor([2., 2., 2., 2., 2.])
n: [2. 2. 2. 2. 2.]
~~~

##### （2）NumPy array to Tensor
~~~py
n = np.ones(5)
t = torch.from_numpy(n)
#---Changes in the NumPy array reflects in the tensor---
np.add(n, 1, out=n)
print(f"t: {t}")
print(f"n: {n}")
#Out:
t: tensor([2., 2., 2., 2., 2.], dtype=torch.float64)
n: [2. 2. 2. 2. 2.]