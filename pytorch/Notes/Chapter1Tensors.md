## Chapter2 Tensors
### 一，Tensors (张量)
> Tensors类似于NumPy的ndarrays，同时Tensors可以使用GPU进行计算

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
x = x.new_ones(5, 3, dtype=torch.double)      
# new_* methods take in sizes
print(x)
x = torch.randn_like(x, dtype=torch.float)    
# override dtype!
print(x)                                      
# result has the same size
#输出:
tensor([[ 1.,  1.,  1.],
        [ 1.,  1.,  1.],
        [ 1.,  1.,  1.],
        [ 1.,  1.,  1.],
        [ 1.,  1.,  1.]], dtype=torch.float64)
tensor([[-0.2183,  0.4477, -0.4053],
        [ 1.7353, -0.0048,  1.2177],
        [-1.1111,  1.0878,  0.9722],
        [-0.7771, -0.2174,  0.0412],
        [-2.1750,  1.3609, -0.3322]])
#获取它的维度信息:
print(x.size())
#输出:
torch.Size([5, 3])
#注意
torch.Size 是一个元组，所以它支持左右的元组操作。
~~~

#### 3，Tensors加法操作
##### （1）方式1
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
##### （2）方式2
~~~py
print(torch.add(x, y))
#输出：
tensor([[-0.1859,  1.3970,  0.5236],
        [ 2.3854,  0.0707,  2.1970],
        [-0.3587,  1.2359,  1.8951],
        [-0.1189, -0.1376,  0.4647],
        [-1.8968,  2.0164,  0.1092]])
~~~
##### （3）提供一个输出tensor作为参数
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

##### （4）加法:in-place（原地修改）
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
#任何使张量会发生变化的操作都有一个前缀 ‘’。例如：x.copy(y), x.t_(), 将会改变 x.
~~~

#### 4，使用NumPy类似的索引操作
~~~py
print(x[:, 1])
#输出
tensor([ 0.4477, -0.0048,  1.0878, -0.2174,  1.3609])
~~~

#### 5，改变Tensors大小
> 如果你想改变一个 tensor 的大小或者形状，你可以使用 torch.view:
~~~py
x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)  # the size -1 is inferred from other dimensions
print(x.size(), y.size(), z.size())
#输出
torch.Size([4, 4]) torch.Size([16]) torch.Size([2, 8])
~~~
#### 6，使用.item()来获得元素tensor的value
~~~py
x = torch.randn(1)
print(x)
print(x.item())
#输出
tensor([ 0.9422])
0.9422121644020081
~~~

#### 7，Tensor与Numpy之间的转换

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