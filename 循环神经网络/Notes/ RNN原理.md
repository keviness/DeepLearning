# 循环神经网络(RNN)知识入门

本文共3700余字，含少量数学公式，预计阅读时间20分钟

**循环神经网络（Recurrent Neural Networks, RNN）** 是一种常用的神经网络结构，它源自于1982年由Saratha Sathasivam提出的霍普菲尔德网络。其特有的循环概念及其最重要的结构——长短时记忆网络——使得它在处理和预测序列数据的问题上有着良好的表现。

本文将从如下几方面来具体阐述RNN：

* RNN的发展历史
* 什么是RNN？
* LSTM结构与GRU结构
* RNN的应用领域
* RNN实战示例

## **一、RNN的发展历史**

1986年，Elman等人提出了用于处理序列数据的 **循环神经网络** 。如同卷积神经网络是专门用于处理二维数据信息（如图像）的神经网络，循环神经网络是专用于处理序列信息的神经网络。循环网络可以扩展到更长的序列，大多数循环神经网络可以处理可变长度的序列，循环神经网络的诞生解决了传统神经网络在处理序列信息方面的局限性。

1997年，Hochreiter和Schmidhuber提出了**长短时记忆单元(Long Short-Term Memory, LSTM)** 用于解决标准循环神经网络时间维度的梯度消失问题(vanishing gradient problem)。标准的循环神经网络结构存储的上下文信息的范围有限，限制了RNN的应用。LSTM型RNN用LSTM单元替换标准结构中的神经元节点，LSTM单元使用输入门、输出门和遗忘门控制序列信息的传输，从而实现较大范围的上下文信息的保存与传输。

1998年，Williams和Zipser提出名为“ **随时间反向传播(Backpropagation Through Time, BPTT)** ”的循环神经网络训练算法。BPTT算法的本质是按照时间序列将循环神经网络展开，展开后的网络包含N(时间步长)个隐含单元和一个输出单元，然后采用反向误差传播方式对神经网络的连接权值进行更新。

2001年，Gers和Schmidhuber提出了具有重大意义的LSTM型RNN优化模型，在传统的LSTM单元中加入了 **窥视孔连接(peephole connections)** 。具有窥视孔连接的LSTM型RNN模型是循环神经网络最流行的模型之一，窥视孔连接进一步提高了LSTM单元对具有长时间间隔相关性特点的序列信息的处理能力。2005年，Graves成功将LSTM型RNN应用于语音处理；2007年，Hochreiter将LSTM型RNN应用于生物信息学研究。

## **二、什么是RNN？**

RNN背后的想法是利用顺序信息。在传统的神经网络中，我们假设所有输入（和输出）彼此独立。但对于许多任务而言，这是一个非常糟糕的想法。如果你想预测句子中的下一个单词，那你最好知道它前面有哪些单词。RNN被称为"循环"，因为它们对序列的每个元素执行相同的任务，输出取决于先前的计算。考虑RNN的另一种方式是它们有一个“记忆”，它可以捕获到目前为止计算的信息。理论上，RNN可以利用任意长序列中的信息，但实际上它们仅限于回顾几个步骤（稍后将详细介绍）。这是典型的RNN网络在![[公式]](https://www.zhihu.com/equation?tex=t)时刻展开的样子：

![](https://pic2.zhimg.com/80/v2-86996683dc2edbe2a2c8ef44013eb1a1_1440w.jpg)

其中，

* ![[公式]](https://www.zhihu.com/equation?tex=x_t)是输入层的输入；
* ![[公式]](https://www.zhihu.com/equation?tex=s_t)是隐藏层的输出，其中![[公式]](https://www.zhihu.com/equation?tex=s_0)是计算第一个隐藏层所需要的，通常初始化为全零；
* ![[公式]](https://www.zhihu.com/equation?tex=o_t)是输出层的输出

从上图可以看出，RNN网络的关键一点是![[公式]](https://www.zhihu.com/equation?tex=s_t)的值不仅取决于![[公式]](https://www.zhihu.com/equation?tex=x_t)，还取决于![[公式]](https://www.zhihu.com/equation?tex=s_%7Bt-1%7D)。

假设：

* ![[公式]](https://www.zhihu.com/equation?tex=f)是隐藏层激活函数，通常是非线性的，如tanh函数或ReLU函数；
* ![[公式]](https://www.zhihu.com/equation?tex=g)是输出层激活函数，可以是softmax函数

那么，循环神经网络的**前向计算过程**用公式表示如下：

![[公式]](https://www.zhihu.com/equation?tex=o_t+%3D+g%28V+%5Ccdot+s_t+%2B+b_2%29+%5C%5C)![[公式]](https://www.zhihu.com/equation?tex=s_t+%3D+f%28U+%5Ccdot+x_t+%2B+W+%5Ccdot+s_%7Bt-1%7D+%2B+b_1%29+%5C%5C)

通过两个公式的循环迭代，有以下推导：

![[公式]](https://www.zhihu.com/equation?tex=%5Cbegin%7Bsplit%7D+o_t+%26%3D+g%28V+%5Ccdot+s_t+%2B+b2%29+%5C%5C++%26%3D+g%28V+%5Ccdot+f%28U+%5Ccdot+x_t+%2B+W+%5Ccdot+s_%7Bt-1%7D+%2B+b_1%29+%2B+b_2%29+%5C%5C+%26%3D+g%28V+%5Ccdot+f%28U+%5Ccdot+x_t+%2B+W+%5Ccdot+f%28U+%5Ccdot+x_%7Bt-1%7D+%2B+W+%5Ccdot+s_%7Bt-2%7D+%2B+b_1%29+%2B+b_1%29+%2B+b_2%29+%5C%5C+%26%3D+g%28V+%5Ccdot+f%28U+%5Ccdot+x_t+%2B+W+%5Ccdot+f%28U+%5Ccdot+x_%7Bt-1%7D+%2B+W+%5Ccdot+f%28U+%5Ccdot+x_%7Bt-2%7D+%2B+...%29%29%29+%2B+b_2%29+%5Cend%7Bsplit%7D+%5C%5C)

可以看到，当前时刻的输出包含了历史信息，这说明循环神经网络对历史信息进行了保存。

这里有几点需要注意：

* 你可以将隐藏的状态![[公式]](https://www.zhihu.com/equation?tex=s_t)看作网络的记忆，它捕获有关所有先前时间步骤中发生的事件的信息。步骤输出![[公式]](https://www.zhihu.com/equation?tex=o_t)根据时间![[公式]](https://www.zhihu.com/equation?tex=t)的记忆计算。正如上面简要提到的，它在实践中有点复杂，因为![[公式]](https://www.zhihu.com/equation?tex=s_t)通常无法从太多时间步骤中捕获信息。
* 与在每层使用不同参数的传统深度神经网络不同，RNN共享相同的参数（所有步骤的![[公式]](https://www.zhihu.com/equation?tex=U)，![[公式]](https://www.zhihu.com/equation?tex=V)，![[公式]](https://www.zhihu.com/equation?tex=W)）。这反映了我们在每个步骤执行相同任务的事实，只是使用不同的输入，这大大减少了我们需要学习的参数总数。
* 上图在每个时间步都有输出，但根据任务，这可能不是必需的。例如，在预测句子的情绪时，我们可能只关心最终的输出，而不是每个单词之后的情绪。同样，我们可能不需要在每个时间步骤输入。所以，RNN结构可以是下列不同的组合：

![](https://pic2.zhimg.com/80/v2-54f5c9ace33ee991f133d15cf02a4cf5_1440w.jpg)

## **三、LSTM结构和GRU结构**

任何一个模型都不是完美的，针对其缺陷人们总会研究出来一些方法来优化该模型，RNN作为一个表现优秀的深度学习网络也不例外。下面首先让我们通过**BPTT公式**推导了解一下标准RNN结构的缺陷是什么。

## **1. RNN的梯度消失和梯度爆炸问题**

假设我们的时间序列只有三段，在![[公式]](https://www.zhihu.com/equation?tex=t%3D3)时刻，损失函数为

![[公式]](https://www.zhihu.com/equation?tex=L_3+%3D+%5Cfrac+12%28Y_3+-+O_3%29%5E2+%5C%5C)

则对于一次训练任务的损失函数为

![[公式]](https://www.zhihu.com/equation?tex=L+%3D+%5Csum_%7Bt%3D1%7D%5ETL_t+%5C%5C)

即每一时刻损失值的累加。

使用随机梯度下降法训练RNN其实就是对![[公式]](https://www.zhihu.com/equation?tex=W_x)、![[公式]](https://www.zhihu.com/equation?tex=W_s)、![[公式]](https://www.zhihu.com/equation?tex=W_o)以及![[公式]](https://www.zhihu.com/equation?tex=b_1)、![[公式]](https://www.zhihu.com/equation?tex=b_2)求偏导，并不断调整它们以使![[公式]](https://www.zhihu.com/equation?tex=L)尽可能达到最小的过程。(注意此处把上文中的![[公式]](https://www.zhihu.com/equation?tex=U)、![[公式]](https://www.zhihu.com/equation?tex=W)、![[公式]](https://www.zhihu.com/equation?tex=V)参数相应替换为![[公式]](https://www.zhihu.com/equation?tex=W_x)、![[公式]](https://www.zhihu.com/equation?tex=W_s)、![[公式]](https://www.zhihu.com/equation?tex=W_o)，以保持和引用的文章一致。)

现在假设我们的时间序列只有三段，![[公式]](https://www.zhihu.com/equation?tex=t1)，![[公式]](https://www.zhihu.com/equation?tex=t2)，![[公式]](https://www.zhihu.com/equation?tex=t3)。

我们只对t3时刻的![[公式]](https://www.zhihu.com/equation?tex=W_x)、![[公式]](https://www.zhihu.com/equation?tex=W_s)、![[公式]](https://www.zhihu.com/equation?tex=W_o)求偏导（其他时刻类似）：

![[公式]](https://www.zhihu.com/equation?tex=%5Cbegin%7Bsplit%7D+%5Cfrac+%7B%5Cpartial+L_3%7D%7B%5Cpartial+W_o%7D+%26%3D+%5Cfrac+%7B%5Cpartial+L_3%7D%7B%5Cpartial+O_3%7D%5Cfrac+%7B%5Cpartial+O_3%7D%7B%5Cpartial+W_o%7D+%5C%5C+%5Cfrac+%7B%5Cpartial+L_3%7D%7B%5Cpartial+W_x%7D+%26%3D+%5Cfrac+%7B%5Cpartial+L_3%7D%7B%5Cpartial+O_3%7D%5Cfrac+%7B%5Cpartial+O_3%7D%7B%5Cpartial+S_3%7D%5Cfrac+%7B%5Cpartial+S_3%7D%7B%5Cpartial+W_x%7D+%2B+%5Cfrac+%7B%5Cpartial+L_3%7D%7B%5Cpartial+O_3%7D%5Cfrac+%7B%5Cpartial+O_3%7D%7B%5Cpartial+S_3%7D%5Cfrac+%7B%5Cpartial+S_3%7D%7B%5Cpartial+S_2%7D%5Cfrac+%7B%5Cpartial+S_2%7D%7B%5Cpartial+W_x%7D+%2B+%5Cfrac+%7B%5Cpartial+L_3%7D%7B%5Cpartial+O_3%7D%5Cfrac+%7B%5Cpartial+O_3%7D%7B%5Cpartial+S_3%7D%5Cfrac+%7B%5Cpartial+S_3%7D%7B%5Cpartial+S_2%7D%5Cfrac+%7B%5Cpartial+S_2%7D%7B%5Cpartial+S_1%7D%5Cfrac+%7B%5Cpartial+S_1%7D%7B%5Cpartial+W_x%7D+%5C%5C+%5Cfrac+%7B%5Cpartial+L_3%7D%7B%5Cpartial+W_s%7D+%26%3D+%5Cfrac+%7B%5Cpartial+L_3%7D%7B%5Cpartial+O_3%7D%5Cfrac+%7B%5Cpartial+O_3%7D%7B%5Cpartial+S_3%7D%5Cfrac+%7B%5Cpartial+S_3%7D%7B%5Cpartial+W_s%7D+%2B+%5Cfrac+%7B%5Cpartial+L_3%7D%7B%5Cpartial+O_3%7D%5Cfrac+%7B%5Cpartial+O_3%7D%7B%5Cpartial+S_3%7D%5Cfrac+%7B%5Cpartial+S_3%7D%7B%5Cpartial+S_2%7D%5Cfrac+%7B%5Cpartial+S_2%7D%7B%5Cpartial+W_s%7D+%2B+%5Cfrac+%7B%5Cpartial+L_3%7D%7B%5Cpartial+O_3%7D%5Cfrac+%7B%5Cpartial+O_3%7D%7B%5Cpartial+S_3%7D%5Cfrac+%7B%5Cpartial+S_3%7D%7B%5Cpartial+S_2%7D%5Cfrac+%7B%5Cpartial+S_2%7D%7B%5Cpartial+S_1%7D%5Cfrac+%7B%5Cpartial+S_1%7D%7B%5Cpartial+W_s%7D+%5Cend%7Bsplit%7D+%5C%5C)

可以看出损失函数对于![[公式]](https://www.zhihu.com/equation?tex=W_o)并没有长期依赖，但是因为![[公式]](https://www.zhihu.com/equation?tex=s_t)着时间序列向前传播，而![[公式]](https://www.zhihu.com/equation?tex=s_t)又是![[公式]](https://www.zhihu.com/equation?tex=W_x)、![[公式]](https://www.zhihu.com/equation?tex=W_s)的函数，所以对于![[公式]](https://www.zhihu.com/equation?tex=W_x)、![[公式]](https://www.zhihu.com/equation?tex=W_s)，会随着时间序列产生长期依赖。

根据上述求偏导的过程，我们可以得出任意时刻对![[公式]](https://www.zhihu.com/equation?tex=W_x)、![[公式]](https://www.zhihu.com/equation?tex=W_s)求偏导的公式：

![[公式]](https://www.zhihu.com/equation?tex=%5Cfrac+%7B%5Cpartial+L_t%7D%7B%5Cpartial+W_x%7D+%3D+%5Csum_%7Bk%3D0%7D%5Et%5Cfrac+%7B%5Cpartial+L_t%7D%7B%5Cpartial+O_t%7D%5Cfrac+%7B%5Cpartial+O_t%7D%7B%5Cpartial+S_t%7D%28%5Cprod_%7Bj%3Dk%2B1%7D%5Et%5Cfrac+%7B%5Cpartial+S_j%7D%7B%5Cpartial+S_%7Bj-1%7D%7D%29%5Cfrac+%7B%5Cpartial+S_k%7D%7B%5Cpartial+W_x%7D+%5C%5C)

任意时刻对![[公式]](https://www.zhihu.com/equation?tex=W_s)求偏导的公式同上。

如果加上激活函数（tanh），

![[公式]](https://www.zhihu.com/equation?tex=S_j+%3D+tanh%28W_xX_j+%2B+W_sS_%7Bj-1%7D+%2B+b_1%29+%5C%5C)

则

![[公式]](https://www.zhihu.com/equation?tex=%5Cprod_%7Bj%3Dk%2B1%7D%5Et%5Cfrac+%7B%5Cpartial+S_j%7D%7B%5Cpartial+S_%7Bj-1%7D%7D+%3D+%5Cprod_%7Bj%3Dk%2B1%7D%5Ettanh%27W_s+%5C%5C)

激活函数tanh的定义如下：

![[公式]](https://www.zhihu.com/equation?tex=tanhx+%3D+%5Cfrac+%7Bsinhx%7D%7Bcoshx%7D+%3D+%5Cfrac+%7Be%5Ex+-+e%5E%7B-x%7D%7D%7Be%5Ex+%2B+e%5E%7B-x%7D%7D+%5C%5C)

下图所示为tanh及其导数图像：

![](https://pic1.zhimg.com/80/v2-e6c34709aac8d1f313e210a30fc24cd4_1440w.jpg)

由上图可以看出![[公式]](https://www.zhihu.com/equation?tex=tanh%27+%5Cle+1)，在绝大部分训练过程中tanh的导数是小于1的，因为很少情况下会恰好出现![[公式]](https://www.zhihu.com/equation?tex=W_xX_j+%2B+W_sS_%7Bj-1%7D+%2B+b_1+%3D+0)。如果![[公式]](https://www.zhihu.com/equation?tex=W_s)也是一个大于0小于1的值，则当![[公式]](https://www.zhihu.com/equation?tex=t)很大时，![[公式]](https://www.zhihu.com/equation?tex=%5Cprod_%7Bj%3Dk%2B1%7D%5Ettanh%27W_s)就会趋近于0。同理当![[公式]](https://www.zhihu.com/equation?tex=W_s)很大时![[公式]](https://www.zhihu.com/equation?tex=%5Cprod_%7Bj%3Dk%2B1%7D%5Ettanh%27W_s)就会趋近于无穷，这就是RNN中梯度消失和爆炸的原因。

在实际问题中，梯度消失是怎样的现象？例如在使用RNN的语言模型预测下一个单词时，无法考虑到过去许多时间步骤的信息。如下面句子：

*Jane walked into the room. John walked in too. It was late in the day. Jane said hi to ____.*

为什么梯度消失是个问题？简单说，当我们看到![[公式]](https://www.zhihu.com/equation?tex=%5Cprod_%7Bj%3Dk%2B1%7D%5Et%5Cfrac+%7B%5Cpartial+S_j%7D%7B%5Cpartial+S_%7Bj-1%7D%7D)趋近于0时，我们不能判断是时刻![[公式]](https://www.zhihu.com/equation?tex=t)和![[公式]](https://www.zhihu.com/equation?tex=t%2Bn)之间没有依赖关系，还是参数的错误配置导致的。

对于梯度爆炸，Mikolov首先提出的解决方案是裁剪梯度，使它们的范数具有一定的最大值。

![](https://pic3.zhimg.com/80/v2-5cec8d01346d0165c18c3ca19ddbf872_1440w.jpg)

至于怎么避免这些现象，再看看![[公式]](https://www.zhihu.com/equation?tex=%5Cfrac+%7B%5Cpartial+L_t%7D%7B%5Cpartial+W_x%7D)。

![](https://pic2.zhimg.com/80/v2-093ac279323a3c7652ca542505a7dcb5_1440w.jpg)

梯度消失和爆炸的根本原因就是![[公式]](https://www.zhihu.com/equation?tex=%5Cprod_%7Bj%3Dk%2B1%7D%5Et%5Cfrac+%7B%5Cpartial+S_j%7D%7B%5Cpartial+S_%7Bj-1%7D%7D)这一部分，要消除这种情况就需要把这一部分在求偏导的过程中去掉，至于怎么去掉，一种办法就是使![[公式]](https://www.zhihu.com/equation?tex=%5Cfrac+%7B%5Cpartial+S_j%7D%7B%5Cpartial+S_%7Bj-1%7D%7D+%5Capprox+1)，另一种办法就是使![[公式]](https://www.zhihu.com/equation?tex=%5Cfrac+%7B%5Cpartial+S_j%7D%7B%5Cpartial+S_%7Bj-1%7D%7D+%5Capprox+0)。其实这就是LSTM做的事情。

## **2. LSTM和GRU的原理**

长短期记忆(LSTM)模型可用来解决稳定性和梯度消失的问题。在这个模型中，常规的神经元被存储单元代替。存储单元中管理向单元移除或添加的结构叫门限，有三种：遗忘门、输入门、输出门，门限由Sigmoid激活函数和逐点乘法运算组成。前一个时间步长的隐藏状态被送到遗忘门、输入门和输出门。在前向计算过程中，输入门学习何时激活让当前输入传入存储单元，而输出门学习何时激活让当前隐藏层状态传出存储单元。单个LSTM神经元的具体结构如图所示：

![](https://pic4.zhimg.com/80/v2-989a52d931de16f6483c403a328211eb_1440w.jpg)

我们假设![[公式]](https://www.zhihu.com/equation?tex=h)为LSTM单元的隐藏层输出，![[公式]](https://www.zhihu.com/equation?tex=c)为LSTM内存单元的值，![[公式]](https://www.zhihu.com/equation?tex=x)为输入数据。LSTM单元的更新与前向传播一样，可以分为以下几个步骤。

1. 首先，我们先计算当前时刻的输入结点![[公式]](https://www.zhihu.com/equation?tex=g_%7B%28t%29%7D)，![[公式]](https://www.zhihu.com/equation?tex=W_%7Bxg%7D)，![[公式]](https://www.zhihu.com/equation?tex=W_%7Bhg%7D)，![[公式]](https://www.zhihu.com/equation?tex=W_%7Bcg%7D)分别是输入数据和上一时刻LSTM 单元输出的权值：

![[公式]](https://www.zhihu.com/equation?tex=%5Cbegin%7Bsplit%7D+1.+%5Calpha_g%5Et+%26%3D+W_%7Bxg%7D%5ETx_%7B%28t%29%7D+%2B+W_%7Bhg%7D%5ETh_%7B%28t-1%29%7D+%2B+b_g+%5C%5C+2.+g_%7B%28t%29%7D+%26%3D+%5Csigma%28%5Calpha_g%5Et%29+%5Cend%7Bsplit%7D+%5C%5C)

1. 计算输入门 (input gate) 的值![[公式]](https://www.zhihu.com/equation?tex=i_%7B%28t%29%7D)。输入门用来控制当前输入数据对记忆单元状态值的影响。所有门的计算受当前输入数据![[公式]](https://www.zhihu.com/equation?tex=x_%7B%28t%29%7D)和上一时刻LSTM单元输出值![[公式]](https://www.zhihu.com/equation?tex=h_%7B%28t-1%29%7D)影响。

![[公式]](https://www.zhihu.com/equation?tex=%5Cbegin%7Bsplit%7D+1.+%5Calpha_i%5Et+%26%3D+W_%7Bxi%7D%5ETx_%7B%28t%29%7D+%2B+W_%7Bhi%7D%5ETh_%7B%28t-1%29%7D+%2B+b_i+%5C%5C+2.+i_%7B%28t%29%7D+%26%3D+%5Csigma%28%5Calpha_i%5Et%29+%5Cend%7Bsplit%7D+%5C%5C)

1. 计算遗忘门的值![[公式]](https://www.zhihu.com/equation?tex=f_%7B%28t%29%7D)。遗忘门主要用来控制历史信息对当前记忆单元状态值的影响，为记忆单元提供了重置的方式。

![[公式]](https://www.zhihu.com/equation?tex=%5Cbegin%7Bsplit%7D+1.+%5Calpha_f%5Et+%26%3D+W_%7Bxf%7D%5ETx_%7B%28t%29%7D+%2B+W_%7Bhf%7D%5ETh_%7B%28t-1%29%7D+%2B+b_f+%5C%5C+2.+f_%7B%28t%29%7D+%26%3D+%5Csigma%28%5Calpha_f%5Et%29+%5Cend%7Bsplit%7D+%5C%5C)

1. 计算当前时刻记忆单元的状态值![[公式]](https://www.zhihu.com/equation?tex=c_%7B%28t%29%7D)。记忆单元是整个LSTM神经元的核心结点。记忆单元的状态更新主要由自身状态![[公式]](https://www.zhihu.com/equation?tex=c_%7B%28t-1%29%7D)和当前时刻的输入结点的值![[公式]](https://www.zhihu.com/equation?tex=g_%7B%28t%29%7D)，并且利用乘法门通过输入门和遗忘门分别对这两部分因素进行调节。乘法门的目的是使 LSTM存储单元存储和访问时间较长的信息，从而减轻消失的梯度。

![[公式]](https://www.zhihu.com/equation?tex=1.+c_%7B%28t%29%7D+%3D+f_%7B%28t%29%7D+%5Cotimes+c_%7B%28t-1%29%7D+%2B+i_%7B%28t%29%7D+%5Cotimes+g_%7B%28t%29%7D+%5C%5C)

1. 计算输出门![[公式]](https://www.zhihu.com/equation?tex=o_%7B%28t%29%7D)。输出门用来控制记忆单元状态值的输出。

![[公式]](https://www.zhihu.com/equation?tex=%5Cbegin%7Bsplit%7D+1.+%5Calpha_o%5Et+%26%3D+W_%7Bxo%7D%5ETx_%7B%28t%29%7D+%2B+W_%7Bho%7D%5ETh_%7B%28t-1%29%7D+%2B+b_o+%5C%5C+2.+o_%7B%28t%29%7D+%26%3D+%5Csigma%28%5Calpha_o%5Et%29+%5Cend%7Bsplit%7D+%5C%5C)

1. 最后计算LSTM单元的输出。

![[公式]](https://www.zhihu.com/equation?tex=1.+h_%7B%28t%29%7D+%3D+o_%7B%28t%29%7D+%5Cotimes+tanh%28c_%7B%28t%29%7D%29+%5C%5C)

由Gers和Schmidhuber在2000年引入的一种流行的LSTM变体是添加“窥视孔连接”。这意味着下面情况的改变。

![](https://pic4.zhimg.com/80/v2-4fa045c83a1117bbb7f51bae91e8ad4b_1440w.jpg)

LSTM稍微有点戏剧性的变化是由Cho等人引入的 **门控循环单元(Gated Recurrent Unit, GRU)** ，它将遗忘和输入门组合成一个“更新门”。它还合并了单元状态和隐藏状态，并进行了一些其他更改。由此产生的模型比标准LSTM模型简单，并且越来越受欢迎。

![](https://pic3.zhimg.com/80/v2-f987c977995327b305381c1ec67232ce_1440w.jpg)

## **3. LSTM如何解决解决梯度消失问题？**

LSTM可以抽象成这样：

![](https://pic1.zhimg.com/80/v2-d41ecf07cce4595834c673cffb4ba914_1440w.jpg)

三个圆圈内为乘号的符号分别代表的就是forget gate，input gate，output gate，而我认为LSTM最关键的就是forget gate这个部件。这三个gate是如何控制流入流出的呢？其实就是通过下面![[公式]](https://www.zhihu.com/equation?tex=f_t)，![[公式]](https://www.zhihu.com/equation?tex=i_t)，![[公式]](https://www.zhihu.com/equation?tex=o_t)三个函数来控制，因为 ![[公式]](https://www.zhihu.com/equation?tex=%5Csigma%28x%29)（代表sigmoid函数）的值是介于0到1之间的，刚好用趋近于0时表示流入不能通过gate，趋近于1时表示流入可以通过gate。

![[公式]](https://www.zhihu.com/equation?tex=%5Cbegin%7Bsplit%7D+f_t+%26%3D+%5Csigma+%28W_fX_t+%2B+b_f%29+%5C%5C+i_t+%26%3D+%5Csigma+%28W_iX_t+%2B+b_i%29+%5C%5C+o_t+%26%3D+%5Csigma+%28W_oX_t+%2B+b_o%29+%5Cend%7Bsplit%7D+%5C%5C)

当前的状态![[公式]](https://www.zhihu.com/equation?tex=S_t+%3D+f_tS_%7Bt-1%7D%2Bi_tX_t)，将LSTM的状态表达式展开后得：

![[公式]](https://www.zhihu.com/equation?tex=S_t+%3D+%5Csigma+%28W_fX_t+%2B+b_f%29S_%7Bt-1%7D+%2B+%5Csigma+%28W_iX_t+%2B+b_i%29X_t+%5C%5C)

如果加上激活函数，

![[公式]](https://www.zhihu.com/equation?tex=S_t+%3D+tanh%28%5Csigma+%28W_fX_t+%2B+b_f%29S_%7Bt-1%7D+%2B+%5Csigma+%28W_iX_t+%2B+b_i%29X_t%29+%5C%5C)

传统RNN求偏导的过程包含 ![[公式]](https://www.zhihu.com/equation?tex=%5Cprod_%7Bj%3Dk%2B1%7D%5Et%5Cfrac+%7B%5Cpartial+S_j%7D%7B%5Cpartial+S_%7Bj-1%7D%7D%3D%5Cprod_%7Bj%3Dk%2B1%7D%5Et+tanh%27W_s)，对于LSTM同样也包含这样的一项，但是在LSTM中，

![[公式]](https://www.zhihu.com/equation?tex=%5Cprod_%7Bj%3Dk%2B1%7D%5Et%5Cfrac+%7B%5Cpartial+S_j%7D%7B%5Cpartial+S_%7Bj-1%7D%7D%3D%5Cprod_%7Bj%3Dk%2B1%7D%5Et+tanh%27%5Csigma+%28W_fX_t+%2B+b_f%29+%5C%5C)

假设![[公式]](https://www.zhihu.com/equation?tex=Z+%3D+tanh%27%28x%29%5Csigma+%28y%29)，则 Z 的函数图像如下图所示：

![](https://pic3.zhimg.com/80/v2-467cc646ff63d7887a2476d842d7d552_1440w.jpg)

可以看到该函数值基本上不是0就是1。再看RNN求偏导过程中有：

![[公式]](https://www.zhihu.com/equation?tex=%5Cfrac+%7B%5Cpartial+L_3%7D%7B%5Cpartial+W_s%7D+%3D+%5Cfrac+%7B%5Cpartial+L_3%7D%7B%5Cpartial+O_3%7D%5Cfrac+%7B%5Cpartial+O_3%7D%7B%5Cpartial+S_3%7D%5Cfrac+%7B%5Cpartial+S_3%7D%7B%5Cpartial+W_s%7D+%2B+%5Cfrac+%7B%5Cpartial+L_3%7D%7B%5Cpartial+O_3%7D%5Cfrac+%7B%5Cpartial+O_3%7D%7B%5Cpartial+S_3%7D%5Cfrac+%7B%5Cpartial+S_3%7D%7B%5Cpartial+S_2%7D%5Cfrac+%7B%5Cpartial+S_2%7D%7B%5Cpartial+W_s%7D+%2B+%5Cfrac+%7B%5Cpartial+L_3%7D%7B%5Cpartial+O_3%7D%5Cfrac+%7B%5Cpartial+O_3%7D%7B%5Cpartial+S_3%7D%5Cfrac+%7B%5Cpartial+S_3%7D%7B%5Cpartial+S_2%7D%5Cfrac+%7B%5Cpartial+S_2%7D%7B%5Cpartial+S_1%7D%5Cfrac+%7B%5Cpartial+S_1%7D%7B%5Cpartial+W_s%7D+%5C%5C)

如果在LSTM中，![[公式]](https://www.zhihu.com/equation?tex=%5Cfrac+%7B%5Cpartial+L_3%7D%7B%5Cpartial+W_s%7D)可能就会变成：

![[公式]](https://www.zhihu.com/equation?tex=%5Cfrac+%7B%5Cpartial+L_3%7D%7B%5Cpartial+W_s%7D+%3D+%5Cfrac+%7B%5Cpartial+L_3%7D%7B%5Cpartial+O_3%7D%5Cfrac+%7B%5Cpartial+O_3%7D%7B%5Cpartial+S_3%7D%5Cfrac+%7B%5Cpartial+S_3%7D%7B%5Cpartial+W_s%7D+%2B+%5Cfrac+%7B%5Cpartial+L_3%7D%7B%5Cpartial+O_3%7D%5Cfrac+%7B%5Cpartial+O_3%7D%7B%5Cpartial+S_3%7D%5Cfrac+%7B%5Cpartial+S_2%7D%7B%5Cpartial+W_s%7D+%2B+%5Cfrac+%7B%5Cpartial+L_3%7D%7B%5Cpartial+O_3%7D%5Cfrac+%7B%5Cpartial+O_3%7D%7B%5Cpartial+S_3%7D%5Cfrac+%7B%5Cpartial+S_3%7D%7B%5Cpartial+S_2%7D%5Cfrac+%7B%5Cpartial+S_1%7D%7B%5Cpartial+W_s%7D+%5C%5C)

这是因为：

![[公式]](https://www.zhihu.com/equation?tex=%5Cprod_%7Bj%3Dk%2B1%7D%5Et%5Cfrac+%7B%5Cpartial+S_j%7D%7B%5Cpartial+S_%7Bj-1%7D%7D%3D%5Cprod_%7Bj%3Dk%2B1%7D%5Et+tanh%27%5Csigma+%28W_fX_t+%2B+b_f%29+%5Capprox+0%7C1+%5C%5C)

这样就解决了传统RNN中梯度消失的问题。

## **四、RNN的应用领域**

RNN在NLP的许多任务上取得巨大成功。主要的应用领域有：

* 语言建模与生成文本（本文示例重点介绍）

  * 给定一系列单词，我们想要预测给定下一单词的概率
  * 能够预测下一个单词的副作用是我们得到一个生成模型，它允许我们通过从输出概率中抽样来生成新文本
  * 应用模式many to one
* 机器翻译

  * 机器翻译类似于语言建模，因为我们的输入是源语言中的一系列单词（例如德语），我们希望以目标语言输出一系列单词（例如英语）
  * 应用模式many to many
* 语音识别
* 给定来自声波的声学信号的输入序列，我们可以预测一系列语音片段及其概率
* 应用模式many to many
* 生成图像描述

  * 与卷积神经网络一起，RNN已被用作模型的一部分，以生成未标记图像的描述
  * 应用模式one to many

![](https://pic4.zhimg.com/80/v2-9c2e1df9ac1b05021e356d680070362f_1440w.jpg)

## **五、RNN实战示例**

## **a. 语言建模**

 **语言建模(Language Modeling)** ，通常指实现预测下一个单词的任务。例如下图，输入了"the students opened their"这个未完成的句子，预测下个单词最有可能的是哪一个？

![](https://pic2.zhimg.com/80/v2-3f19df77872ff4b332525335fbc423d5_1440w.jpg)

更为形式的描述为：给定一个单词序列![[公式]](https://www.zhihu.com/equation?tex=x%5E%7B%281%29%7D%2Cx%5E%7B%282%29%7D%2C...%2Cx%5E%7B%28t%29%7D)，计算下一个单词![[公式]](https://www.zhihu.com/equation?tex=x%5E%7B%28t%2B1%29%7D)的概率分布：

![[公式]](https://www.zhihu.com/equation?tex=P%28x%5E%7B%28t%2B1%29%7D+%3D+%5Comega_j%7Cx%5E%7B%28t%29%7D%2C...%2Cx%5E%7B%281%29%7D%29+%5C%5C)

这里![[公式]](https://www.zhihu.com/equation?tex=%5Comega_j)是在词汇表![[公式]](https://www.zhihu.com/equation?tex=V%3D%5C%7B%5Comega_1%2C...%2C%5Comega_%7B%7CV%7C%7D%5C%7D)里的任一单词。

或者从句子的角度看，我们预测观察句子（在给定数据集中）的概率为：

![[公式]](https://www.zhihu.com/equation?tex=P%28%5Comega_1%2C...%2C%5Comega_m%29+%3D+%5Cprod_%7Bi%3D1%7D%5EmP%28%5Comega_i%7C%5Comega_1%2C...%2C%5Comega_%7Bi-1%7D%29+%5C%5C)

对语言来说，句子的概率是每个单词概率的乘积。因此，判断“他去买一些巧克力”这句话的可能性是“巧克力”在给出“他去买一些”的概率，乘以“一些”在给出“他去买”的概率，依次等等。

## **b. 语言建模的应用**

日常生活中天天会用到这个模型。例如在手机输入法中：

![](https://pic3.zhimg.com/80/v2-7ca50cb964bb7323e0dd4cc47c906b26_1440w.jpg)

## **c. n-gram语言模型**

* 问：怎样学习一个语言模型？
* 答：（深度学习前）：使用n-gram语言模型
* 定义：n-gram是由n个连续单词组成的块。

  * unigrams: “the”, “students”, “opened”, ”their”
  * bigrams: “the students”, “students opened”, “opened their”
  * trigrams: “the students opened”, “students opened their”
  * **4-**grams: “the students opened their”
* 思想：收集关于不同n-grams频率的统计信息，用这些来预测下一个单词。
* 具体实现参考文献[5]。

## **d. RNN语言模型源代码**

参考： **[rnn-tutorial-rnnlm](https://link.zhihu.com/?target=https%3A//github.com/dennybritz/rnn-tutorial-rnnlm)**

其中有：**[RNNLM.ipynb](https://link.zhihu.com/?target=https%3A//github.com/dennybritz/rnn-tutorial-rnnlm/blob/master/RNNLM.ipynb)** 可以演示具体的运行过程。

## **六、参考文献**

1. The Unreasonable Effectiveness of Recurrent Neural Networks
2. Recurrent Neural Networks Tutorial, Part 1 – Introduction to RNNs
3. Recurrent Neural Networks Tutorial, Part 2 – Implementing a RNN with Python, Numpy and Theano
4. Recurrent Neural Networks Tutorial, Part 3 – Backpropagation Through Time and Vanishing Gradients
5. Recurrent Neural Networks and Language Models
6. On the difficulty of training recurrent neural networks
7. Understanding LSTM Networks
8. 张尧. 激活函数导向的RNN算法优化[D].浙江大学,2017.
9. 高茂庭,徐彬源.基于循环神经网络的推荐算法[J/OL].计算机工程:1-7[2018-11-30]
10. 成烯，钟波. 基于LSTM神经网络的股价短期预测模型[EB/OL]. 北京：中国科技论文在线 [2018-04-04].
11. Text generation using a RNN with eager execution
12. 新型RNN——将层内神经元相互独立以提高长程记忆
13. RNN梯度消失和爆炸的原因
14. LSTM如何解决梯度消失问题

**好了，本篇RNN入门就介绍到这里了。谢谢大家，希望大家多提宝贵意见，并持续关注我们的公众号。**

![](https://pic4.zhimg.com/80/v2-7050680063b1d9a6fa58e6840183856f_1440w.jpg)

微信文章链接：

[循环神经网络(RNN)知识入门**mp.weixin.qq.com/s?__biz=MzI0MDQ0NzA1Nw==&amp;mid=2247484560&amp;idx=1&amp;sn=b4be3b1c7b4b77c61594e65eadc92423&amp;chksm=e91bffaede6c76b83d2c0afa44a18f2b3e23952842511104cdb44b648af14983e5a9a0461f7d&amp;scene=21#wechat_redirect](https://link.zhihu.com/?target=https%3A//mp.weixin.qq.com/s%3F__biz%3DMzI0MDQ0NzA1Nw%3D%3D%26mid%3D2247484560%26idx%3D1%26sn%3Db4be3b1c7b4b77c61594e65eadc92423%26chksm%3De91bffaede6c76b83d2c0afa44a18f2b3e23952842511104cdb44b648af14983e5a9a0461f7d%26scene%3D21%23wechat_redirect)

# 史上最详细循环神经网络讲解（RNN/LSTM/GRU）

---

## **一。什么是循环神经网络：**

循环神经网络（Rerrent Neural Network, RNN），历史啊，谁发明的都不重要，说了你也记不住，你只要记住RNN是神经网络的一种，类似的还有深度神经网络DNN，卷积神经网络CNN，生成对抗网络GAN，等等。另外你需要记住RNN的特点， **RNN对具有序列特性的数据非常有效，它能挖掘数据中的时序信息以及语义信息，** 利用了RNN的这种能力，使深度学习模型在解决语音识别、语言模型、机器翻译以及时序分析等NLP领域的问题时有所突破。

我们需要重点来了解一下RNN的特点这句话，什么是**序列特性**呢？我个人理解，就是 **符合时间顺序，逻辑顺序，或者其他顺序就叫序列特性** ，举几个例子：

* 拿人类的某句话来说，也就是人类的自然语言，是不是符合某个逻辑或规则的字词拼凑排列起来的，这就是符合序列特性。
* 语音，我们发出的声音，每一帧每一帧的衔接起来，才凑成了我们听到的话，这也具有序列特性、
* 股票，随着时间的推移，会产生具有顺序的一系列数字，这些数字也是具有序列特性。

## **二。为什么要发明循环神经网络：**

我们先来看一个NLP很常见的问题，命名实体识别，举个例子，现在有两句话：

第一句话：I like eating apple！（我喜欢吃苹果！）

第二句话：The Apple is a great company！（苹果真是一家很棒的公司！）

现在的任务是要给apple打Label，我们都知道第一个apple是一种水果，第二个apple是苹果公司，假设我们现在有大量的已经标记好的数据以供训练模型，当我们使用全连接的神经网络时，我们做法是把apple这个单词的特征向量输入到我们的模型中（如下图），在输出结果时，让我们的label里，正确的label概率最大，来训练模型，但我们的语料库中，有的apple的label是水果，有的label是公司，这将导致，模型在训练的过程中，预测的准确程度，取决于训练集中哪个label多一些，这样的模型对于我们来说完全没有作用。**问题就出在了我们没有结合上下文去训练模型，而是单独的在训练apple这个单词的label，这也是全连接神经网络模型所不能做到的，于是就有了我们的循环神经网络。**

![](https://pic3.zhimg.com/80/v2-9a86430ba17aa299ce5c44c7b75c5ece_1440w.jpg)
（全连接神经网络结构）

## **三。循环神经网络的结构及原理：**

![](https://pic1.zhimg.com/80/v2-8f534b5db1f3d8a5c4ccd029be4a15b4_1440w.jpg)
（RNN结构）

上图就是RNN的结构，我第一次看到这图的第一反应是，不是说好的循环神经网络么，起码得是神经网络啊，神经网络不是有很多球球么，也就是神经元，这RNN咋就这几个球球，不科学啊，看不懂啊！！！！随着慢慢的了解RNN，才发现这图看着是真的清楚，因为RNN的特殊性，如果展开画成那种很多神经元的神经网络，会很麻烦。

我们先来讲解一下上面这幅图，首先不要管右边的W，只看X,U,S,V,O，这幅图就变成了，如下：

![](https://pic3.zhimg.com/80/v2-9a86430ba17aa299ce5c44c7b75c5ece_1440w.jpg)

等等，这图看着有点眼熟啊，这不就是全连接神经网络结构吗？对，没错，不看W的话，上面那幅图展开就是全连接神经网络，其中X是一个向量，也就是某个字或词的特征向量，作为输入层，如上图也就是3维向量，U是输入层到隐藏层的参数矩阵，在上图中其维度就是3X4，S是隐藏层的向量，如上图维度就是4，V是隐藏层到输出层的参数矩阵，在上图中就是4X2，O是输出层的向量，在上图中维度为2。有没有一种顿时豁然开朗的感觉，正是因为我当初在学习的时候，可能大家都觉得这个问题比较小，所以没人讲，我一直搞不清楚那些神经元去哪了。。所以我觉得讲出来，让一些跟我一样的小白可以更好的理解。

弄懂了RNN结构的左边，那么右边这个W到底是什么啊？把上面那幅图打开之后，是这样的：

![]()

等等，这又是什么？？别慌，很容易看，举个例子，有一句话是，I love you，那么在利用RNN做一些事情时，比如命名实体识别，上图中的 ![[公式]](https://www.zhihu.com/equation?tex=X_%7Bt-1%7D) 代表的就是I这个单词的向量， ![[公式]](https://www.zhihu.com/equation?tex=X) 代表的是love这个单词的向量， ![[公式]](https://www.zhihu.com/equation?tex=X_%7Bt%2B1%7D) 代表的是you这个单词的向量，以此类推，我们注意到，上图展开后，W一直没有变， **W其实是每个时间点之间的权重矩阵** ，我们注意到，RNN之所以可以解决序列问题， **是因为它可以记住每一时刻的信息，每一时刻的隐藏层不仅由该时刻的输入层决定，还由上一时刻的隐藏层决定** ，公式如下，其中 ![[公式]](https://www.zhihu.com/equation?tex=O_t) 代表t时刻的输出, ![[公式]](https://www.zhihu.com/equation?tex=S_t) 代表t时刻的隐藏层的值：

![]()

**值得注意的一点是，在整个训练过程中，每一时刻所用的都是同样的W。**

## **四。举个例子，方便理解：**

假设现在我们已经训练好了一个RNN，如图，我们假设每个单词的特征向量是二维的，也就是输入层的维度是二维，且隐藏层也假设是二维，输出也假设是二维，所有权重的值都为1且没有偏差且所有激活函数都是线性函数，现在输入一个序列，到该模型中，我们来一步步求解出输出序列：

![]()

![]()

你可能会好奇W去哪了？W在实际的计算中，在图像中表示非常困难 ，所以我们可以想象上一时刻的隐藏层的值是被存起来，等下一时刻的隐藏层进来时，上一时刻的隐藏层的值通过与权重相乘，两者相加便得到了下一时刻真正的隐藏层，如图 ![[公式]](https://www.zhihu.com/equation?tex=a_1) , ![[公式]](https://www.zhihu.com/equation?tex=a_2) 可以看做每一时刻存下来的值，当然初始时![[公式]](https://www.zhihu.com/equation?tex=a_1) , ![[公式]](https://www.zhihu.com/equation?tex=a_2)是没有存值的，因此初始值为0：

![]()

当我们输入第一个序列，【1,1】，如下图，其中隐藏层的值，也就是绿色神经元，是通过公式 ![[公式]](https://www.zhihu.com/equation?tex=S_%7Bt%7D%3Df%5Cleft%28U+%5Ccdot+X_%7Bt%7D%2BW+%5Ccdot+S_%7Bt-1%7D%5Cright%29) 计算得到的，因为所有权重都是1，所以也就是 ![[公式]](https://www.zhihu.com/equation?tex=1%2A1%2B1%2A1%2B1%2A0%2B1%2A0%3D2) （我把向量X拆开计算的，由于篇幅关系，我只详细列了其中一个神经元的计算过程，希望大家可以看懂，看不懂的请留言），输出层的值4是通过公式 ![[公式]](https://www.zhihu.com/equation?tex=O_%7Bt%7D%3Dg%5Cleft%28V+%5Ccdot+S_%7Bt%7D%5Cright%29) 计算得到的，也就是 ![[公式]](https://www.zhihu.com/equation?tex=2%2A1%2B2%2A1%3D4) （同上，也是只举例其中一个神经元），得到输出向量【4,4】：

当【1,1】输入过后，我们的记忆里的 ![[公式]](https://www.zhihu.com/equation?tex=a_1%2Ca_2) 已经不是0了，而是把这一时刻的隐藏状态放在里面，即变成了2，如图，输入下一个向量【1,1】，隐藏层的值通过公式![[公式]](https://www.zhihu.com/equation?tex=S_%7Bt%7D%3Df%5Cleft%28U+%5Ccdot+X_%7Bt%7D%2BW+%5Ccdot+S_%7Bt-1%7D%5Cright%29) 得到， ![[公式]](https://www.zhihu.com/equation?tex=1%2A1%2B1%2A1%2B1%2A2%2B1%2A2%3D6) ，输出层的值通过公式![[公式]](https://www.zhihu.com/equation?tex=O_%7Bt%7D%3Dg%5Cleft%28V+%5Ccdot+S_%7Bt%7D%5Cright%29)，得到 ![[公式]](https://www.zhihu.com/equation?tex=6%2A1%2B6%2A1%3D12) ，最终得到输出向量【12,12】：

![]()

同理，该时刻过后 ![[公式]](https://www.zhihu.com/equation?tex=a_1%2Ca_2) 的值变成了6，也就是输入第二个【1,1】过后所存下来的值，同理，输入第三个向量【2,2】，如图，细节过程不再描述，得到输出向量【32,32】：

![]()

由此，我们得到了最终的输出序列为：

![]()

至此，一个完整的RNN结构我们已经经历了一遍，我们注意到，每一时刻的输出结果都与上一时刻的输入有着非常大的关系，如果我们将输入序列换个顺序，那么我们得到的结果也将是截然不同，这就是RNN的特性，可以处理序列数据，同时对序列也很敏感。

## 五。什么是LSTM：

如果你经过上面的文章看懂了RNN的内部原理，那么LSTM对你来说就很简单了，首先大概介绍一下LSTM，是四个单词的缩写，Long short-term memory，翻译过来就是长短期记忆，是RNN的一种，比普通RNN高级（上面讲的那种），基本一般情况下说使用RNN都是使用LSTM，现在很少有人使用上面讲的那个最基础版的RNN，因为那个存在一些问题，LSTM效果好，当然会选择它了！

## **六。为什么LSTM比普通RNN效果好？**

这里就牵扯到梯度消失和爆炸的问题了，我简单说两句，上面那个最基础版本的RNN，我们可以看到，每一时刻的隐藏状态都不仅由该时刻的输入决定，还取决于上一时刻的隐藏层的值，如果一个句子很长，到句子末尾时，它将记不住这个句子的开头的内容详细内容，具体原因可以看我之前写的文章，如下：

[](https://zhuanlan.zhihu.com/p/76772734)

LSTM通过它的“门控装置”有效的缓解了这个问题，这也就是为什么我们现在都在使用LSTM而非普通RNN。

## 七。揭开LSTM神秘的面纱：

既然前面已经说了，LSTM是RNN的一种变体，更高级的RNN，那么它的本质还是一样的，还记得RNN的特点吗， **可以有效的处理序列数据，** 当然LSTM也可以，还记得RNN是如何处理有效数据的吗，是不是 **每个时刻都会把隐藏层的值存下来，到下一时刻的时候再拿出来用，这样就保证了，每一时刻含有上一时刻的信息** ，如图，我们把存每一时刻信息的地方叫做Memory Cell，中文就是记忆细胞，可以这么理解。

![]()

打个比喻吧，普通RNN就像一个乞丐，路边捡的，别人丢的，什么东西他都想要，什么东西他都不嫌弃，LSTM就像一个贵族，没有身份的东西他不要，他会精心挑选符合自己身份的物品。这是为什么呢？有没有思考过，原因很简单，乞丐没有选择权，他的能力注定他只能当一个乞丐，因此他没有挑选的权利，而贵族不一样，贵族能力比较强，经过自己的打拼，终于有了地位和身份，所以可以选择舍弃一些低档的东西，这也是能力的凸显。

 **LSTM和普通RNN正是贵族和乞丐，RNN什么信息它都存下来，因为它没有挑选的能力，而LSTM不一样，它会选择性的存储信息，因为它能力强，它有门控装置，它可以尽情的选择。** 如下图，普通RNN只有中间的Memory Cell用来存所有的信息，而从下图我们可以看到，LSTM多了三个Gate，也就是三个门，什么意思呢？在现实生活中，门就是用来控制进出的，门关上了，你就进不去房子了，门打开你就能进去，同理，这里的门是用来控制每一时刻信息记忆与遗忘的。

**依次来解释一下这三个门：**

1. Input Gate：中文是输入门，在每一时刻从输入层输入的信息会首先经过输入门，输入门的开关会决定这一时刻是否会有信息输入到Memory Cell。
2. Output Gate：中文是输出门，每一时刻是否有信息从Memory Cell输出取决于这一道门。
3. Forget Gate：中文是遗忘门，每一时刻Memory Cell里的值都会经历一个是否被遗忘的过程，就是由该门控制的，如果打卡，那么将会把Memory Cell里的值清除，也就是遗忘掉。

**按照上图的顺序，信息在传递的顺序，是这样的：**

先经过输入门，看是否有信息输入，再判断遗忘门是否选择遗忘Memory Cell里的信息，最后再经过输出门，判断是否将这一时刻的信息进行输出。

## 八。LSTM内部结构：

抱歉最近事比较多，没有及时更新。。让我们先回顾一下之前讲了点啥，关于LSTM，我们了解了它的能力比普通RNN要强，因为它可以对输入的信息，选择性的记录或遗忘，这是因为它拥有强大的门控系统，分别是记忆门，遗忘门，和输出门，至于这三个门到底是如何工作的，如何起作用的。本节我们就来详细讲解LSTM的内部结构。

在了解LSTM的内部结构之前，我们需要先回顾一下普通RNN的结构，以免在这里很多读者被搞懵，如下：

![]()

我们可以看到，左边是为了简便描述RNN的工作原理而画的缩略图，右边是展开之后，每个时间点之间的流程图，**注意，我们接下来看到的LSTM的结构图，是一个时间点上的内部结构，就是整个工作流程中的其中一个时间点，也就是如下图：**

![]()

注意，上图是普通RNN的一个时间点的内部结构，上面已经讲过了公式和原理，LSTM的内部结构更为复杂，不过如果这么类比来学习，我认为也没有那么难。

![]()

我们类比着来学习，首先看图中最中间的地方，Cell，我们上面也讲到了memory cell，也就是一个记忆存储的地方，这里就类似于普通RNN的 ![[公式]](https://www.zhihu.com/equation?tex=S_t) ，都是用来存储信息的，这里面的信息都会保存到下一时刻，其实标准的叫法应该是 ![[公式]](https://www.zhihu.com/equation?tex=h_t) ，因为这里对应神经网络里的隐藏层，所以是hidden的缩写，无论普通RNN还是LSTM其实t时刻的记忆细胞里存的信息，都应该被称为 ![[公式]](https://www.zhihu.com/equation?tex=h_t) 。再看最上面的 ![[公式]](https://www.zhihu.com/equation?tex=a) ，是这一时刻的输出，也就是类似于普通RNN里的 ![[公式]](https://www.zhihu.com/equation?tex=O_t) 。最后，我们再来看这四个 ![[公式]](https://www.zhihu.com/equation?tex=Z%EF%BC%8CZ_i%EF%BC%8CZ_f%EF%BC%8CZ_o) ，这四个相辅相成，才造就了中间的Memory Cell里的值，你肯恩要问普通RNN里有个 ![[公式]](https://www.zhihu.com/equation?tex=X_t+) 作为输入，那LSTM的输入在哪？别着急，其实这四个 ![[公式]](https://www.zhihu.com/equation?tex=Z%EF%BC%8CZ_i%EF%BC%8CZ_f%EF%BC%8CZ_o) 都有输入向量 ![[公式]](https://www.zhihu.com/equation?tex=X_t) 的参与。对了，在解释这四个分别是什么之前，我要先解释一下上图的所有这个符号，都代表一个激活函数，LSTM里常用的激活函数有两个，一个是tanh，一个是sigmoid。

![[公式]](https://www.zhihu.com/equation?tex=Z%3Dtanh%28W%5Bx_t%2Ch_%7Bt-1%7D%5D%29%5C%5CZ_i%3D%5Csigma%28W_i%5Bx_t%2Ch_%7Bt-1%7D%5D%29%5C%5CZ_f%3D%5Csigma%28W_f%5Bx_t%2Ch_%7Bt-1%7D%5D%29%5C%5CZ_o%3D%5Csigma%28W_o%5Bx_t%2Ch_%7Bt-1%7D%5D%29%5C%5C)

其中 ![[公式]](https://www.zhihu.com/equation?tex=Z) 是最为普通的输入，可以从上图中看到， ![[公式]](https://www.zhihu.com/equation?tex=Z) 是通过该时刻的输入 ![[公式]](https://www.zhihu.com/equation?tex=X_t) 和上一时刻存在memory cell里的隐藏层信息 ![[公式]](https://www.zhihu.com/equation?tex=h_%7Bt-1%7D) 向量拼接，再与权重参数向量 ![[公式]](https://www.zhihu.com/equation?tex=W) 点积，得到的值经过激活函数tanh最终会得到一个数值，也就是 ![[公式]](https://www.zhihu.com/equation?tex=Z) ，注意只有 ![[公式]](https://www.zhihu.com/equation?tex=Z) 的激活函数是tanh，因为 ![[公式]](https://www.zhihu.com/equation?tex=Z) 是真正作为输入的，其他三个都是门控装置。

再来看 ![[公式]](https://www.zhihu.com/equation?tex=Z_i) ，input gate的缩写i，所以也就是输入门的门控装置， ![[公式]](https://www.zhihu.com/equation?tex=Z_i+) 同样也是通过该时刻的输入 ![[公式]](https://www.zhihu.com/equation?tex=X_t) 和上一时刻隐藏状态，也就是上一时刻存下来的信息 ![[公式]](https://www.zhihu.com/equation?tex=h_%7Bt-1%7D) 向量拼接，在与权重参数向量 ![[公式]](https://www.zhihu.com/equation?tex=W_i) 点积（注意每个门的权重向量都不一样，这里的下标i代表input的意思，也就是输入门）。得到的值经过激活函数sigmoid的最终会得到一个0-1之间的一个数值，用来作为输入门的控制信号。

以此类推，就不详细讲解 ![[公式]](https://www.zhihu.com/equation?tex=Z_f%EF%BC%8CZ_o) 了，分别是缩写forget和output的门控装置，原理与上述输入门的门控装置类似。

上面说了，只有 ![[公式]](https://www.zhihu.com/equation?tex=Z) 是输入，其他的三个都是门控装置，负责把控每一阶段的信息记录与遗忘，具体是怎样的呢？我们先来看公式：

首先解释一下，经过这个sigmod激活函数后，得到的 ![[公式]](https://www.zhihu.com/equation?tex=Z_i%EF%BC%8CZ_f%EF%BC%8CZ_o) 都是在0到1之间的数值，1表示该门完全打开，0表示该门完全关闭.
