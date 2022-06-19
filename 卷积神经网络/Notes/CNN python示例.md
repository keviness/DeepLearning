# 使用 Python 实现的卷积神经网络初学者指南

## CNN简介

卷积神经网络是一种专为处理图像和视频而设计的深度学习算法。它以图像为输入，提取和学习图像的特征，并根据学习到的特征进行分类。

该算法的灵感来自于人脑的一部分，即视觉皮层。视觉皮层是人脑的一部分，负责处理来自外界的视觉信息。它有不同的层，每一层都有自己的功能，即每一层从图像或任何视觉中提取一些信息，最后将从每一层接收到的所有信息组合起来，对图像/视觉进行解释或分类。

同样，CNN有各种滤波器，每个滤波器从图像中提取一些信息，例如边缘、不同种类的形状（垂直、水平、圆形），然后将所有这些组合起来识别图像。

现在，这里的问题可能是：为什么我们不能将人工神经网络用于相同的目的？这是因为ANN有一些缺点：

* 对于 ANN 模型来说，训练大尺寸图像和不同类型的图像通道的计算量太大。
* 它无法从图像中捕获所有信息，而 CNN 模型可以捕获图像的空间依赖性。
* 另一个原因是人工神经网络对图像中物体的位置很敏感，即如果同一物体的位置或地点发生变化，它将无法正确分类。

## CNN的组成部分

CNN模型分两步工作：**特征提取和分类**

**特征提取**是将各种过滤器和图层应用于图像以从中提取信息和特征的阶段，完成后将传递到下一阶段，即 **分类** ，根据问题的目标变量对它们进行分类。

**典型的 CNN 模型如下所示：**

* 输入层
* 卷积层+激活函数
* 池化层
* 全连接层

![](https://pic4.zhimg.com/80/v2-b23367c6f659a7652969ec453c1429a7_1440w.jpg)

来源：[https://**learnopencv.com/image-c**lassification-using-convolutional-neural-networks-in-keras/](https://link.zhihu.com/?target=https%3A//learnopencv.com/image-classification-using-convolutional-neural-networks-in-keras/)

让我们详细了解每一层。

## 输入层

顾名思义，它是我们的输入图像，可以是灰度或 RGB。每个图像由范围从 0 到 255 的像素组成。我们需要对它们进行归一化，即在将其传递给模型之前转换 0 到 1 之间的范围。

下面是大小为 4*4 的输入图像的示例，它有 3 个通道，即 RGB 和像素值。

![](https://pic2.zhimg.com/80/v2-57fc11bff0fdf05d8cbaa8204ea432f5_1440w.jpg)

来源：[https://**medium.com/@raycad.seed**otech/convolutional-neural-network-cnn-8d1908c010ab](https://link.zhihu.com/?target=https%3A//medium.com/%40raycad.seedotech/convolutional-neural-network-cnn-8d1908c010ab)

## 卷积层

卷积层是将**过滤器应用于我们的输入图像**以提取或检测其特征的层。过滤器多次应用于图像并创建一个有助于对输入图像进行分类的特征图。让我们借助一个例子来理解这一点。为简单起见，我们将采用具有归一化像素的 2D 输入图像。

![](https://pic2.zhimg.com/80/v2-065aba85f84f972a737f091a7517fc81_1440w.jpg)

在上图中，我们有一个大小为 66 的输入图像，并对其应用了 33 的过滤器来检测一些特征。在这个例子中，我们只应用了一个过滤器，但在实践中，许多这样的过滤器被用于从图像中提取信息。

将过滤器应用于图像的结果是我们得到一个 4*4 的特征图，其中包含有关输入图像的一些信息。许多这样的特征图是在实际应用中生成的。

让我们深入了解获取上图中特征图的一些数学原理。

![](https://pic2.zhimg.com/80/v2-d57b8831794f3c22c2dfd865276b7199_1440w.jpg)

如上图所示，**第一步**过滤器应用于图像的绿色高亮部分，将图像的像素值与过滤器的值相乘（如图中使用线条所示），然后相加得到最终值。

 **在下一步中，过滤器将移动一列** ，如下图所示。这种跳转到下一列或行的过程称为  **stride** ，在本例中，我们将 stride设为1，这意味着我们将移动一列。

![](https://pic3.zhimg.com/80/v2-11efca37f43b74a006b56a276e90e20a_1440w.jpg)

类似地，过滤器通过整个图像，我们得到最终的 **特征图** 。一旦我们获得特征图，就会对其应用激活函数来引入非线性。

这里需要注意的一点是，我们得到的特征图小于我们图像的大小。随着我们增加 stride 的值，特征图的大小会减小。

![](https://pic2.zhimg.com/80/v2-55e2c4ffd5bf2dfffe0f26abfdbe0c8d_1440w.jpg)

**这就是过滤器如何以 1 的步幅穿过整个图像**

## 池化层

池化层应用在卷积层之后，用于降低特征图的维度，有助于保留输入图像的重要信息或特征，并减少计算时间。

> 使用池化，可以创建一个较低分辨率的输入版本，该版本仍然包含输入图像的大元素或重要元素。

最常见的池化类型是最大池化和平均池化。

下图显示了最大池化的工作原理。使用我们从上面的例子中得到的特征图来应用池化。这里我们使用了一个大小为 2*2的池化层，步长为 2。

取每个突出显示区域的最大值，并获得大小为 2*2的新版本输入图像，因此在应用池化后，特征图的维数减少了。

![](https://pic2.zhimg.com/80/v2-9d7e785c59f00d30972e2d90debd05e5_1440w.jpg)

## 全连接层

到目前为止，我们已经执行了特征提取步骤，现在是分类部分。全连接层（如我们在 ANN 中所使用的）用于将输入图像分类为标签。该层将从前面的步骤（即卷积层和池化层）中提取的信息连接到输出层，并最终将输入分类为所需的标签。

CNN 模型的完整过程可以在下图中看到。

![](https://pic3.zhimg.com/80/v2-91b42bdaddcdddeb3290cc15b5ca7342_1440w.jpg)

来源：[https://**developersbreach.com/co**nvolution-neural-network-deep-learning/](https://link.zhihu.com/?target=https%3A//developersbreach.com/convolution-neural-network-deep-learning/)

## CNN在 Python 中的实现

我们将使用 Mnist Digit 分类数据集，我们在ANN的实际实现的上一篇博客中使用了该数据集。为了更好地理解CNN的应用，请先参考上一篇博客：[https://www.**analyticsvidhya.com/blo**g/2021/08/implementing-artificial-neural-network-on-unstructured-data/](https://link.zhihu.com/?target=https%3A//www.analyticsvidhya.com/blog/2021/08/implementing-artificial-neural-network-on-unstructured-data/)

```python3
#importing the required libraries
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense

#loading data
(X_train,y_train) , (X_test,y_test)=mnist.load_data()
#reshaping data
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2], 1))
X_test = X_test.reshape((X_test.shape[0],X_test.shape[1],X_test.shape[2],1)) 
#checking the shape after reshaping
print(X_train.shape)
print(X_test.shape)
#normalizing the pixel values
X_train=X_train/255
X_test=X_test/255

#defining model
model=Sequential()
#adding convolution layer
model.add(Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1)))
#adding pooling layer
model.add(MaxPool2D(2,2))
#adding fully connected layer
model.add(Flatten())
model.add(Dense(100,activation='relu'))
#adding output layer
model.add(Dense(10,activation='softmax'))
#compiling the model
model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
#fitting the model
model.fit(X_train,y_train,epochs=10)
```

**输出：**

![](https://pic1.zhimg.com/80/v2-40326964d08ac5045f6293f43c3ce994_1440w.jpg)

```python3
#evaluting the model
model.evaluate(X_test,y_test)
```

![](https://pic1.zhimg.com/80/v2-666ba8905e51d03c18e33a27face4e20_1440w.jpg)

希望这篇文章对你有所帮助。
