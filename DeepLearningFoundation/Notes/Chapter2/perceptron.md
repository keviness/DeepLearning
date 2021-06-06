## 感知机（perceptron）
![perceptron](../imgs/perceptron.png)
### （一）定义
* 输入信号被送往神经元时，会被分别乘以固定的权重（w1x1、w2x2）。神经元会计算传送过来的信号的总和，只有当这个总和超过了某个界限值时，才会输出1。这也称为“神经元被激活”。这里将这个界限值称为阈值，用符号θ表示。
* 数学公式：
![perceptron2](../imgs/perceptron2.png)