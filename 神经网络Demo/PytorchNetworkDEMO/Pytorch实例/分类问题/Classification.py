import torch
import numpy as np
from torch.autograd import Variable, variable
import torch.nn.functional as F
import matplotlib.pyplot as plt

cachePath = "/Users/kevin/Desktop/program files/DeepLearning/神经网络Demo/PytorchNetworkDEMO/Pytorch实例/分类问题/model/"

n_data = torch.ones(100,2)
x0 = torch.normal(2*n_data,1)
y0 = torch.zeros(100)
#print("n_data:\n", n_data.shape)
#print("x0:\n", x0)
#print("y0:\n", y0)

x1 = torch.normal(-2*n_data,1)
y1 = torch.ones(100)

x = torch.cat((x0,x1),0).type(torch.FloatTensor)
y = torch.cat((y0,y1)).type(torch.LongTensor)
x,y = Variable(x),Variable(y)
#print('x:\n', x)
#print('y:\n', y)

plt.scatter(x[:,0], x[:,1], c=y.data.numpy(), s=100, lw=0, cmap='RdYlGn')
plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=y.data.numpy(), s=100, lw=0, cmap='RdYlGn')
plt.show()

class Net(torch.nn.Module):
    def __init__(self,n_input,n_hidden,n_output):
        super(Net,self).__init__()
        self.hidden1 = torch.nn.Linear(n_input,n_hidden)
        self.hidden2 = torch.nn.Linear(n_hidden,n_hidden)
        self.predict = torch.nn.Linear(n_hidden,n_output)

    def forward(self, input):
        out = self.hidden1(input)
        out = F.sigmoid(out)
        out = self.hidden2(out)
        out = F.sigmoid(out)
        out = self.predict(out)
        out = F.softmax(out)
        return out

net = Net(2,20,2)
#print(net)

optimizer = torch.optim.SGD(net.parameters(),lr=0.02)
loss_func = torch.nn.CrossEntropyLoss()

plt.ion()
plt.show()

# train model
'''
acc = 0.0
for t in range(500):
    out = net(x)
    # print(prediction)
    loss = loss_func(out,y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    prediction = torch.max(out,1)[1]
    pred_y = prediction.data.numpy().squeeze()
    target_y = y.data.numpy()
    accuracy = sum(pred_y == target_y) / 200.  # 预测中有多少和真实值一样
    if accuracy > acc:
        torch.save(net, cachePath+'net.pkl')
        acc = accuracy
        
    if t%2==0:
        plt.cla()
        # 过了一道 softmax 的激励函数后的最大概率才是预测值
        # print(F.softmax(prediction))
        plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=pred_y, s=100, lw=0, cmap='RdYlGn')
        plt.text(1.5, -4, 'Accuracy=%.2f' % accuracy, fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.1)

plt.ioff()  # 停止画图
plt.show()
'''
# predict
net = torch.load(cachePath+'net.pkl')
test_data = torch.ones(80,2)
test1 = torch.normal(2*test_data, 1)
test1_label = torch.zeros(80)
test2 = torch.normal(-2*test_data, 1)
test2_label = torch.ones(80)
test = torch.cat((test1, test2))
label = torch.cat((test1_label, test2_label))
test = Variable(test)
label = Variable(label)
print('test:\n', test)
print('labels:\n', label)
Out = net(test)
prediction = torch.max(Out,1)[1]
pred_y = prediction.data.numpy().squeeze()
target_y = label.data.numpy()
accuracy = sum(pred_y == target_y) / 160
print('accuracy:\n', accuracy)