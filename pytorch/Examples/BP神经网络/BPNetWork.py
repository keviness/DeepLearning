import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from torch.autograd import Variable
from torch.nn import *
from torch.optim import Adam
import pandas as pd

# 超参数定义(由于我们的隐藏层只有一层，所以可以直接定义为超参数)
batch_size=100
input_feature=400
hidden_feature=23
output_feature=500
learning_rate=1e-3
epochs=1000
loss_f=MSELoss()


# 参数初始化
x=Variable(torch.randn(batch_size,input_feature),requires_grad=False)
y=Variable(torch.randn(batch_size,output_feature),requires_grad=False)
w1=Variable(torch.randn(input_feature,hidden_feature),requires_grad=True)
w2=Variable(torch.randn(hidden_feature,output_feature),requires_grad=True)

print('x:\n', x)
print('y:\n', y)
#print('w1:\n', y)
#print('w1:\n', y)


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.hidden = nn.Linear(input_feature,hidden_feature)
        self.act = nn.ReLU()
        self.output = nn.Linear(hidden_feature,output_feature)
    def forward(self, x):
        a = self.act(self.hidden(x))
        return self.output(a)


Epoch=[]
Loss=[]
'''
model=Sequential(
    Linear(input_feature,hidden_feature),
    Linear(hidden_feature,output_feature)
)
'''
# optimizer需要传入训练参数和lr
model = MLP()
optim=Adam(model.parameters(),lr=learning_rate)
print(model)


# 迭代训练
for epoch in tqdm.tqdm(range(1,epochs+1)):
    # 前向传播
    y_pred=model(x)
    loss=loss_f(y_pred,y)

    Epoch.append(epoch)
    Loss.append(loss.data)

    if epoch%50==0:
        print("Epoch:{},loss:{}".format(epoch,loss))
    optim.zero_grad()
    # 后向传播
    loss.backward()
    # 参数微调
    optim.step()

d = {}
i=0
for parm in model.parameters(): 
    d[str(i)] = parm.data
    print(f'{parm.names}---{parm.data}')
    i+=1
print('d[1]:\n', d['0'].T.numpy().shape)
dataFrame = pd.DataFrame(data=d['0'].T.numpy())
print('dataFrame:\n', dataFrame)
#dataFrame.to_excel()
Epoch=np.array(Epoch)
Loss=np.array(Loss)
plt.plot(Epoch,Loss)
plt.show()