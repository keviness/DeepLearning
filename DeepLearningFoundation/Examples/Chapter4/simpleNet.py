import numpy as np
from common.functions import *
from common.gradient import numerical_gradient

class simpleNet(object):
    def __init__(self):
        self.W = np.random.randn(2, 3)

    def predict(self, x):
        z = np.dot(x, self.W)
        y = softmax(z)
        return y
    
    def loss(self, x, t):
        y = self.predict(x)
        loss = cross_entropy_error(y, t)
        return loss

    def numDiff(self, x, t):
        function = lambda W:self.loss(x, t)
        df = numerical_gradient(function, self.W)
        return df

if __name__ == "__main__":
    net = simpleNet()
    print("net.W:\n", net.W)
    prediction = net.predict(np.array([0.6, 0.9]))
    print("prediction:\n", prediction)
    xSet = np.array([0.6, 0.9])
    tSet = np.array([0, 0, 1])
    loss = net.loss(xSet, tSet)
    print("loss:\n", loss)

    function = lambda W : net.loss(xSet, tSet)
    
    df = numerical_gradient(function, net.W)
    print("df:\n", df)