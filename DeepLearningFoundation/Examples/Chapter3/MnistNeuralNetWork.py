#--- Mnist classficate by Neural Net Work ---
import numpy as np
import pickle
from dataset.mnist import load_mnist
#import sys, os
#sys.path.append(os.pardir)

def Loadata():
    (trainSet, trainLable), (testSet, testLabel) = load_mnist(flatten=True, normalize=True, one_hot_label = False)
    return testSet, testLabel

def InitNetWork(weightPath):
    with open(weightPath, 'rb') as f:
        network = pickle.load(f)
        f.close()
    #print("weight:\n", weight)
    #w1, w2, w3 = network['W1'], network['W2'], network['W3']
    #b1, b2, b3 = network['b1'], network['b2'], network['b3']
    return network

def sigmoid(x):
    return 1/(1+np.exp(-x))

def softmax(x):
    max = np.max(x)
    ex = np.exp(x - max)
    return ex/np.sum(ex)

def predict(data, network):
    w1, w2, w3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    #print("w1:\n",w1.shape, "\nw2:\n",w2.shape, "\nw3:\n",w3.shape)
    z1 = np.dot(data, w1) + b1
    r1 = sigmoid(z1)
    z2 = np.dot(r1, w2)
    r2 = sigmoid(z2)
    z3 = np.dot(r2, w3)
    r3 = softmax(z3)
    #print(r3.shape)
    return r3


if __name__ == "__main__":
    weightPath = "/Users/kevin/Desktop/program files/DeepLearning/DeepLearningFoundation/Examples/Chapter3/dataset/sample_weight.pkl"
    testSet, testLabel = Loadata() 

    network = InitNetWork(weightPath)
    #--Batch Handle--
    batchSize = 100
    accuracyCount = 0
    for i in range(0, testSet.shape[0], batchSize):
        xBatch = testSet[i:i+batchSize]
        yBatch = predict(xBatch, network)
        preRe = np.argmax(yBatch, axis=1)
        accuracyCount += np.sum(preRe == testLabel[i:i+batchSize])
    print("accuracy:\n", accuracyCount/testSet.shape[0])

    #-- Loop Handle --
    '''
    result = predict(testSet, network)
    accuracyCount = 0
    for i in range(result.shape[0]):
        preRe = np.argmax(result[i])
        if preRe == testLabel[i]:
            accuracyCount += 1
    accuracy = accuracyCount/result.shape[0]
    print("accuracy:\n", accuracy)
    '''

