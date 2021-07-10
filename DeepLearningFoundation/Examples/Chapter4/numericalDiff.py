# ---numerical differentiation---
import numpy as np
import matplotlib.pylab as plt

# ---numerical differentiation---
def numerical_diff(function, x):
    h = 1e-4
    y1 = function(x-h)
    y2 = function(x+h)
    return (y2 - y1)/(h*2)

def numericalDiffAdvance(function, x):
    h = 1e-4
    grad = np.zeros_like(x)
    for i in range(x.size):
        temp = x[i]
        x[i] = temp + h
        y1 = function(x)

        x[i] = temp - h
        y2 = function(x)

        grad[i] = (y1-y2)/(h*2)
        x[i] = temp

    return grad

# ---functions---
def function1(x):
    return 0.01 * x**2 + 0.1*x 

def function2(x):
    return np.sum(x**2)

def tangent_line(f, x):
    d = numerical_diff(f, x)
    print(d)
    y = f(x) - d*x
    return lambda t: d*t + y
    
# --- gradient ----
def gradient_dscent(function, init_x, learningrate=0.1, step_num=100):
    x = init_x
    result = []
    for i in range(step_num):
        result.append(x.copy())
        grad = numericalDiffAdvance(function, x)
        x -= learningrate * grad
    result = np.array(result)

    return result

# ---plot picture---
def plotPicture(x, y):
    #x = np.arange(0, 20, 0.1)
    #y = function1(x)
    plt.xlabel("x0")
    plt.ylabel("x1")
    plt.scatter(x, y)
    plt.show()

# ---test---
if __name__ == "__main__":
    #---numerf_ical_diff-----
    '''
    x = np.arange(0.0, 20.0, 0.1)
    y = function1(x)
    #plotPicture(x, y)
    x0 = 5
    y0 = function1(x0)

    diff = numerical_diff(function1, x0)
    print("diff:", diff, "\ny0:\n", y0)
    yLiner = diff * (x-x0) + y0

    #tf = tangent_line(function1, x0)
    #yLiner = tf(x)
    
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.plot(x, y, color='r')
    plt.plot(x, yLiner, color='b')
    plt.scatter(x0, y0, color='g')
    plt.show()
    '''
    #---numericalDiffAdvance----
    '''
    xTest = np.array([1.0, 3.0])
    yTest = function2(xTest)
    print("yTest:\n", yTest)
    grad = numericalDiffAdvance(function2, xTest)
    print("grad:\n", grad)
    '''
    test = np.array([-3.0, 4.0])
    result = gradient_dscent(function2, init_x=test, learningrate=0.1, step_num=100)
    print(result)
    plotPicture(result[:,0], result[:,1])