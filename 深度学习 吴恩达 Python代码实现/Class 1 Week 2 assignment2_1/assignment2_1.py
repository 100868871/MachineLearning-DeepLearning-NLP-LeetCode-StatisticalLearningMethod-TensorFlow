"""
Created on Sat Mar  9 22:02:25 2019

@author: liweimin
"""

import turtle
import random
import time
import jieba
import math
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
import seaborn as sns 
import tensorflow as tf 


def sigmoidm(x):
    s=1/(1+math.exp(-x))
    return s

def sigmoid(x):
    s=1/(1+np.exp(-x))
    return s

def sigmoid_derivative(x):
    s = 1 / (1 + np.exp(-x))
    ds=s*(1-s)
    return ds

def image2vector(image):
    """
    Argument:
    image -- a numpy array of shape (length, height, depth)

    Returns:
    v -- a vector of shape (length*height*depth, 1)
    """

    ### START CODE HERE ### (≈ 1 line of code)
    v = image.reshape((image.shape[0] * image.shape[1] * image.shape[2], 1))
    ### END CODE HERE ###

    return v

def normalizeRows(x):
    x_norm=np.linalg.norm(x,ord=2,axis=1,keepdims=True)
    x=x/x_norm
    return x


s = np.array([1, 2, 3])
print(sigmoid_derivative(s))


x=np.array([[0,3,4],
            [1,4,6]])   
print(normalizeRows(x))
    
def softmax(x):
    x_exp = np.exp(x)
    x_sum = np.sum(x_exp, axis=1, keepdims=True)
    s=x/x_sum
    return s
    
smax = np.array([[9, 2, 5, 0, 0],
                [7, 5, 0, 0 ,0]])
print(softmax(smax))
    


x1 = [9, 2, 5, 0, 0, 7, 5, 0, 0, 0, 9, 2, 5, 0, 0]
x2 = [9, 2, 2, 9, 0, 9, 2, 5, 0, 0, 9, 2, 5, 0, 0]
  

### CLASSIC DOT PRODUCT OF VECTORS IMPLEMENTATION ### 
start=time.perf_counter()
dot = 0
for i in range(len(x1)):
    dot+= x1[i]*x2[i]
    

end=time.perf_counter()  
    
    
a_time=end -start
print(dot)
print (1000*a_time)
    
### CLASSIC OUTER PRODUCT IMPLEMENTATION ###  
start=time.perf_counter()
outer = np.zeros((len(x1),len(x2))) 
for i in range(len(x1)):
    for j in range(len(x2)):
        outer[i,j] = x1[i]*x2[j]

end=time.perf_counter()  
a_time=end -start
print(outer)
print (1000*a_time)
    
### CLASSIC ELEMENTWISE IMPLEMENTATION ###
start=time.perf_counter()
mul = np.zeros(len(x1))
for i in range(len(x1)):
    mul[i] = x1[i]*x2[i]
end=time.perf_counter()  

end=time.perf_counter()  
a_time=end -start
print(mul)
print (1000*a_time)

## CLASSIC GENERAL DOT PRODUCT IMPLEMENTATION ###
W = np.random.rand(3,len(x1)) # Random 3*len(x1) numpy array
start=time.perf_counter()
gdot = np.zeros(W.shape[0])
for i in range(W.shape[0]):
    for j in range(len(x1)):
        gdot[i] += W[i,j]*x1[j]
end=time.perf_counter()  
a_time=end -start
print(gdot)
print (1000*a_time)


### VECTORIZED DOT PRODUCT OF VECTORS ###

start=time.perf_counter()
dot=np.dot(x1,x2)

end=time.perf_counter()  
a_time=end -start
print(dot)
print (1000*a_time)

### VECTORIZED OUTER PRODUCT ###

start=time.perf_counter()
outer=np.outer(x1,x2)

end=time.perf_counter()  
a_time=end -start
print(outer)
print (1000*a_time)

### VECTORIZED ELEMENTWISE MULTIPLICATION ###
start=time.perf_counter()
mul=np.multiply(x1,x2)

end=time.perf_counter()  
a_time=end -start
print(mul)
print (1000*a_time)

### VECTORIZED GENERAL DOT PRODUCT ###
start=time.perf_counter()
gdot=np.dot(W,x1)

end=time.perf_counter()  
a_time=end -start
print(gdot)
print (1000*a_time)


# GRADED FUNCTION: L1
def L1(y_hat,y):
    loss=np.sum(np.abs(y-y_hat))
    return loss

y_hat=np.array([.9,0.2,0.1,.4,.9])
y=np.array([1,0,0,1,1])
print(L1(y_hat,y))

# GRADED FUNCTION: L2

def L2(y_hat, y):
    loss=np.sum(np.dot(y-y_hat,y-y_hat))
    #这个需要仔细看作业的提示
    return loss
yhat = np.array([.9, 0.2, 0.1, .4, .9])
y = np.array([1, 0, 0, 1, 1])
print(L2(yhat,y))
    










