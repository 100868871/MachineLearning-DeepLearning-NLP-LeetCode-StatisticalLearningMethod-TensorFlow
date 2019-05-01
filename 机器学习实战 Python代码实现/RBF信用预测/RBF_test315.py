"""
Created on Sat Mar  9 22:02:25 2019

@author: liweimin
"""

#import turtle
#import random
#import time
#import jieba
#import numpy as np
#import matplotlib.pyplot as plt
#import pandas as pd
#import sklearn
#import seaborn as sns 
#import tensorflow as tf 
#
#
#A=np.array([[56,0.0,4.4,68.0],
#           [1.2,104,52,8],
#           [1.8,135,99,0.9]])
#
#print(A)
#
#
#cal=A.sum(axis=0)
#print(cal)
#
#per=100*A/cal
#print(per)




from scipy import *
from scipy.linalg import norm, pinv
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cross_validation import train_test_split
import pandas as pd

class RBF:
    def __init__(self, indim, numCenters, outdim): 
        self.indim = indim
        self.outdim = outdim
        self.numCenters = numCenters
        self.centers = [random.uniform(-1, 1, indim) for i in range(numCenters)]
        self.beta = 8
        self.W = random.random((self.numCenters, self.outdim))

    def _basisfunc(self, c, d):
        assert len(d) == self.indim
        return exp(-self.beta * norm(c-d)**2)

    def _calcAct(self, X):
 # calculate activations of RBFs
        G = zeros((X.shape[0], self.numCenters), float)
        for ci, c in enumerate(self.centers):
            for xi, x in enumerate(X):
                G[xi,ci] = self._basisfunc(c, x)
        return G

    def train(self, X, Y):
        """ X: matrix of dimensions n x indim
        y: column vector of dimension n x 1 """
 # choose random center vectors from training set
        rnd_idx = random.permutation(X.shape[0])[:self.numCenters]
        self.centers = [X[i,:] for i in rnd_idx]
        print("center", self.centers)
 # calculate activations of RBFs
        G = self._calcAct(X)
        print(G)
 # calculate output weights (pseudoinverse)
        self.W = dot(pinv(G), Y)

    def test(self, X):
        """ X: matrix of dimensions n x indim """
        rnd_idx = random.permutation(X.shape[0])[:self.numCenters]
        self.centers = [X[i,:] for i in rnd_idx]
        G = self._calcAct(X)
        Y = dot(G, self.W)
        return Y
#x = np.array([[0,0],[0,1],[1,0],[1,1]])
#y = np.array([[0],[1],[1],[0]])
data = pd.read_table('german.data-numeric', header=None, sep='\s+')
data = data.as_matrix()
x = data[:,0:24]
y = data[:,-1:]
#对信用好坏进行 0 和 1 处理
y[y==1]=0
y[y==2]=1
#对 x 的取值进行归一化
#x = (x - np.mean(x,axis=0))/np.std(x,axis=0)
x = (x - np.min(x,axis=0))/(np.max(x,axis=0) - np.min(x,axis=0))
#拆分训练数据与测试数据
#x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3)
innums,indim = x_train.shape
#centernum = 24
centernum = 36
outnums, outdim = y_train.shape
rbf = RBF(indim, centernum, outdim)
rbf.train(x_train, y_train)
result = rbf.test(x_test)
#结果和输出 y_test 对比
result.astype(int)
count = 0
for i in range(len(y_test)):
    if y_test[i] == result.astype(int)[i]:
        count += 1
print('accuracy rate is ', count*100/len(y_test),'%')














#a=np.random.rand(1000000)
#b=np.random.rand(1000000)
#
#start_time=time.perf_counter()
#
#c=np.dot(a,b)
#
#end_time=time.perf_counter()
#
#time1=end_time-start_time
#
#print(time1)
#print(c)
#
#
#start_time2=time.perf_counter()
#
#
#c=0
#for i in range(1000000):
#    c+=a[i]*b[i]
#
#end_time2=time.perf_counter()
#
#time2=end_time2-start_time2
#
#print(time2)
#print(c)













#






    



       
    
    

