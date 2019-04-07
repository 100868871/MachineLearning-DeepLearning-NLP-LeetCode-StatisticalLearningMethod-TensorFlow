# encoding: utf-8
#author :'liweimin'
#2019-3-26 17:09:11

import numpy as np
import matplotlib.pyplot as plt
import itertools as it
import matplotlib as mpl
from matplotlib import colors
from sklearn import datasets
from sklearn import model_selection
import random
import pandas
from sklearn.linear_model import LogisticRegression #线性模型中的逻辑回归  用于对比参照
from sklearn.metrics import confusion_matrix #计算混淆矩阵，主要来评估分类的准确性
from sklearn.metrics import accuracy_score #计算精度得分
from sklearn.metrics import classification_report  #将主要分类指标以文本输出

#定义Sigmoid曲线
def Sigmoid(x):  
    return 1.0 / (1.0 + np.exp(-x))  

# 逻辑回归算法
def LogReg(datas,labels):
    kinds = list(set(labels))  # 3个类别
    means=datas.mean(axis=0) #属性的均值
    stds=datas.std(axis=0) #标准差

    N,M= datas.shape[0],datas.shape[1]+1  #N样本数，M参数向量的维
    K=3 #类别数

    data=np.ones((N,M))
    data[:,1:]=(datas-means)/stds #对原始数据进行标准差归一化

    W=np.zeros((K-1,M))  #存储参数矩阵
    priorEs=np.array([1.0/N*np.sum(data[labels==kinds[i]],axis=0) for i in range(K-1)]) #属性的先验期望值

    liklist=[]
    for it in range(1000):
        lik=0 #当前的对数似然函数值
        for k in range(K-1): #似然函数值的第一部分
            lik -= np.sum(np.dot(W[k],data[labels==kinds[k]].transpose()))
        lik +=1.0/N *np.sum(np.log(np.sum(np.exp(np.dot(W,data.transpose())),axis=0)+1)) #似然函数的第二部分
        liklist.append(lik)

        wx=np.exp(np.dot(W,data.transpose()))
        probs=np.divide(wx,1+np.sum(wx,axis=0).transpose()) # K-1 *N的矩阵
        posteriorEs=1.0/N*np.dot(probs,data) #各个属性的后验期望值
        gradients=posteriorEs - priorEs +1.0/100 *W #梯度，最后一项是高斯项，防止过拟合
        W -= gradients #对参数进行修正
    print("输出W为：",W)
    return W

#根据训练得到的参数W和数据集，进行预测。输入参数为数据集和由LogReg算法得到的参数W，返回值为预测的值
def iris_predict(datas,W):
    N, M = datas.shape[0], datas.shape[1] + 1  # N是样本数，M是参数向量的维
    K = 3  # k=3是类别数
    data = np.ones((N, M))
    means = datas.mean(axis=0)  # 各个属性的均值
    stds = datas.std(axis=0)  # 各个属性的标准差
    data[:, 1:] = (datas - means) / stds  # 对原始数据进行标准差归一化

    # probM每行三个元素，分别表示data中对应样本被判给三个类别的概率
    probM = np.ones((N, K))
    print("data.shape:", data.shape)
    print("datas.shape:", datas.shape)
    print("W.shape:", W.shape)
    print("probM.shape:", probM.shape)
    probM[:, :-1] = np.exp(np.dot(data, W.transpose()))
    probM /= np.array([np.sum(probM, axis=1)]).transpose()  # 得到概率

    predict = np.argmax(probM, axis=1).astype(int)  # 取最大概率对应的类别
    print("输出predict为：", predict)
    return predict


if __name__ == '__main__':
    
    attributes=['sepal_length','sepal_width','petal_length','petal_width'] #鸢尾花属性名

    datas=[]
    labels=[]
    datas_test=[]
    labels_test=[]

with open('IRIS.txt', 'r') as f_iris:
    lines = f_iris.readlines()

with open('train.txt', 'w') as ftr, open('test.txt', 'w') as fte:
    for i in range(105):   #70%进行训练，30%进行测试
        ftr.write(lines.pop(random.randint(0, len(lines) - 1)))
    fte.writelines(lines)
data_file=open('train.txt','r')
data_file_test=open('test.txt','r')

for line in data_file.readlines():
    linedata = line.split(',')
    datas.append(linedata[:-1])  # 选取前4列是4个属性的值
    labels.append(linedata[-1].replace('\n', ''))  # 最后一列是类别
for line in data_file_test.readlines():
    linedata = line.split(',')
    datas_test.append(linedata[:-1])  # 选取前4列是4个属性的值
    labels_test.append(linedata[-1].replace('\n', ''))  # 最后一列是类别

datas=np.array(datas)
datas_test=np.array(datas_test)
datas=datas.astype(float) #将二维的字符串数组转换成浮点数数组
datas_test=datas_test.astype(float)
labels=np.array(labels)
labels_test=np.array(labels_test)
kinds=list(set(labels)) #3个类别的名字
kinds_test=list(set(labels_test))

#通过LogReg算法得到参数 W
W=LogReg(datas,labels)

#通过iris_predict（）函数进行预测
predict=iris_predict(datas_test,W)

# rights列表储存原始标签数据的序号，根据labels数据生成
N = datas_test.shape[0]
rights = np.zeros(N)
rights[labels_test == kinds_test[1]] = 1
rights[labels_test == kinds_test[2]] = 2
rights = rights.astype(int)


err=np.sum(predict != rights)
right=45-err
print("预测正确个数为：%d\n" % right)
print("预测错误个数为：%d\n" % err)
acc=(45-err)/45
print('logistic回归准确度为:%f' % acc)


#Python 自带的logistic算法(95%左右)，对比手写算法（98%左右）
data = "iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(data, names=names) #读取csv数据
 
array = dataset.values #将数据库转换成数组形式
X = array[:,0:4] #取前四列，即属性数值
Y = array[:,4] #取最后一列，种类
validation_size = 0.30 #验证集规模
seed = 2019
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed) #分割数据集 

seed = 2019
scoring = 'accuracy'

log = LogisticRegression()
log.fit(X_train, Y_train) #log拟合序列集
predictions = log.predict(X_validation) #预测验证集
print('log验证集精度得分:%f' % accuracy_score(Y_validation, predictions)) 
print('分类预测报告' +classification_report(Y_validation, predictions)) 


