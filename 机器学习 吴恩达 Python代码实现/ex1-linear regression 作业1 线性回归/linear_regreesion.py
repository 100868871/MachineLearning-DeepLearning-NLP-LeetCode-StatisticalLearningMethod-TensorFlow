# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 20:53:12 2019

@author: liweimin
"""
# ex1_1linear regreesion当有推导好的公式后，有很多东西直接照着公式写代码就可以了，并不是很难
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

df = pd.read_csv('ex1data1.txt', names=['population', 'profit'])#读取数据并赋予列名
print(df.head())#查看前5行
print(df.info())#查看相关信息
print(df.describe())#查看数据基本特征
sns.lmplot('population', 'profit', df, size=6, fit_reg=False)
plt.show()#画出相关数据

def get_X(df):#读取x特征
    ones = pd.DataFrame({'ones': np.ones(len(df))})#ones是m行1列的dataframe
    data = pd.concat([ones, df], axis=1)  # 合并数据，根据列合并
    return data.iloc[:, :-1].as_matrix()  # 这个操作返回 ndarray,不是矩阵


def get_y(df):#读取y标签
#默认最后一列是目标值
    return np.array(df.iloc[:, -1])#df.iloc[:, -1]是指df的最后一列


def normalize_feature(df):
#沿DataFrame的输入轴(默认值为0)应用函数
    return df.apply(lambda column: (column - column.mean()) / column.std())#特征缩放

def linear_regression(X_data, y_data, alpha, epoch, optimizer=tf.train.GradientDescentOptimizer):# 这个函数是旧金山的一个大神Lucas Shen写的
      # 图输入占位符
    X = tf.placeholder(tf.float32, shape=X_data.shape)
    y = tf.placeholder(tf.float32, shape=y_data.shape)

    # 构建图
    with tf.variable_scope('linear-regression'):
        W = tf.get_variable("weights",
                            (X_data.shape[1], 1),
                            initializer=tf.constant_initializer())  # n*1

        y_pred = tf.matmul(X, W)  # m*n @ n*1 -> m*1

        loss = 1 / (2 * len(X_data)) * tf.matmul((y_pred - y), (y_pred - y), transpose_a=True)  # (m*1).T @ m*1 = 1*1
        #损失函数定义
    opt = optimizer(learning_rate=alpha)#学习率
    opt_operation = opt.minimize(loss)#最小化损失函数

    # 运行session
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        loss_data = []

        for i in range(epoch):
            _, loss_val, W_val = sess.run([opt_operation, loss, W], feed_dict={X: X_data, y: y_data})
            loss_data.append(loss_val[0, 0])  # 因为每一个损失值都是1*1 ndarray

            if len(loss_data) > 1 and np.abs(loss_data[-1] - loss_data[-2]) < 10 ** -9:  # 当满足收敛条件时跳出
                break

    # 清理图
    tf.reset_default_graph()
    return {'loss': loss_data, 'parameters': W_val}  # 只想以行向量格式返回


data = pd.read_csv('ex1data1.txt', names=['population', 'profit'])#读取数据，并赋予列名

print(data.head())#看下数据前5行
X = get_X(data)
print(X.shape, type(X))

y = get_y(data)
print(y.shape, type(y))
#看下数据维度
theta = np.zeros(X.shape[1])#X.shape[1]=2,代表特征数n
def lr_cost(theta, X, y):
#     """
#     X: R(m*n), m 样本数, n 特征数
#     y: R(m)
#     theta : R(n), 线性回归的参数
#     """
    m = X.shape[0]#m为样本数

    inner = X @ theta - y  # R(m*1)，X @ theta等价于X.dot(theta)

    # 1*m @ m*1 = 1*1 矩阵乘法
    # numpy没有在1d数组中进行转置，所以这里只有一个
    # 向量内积
    square_sum = inner.T @ inner
    cost = square_sum / (2 * m)

    return cost
print(lr_cost(theta, X, y))#返回theta的值
def gradient(theta, X, y):
    m = X.shape[0]

    inner = X.T @ (X @ theta - y)  # (m,n).T @ (m, 1) -> (n, 1)，X @ theta等价于X.dot(theta)

    return inner / m
def batch_gradient_decent(theta, X, y, epoch, alpha=0.01):
#   拟合线性回归，返回参数和代价
#     epoch: 批处理的轮数
#     """
    cost_data = [lr_cost(theta, X, y)]
    _theta = theta.copy()  # 拷贝一份，不和原来的theta混淆

    for _ in range(epoch):
        _theta = _theta - alpha * gradient(_theta, X, y)
        cost_data.append(lr_cost(_theta, X, y))

    return _theta, cost_data
#批量梯度下降函数
epoch = 500
final_theta, cost_data = batch_gradient_decent(theta, X, y, epoch)
print (final_theta)
#最终的theta
print (cost_data)
# 看下代价数据
# 计算最终的代价
print (lr_cost(final_theta, X, y))
ax = sns.tsplot(cost_data, time=np.arange(epoch+1))
ax.set_xlabel('epoch')
ax.set_ylabel('cost')
plt.show()
#可以看到从第二轮代价数据变换很大，接下来平稳了
b = final_theta[0] # intercept，Y轴上的截距
m = final_theta[1] # slope，斜率

plt.scatter(data.population, data.profit, label="Training data")
plt.plot(data.population, data.population*m + b, label="Prediction")
plt.legend(loc=2)
plt.show()
