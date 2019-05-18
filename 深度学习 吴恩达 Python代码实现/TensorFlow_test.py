"""
Created on 2019-5-18 22:26:57

@author: liweimin
"""

import random
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf 

w=tf.Variable(0,dtype=tf.float32)
x = tf.placeholder(tf.float32,[3,1])
#coefficients = np.array([[1.],[-10.],[25.]])
coefficients = np.array([[1.],[-20.],[100.]])
#cost=tf.add(tf.add(w**2,tf.multiply(-10.,w)),25)
#cost=w**2-10*w+25
cost = x[0][0]*w**2 +x[1][0]*w + x[2][0]

train=tf.train.GradientDescentOptimizer(0.01).minimize(cost)

init=tf.global_variables_initializer()
session=tf.Session()
session.run(init)
print(session.run(w))

for i in range(1000):
    session.run(train,feed_dict = {x:coefficients})
print(session.run(w))





