"""
Created on Sat Mar  9 22:02:25 2019

@author: liweimin
"""

import turtle
import random
import time
import jieba
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from sklearn.cluster import KMeans
from sklearn.datasets import load_boston
from sklearn.datasets import load_iris



a=np.random.rand(1000000)
b=np.random.rand(1000000)

start_time=time.perf_counter()

c=np.dot(a,b)

end_time=time.perf_counter()

time1=end_time-start_time

print(time1)
print(c)


start_time2=time.perf_counter()


c=0
for i in range(1000000):
    c+=a[i]*b[i]

end_time2=time.perf_counter()

time2=end_time2-start_time2

print(time2)
print(c)













#






    



       
    
    

