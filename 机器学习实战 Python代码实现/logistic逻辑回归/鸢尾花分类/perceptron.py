import numpy as np
import math
import random
import pandas
from sklearn import datasets
from sklearn.model_selection import  train_test_split

#sigmoid函数
def sigmoid(x):  
        return 1.0 / (1.0 + np.exp(-x))  
    
def sample_mean(data):
    return sum(data) / len(data)


def sample_variance(data, mean=None):
    if mean is None:
        mean = sample_mean(data)

    return sum([(x - mean) ** 2 for x in data]) / len(data)


def get_covariance_matrix(sample_input):
    row, column = sample_input.shape
    means = np.zeros((column))

    for i in range(column):
        for j in range(row):
            means.itemset(i, means.item(i) + sample_input.item((j, i))/float(row))

    result = np.asmatrix(np.zeros((column, column)))

    for i in range(column):
        for j in range(i+1):
            total = 0.0
            for k in range(row):
                total += (sample_input.item((k, i)) - means.item(i))*(sample_input.item((k, j)) - means.item(j))

            total /= float(row - 1)
            result.itemset((i, j), total)
            result.itemset((j, i), total)
    return result


def __init__(self, k, d):
        self.k = k
        self.d = d
        self.learning_rate = 0.01
        self.weights = np.random.rand(k, d + 1)
#训练算法
def train(self, input_data, output_data):
        for outer in range(1000):
            error = 0.0
            delta = np.zeros((self.k, self.d + 1))

            for t in range(input_data.shape[0]):
                outputs = [0.0] * self.k
                positive_count = 0
                for i in range(self.k):
                    for j in range(self.d + 1):
                        outputs[i] += self.weights.item((i, j)) * input_data.item((t, j))

                    if outputs[i] > 0:
                        positive_count += 1

                y = [0.0] * self.k
                total = 0.0

                for i in range(self.k):
                    y[i] = math.pow(math.e, outputs[i])
                    total += y[i]

               log_index = 0
                for i in range(self.k):
                    y[i] /= total
                    if y[i] > y[m_index]:
                       log_index = i

                for i in range(self.k):
                    expected_result = output_data.item((t, i))
                    if (expected_result == 1 and outputs[i] <= 0.0) or (expected_result == 0 and outputs[i] >= 0.0):
                        error += 1
                        for j in range(self.d + 1):
                            delta.itemset((i, j), delta.item((i, j)) + (output_data.item((t, i)) - y[i]) * input_data.item((t, j)))

            if error == 0.0:
                break

            for i in range(self.k):
                for j in range(self.d + 1):
                    self.weights.itemset((i, j), self.weights.item((i, j)) + self.learning_rate * delta.item((i, j)))
#反馈
def response(self, input_data):
        outputs = [0.0] * self.k
        for i in range(self.k):
            for j in range(self.d + 1):
                outputs[i] += self.weights.item((i, j)) * input_data.item((j,))

        y = [0.0] * self.k
        total = 0.0

        for i in range(self.k):
            y[i] = math.pow(math.e, outputs[i])
            total += y[i]

       log_index = 0
        for i in range(self.k):
            y[i] /= total
            if y[i] > y[m_index]:
               log_index = i

        returnlog_index
    

if __name__ == '__main__':

    datas=[] 
    labels=[]
    datas_test=[]
    labels_test=[]

with open('IRIS.txt', 'r') as f_iris:
    lines = f_iris.readlines()

with open('train.txt', 'w') as ftr, open('test.txt', 'w') as fte:
    for i in range(105):   #70%进行训练，30%进行测试，有一点小问题
        ftr.write(lines.pop(random.randint(0, len(lines) - 1)))
    fte.writelines(lines)
data_file=open('train.txt','r')
data_file_test=open('test.txt','r')

for line in data_file.readlines():
    linedata = line.split(',')
    datas.append(linedata[:-1])  
    labels.append(linedata[-1].replace('\n', ''))  
for line in data_file_test.readlines():
    linedata = line.split(',')
    datas_test.append(linedata[:-1])  
    labels_test.append(linedata[-1].replace('\n', ''))  
        
iris=datasets.load_iris() # 加载数据集

X = iris.data # 数据集的特征部分
Y = iris.target # 数据集的分类标签

x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size = 0.3, random_state = 0)

    
train(x_train,x_test,'output2.txt')

# rights列表储存标签数据的序号
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
dataset = pandas.read_csv(data, names=names) 
 
array = dataset.values 
X = array[:,0:4] 
Y = array[:,4] 
validation_size = 0.30 
seed = 2019
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed) #分割数据集 

seed = 2019
scoring = 'accuracy'

log = LogisticRegression()
log.fit(X_train, Y_train) #log拟合序列集
predictions = log.predict(X_validation) #预测验证集
print('log验证集精度得分:%f' % accuracy_score(Y_validation, predictions)) 
print('分类预测报告' +classification_report(Y_validation, predictions)) 