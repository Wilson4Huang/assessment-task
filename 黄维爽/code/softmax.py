#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


train = pd.read_csv(open(r'C:\Users\David\Desktop\softmax数据\train.csv'))
test = pd.read_csv(open(r'C:\Users\David\Desktop\softmax数据\test.csv'))


# In[3]:


train


# In[4]:


train.shape


# In[5]:


test


# In[6]:


test.shape


# In[7]:


test.columns = train.columns[: -1]   #训练集和测试集的列名不一样，这里都按训练集列名统一处理


# In[8]:


train.isna().sum().sum()  #检查缺失值(两个.sum())


# In[9]:


test.isna().sum().sum()


# In[10]:


feature_names = np.array(['feature{}'.format(i) for i in range(1, 785)])


# In[11]:


X_train = train[feature_names].values
X_test = test[feature_names].values

y_train = train['label'].values


# In[12]:


train.label.value_counts().sort_index()   ##查看每个标签的总和


# In[13]:


X_train.dtype


# In[14]:


X_train.max()   #查看到最大值为255


# In[15]:


X_train.min()    #查看到最小值为0


# In[16]:


X_train = X_train.astype(np.float64)        #转换成float类型
X_test = X_test.astype(np.float64)      

X_train /= 255.0   #加小数，不然会变成整数
X_test /= 255.0


# In[17]:


def cross_entropy(y_true, y_hat):     #计算预测结果和真实标签之间的交叉熵损失
    n = len(y_true)
    loss = - (1 / n) * np.sum(y_true * np.log(y_hat))
    return loss


# In[18]:


def softmax(probs):     #Softmax函数
    
    exp = np.exp(probs)
    sum_exp = np.sum(np.exp(probs), axis=1, keepdims=True)  #keepdims是让sum保持原来的维度
    p = exp / sum_exp
    return p


# In[19]:


len(y_train)


# In[20]:


y_train


# In[21]:


len(set(y_train))


# In[22]:


np.zeros((len(y_train), len(set(y_train)))) .shape


# In[23]:


# def one_hot_fit(self, y):  #更好地onehot，主要是记录下标与标签值的映射（对应）关系
#     #for i in range(len(sorted(np.unique(y))))
#     #优化上面的代码 
#     #创建对应的字典（enumerate只能在字典迭代），np.unique(y)为每个不同的标签值
#     mapping = {e: i for i, e in enumerate(sorted(np.unique(y)))}    #{y[0]：0, y[1],: 1, y[2]: 2}（真实标签：下标）
#     mapping_rev = {i: e for e, i in self.mapping.items()}           #{0：y[0], 1: y[1], 2: y[2]}


# In[24]:


# def one_hot_transform(y):    #OneHot编码，将一维数组y转换为二维的one hot形式
        
#         n = len(y)
#         y_numeric = [mapping[e] for e in y]   # y为20000个   mapping[e]为e值的索引 也就是y_numeric=[{6，5,2...,2,9,5}]
        
#         #len(sorted(np.unique(y)))=9  （为有序不重复）
#         #len(set(y))=9  无序不重复 这里两种都可以
#         one_hot = np.zeros((n, len(set(y))))   #创建0矩阵，shape为（20000，10）
#         one_hot[np.arange(n), y_numeric] = 1    #再把0矩阵中的对应位置改为1，这就是y_train的onehot，y_train的每个值都有[0,0....1,...0]对应
#         return one_hot


# In[25]:


# def train(X, y):    #训练模型
      

#         # 样本数、特征维度（特征类别）
#         n_samples, n_features = X.shape
#         n_classes = len(set(y))   #0-9共10个

#         # 权重&bias
#         weights = np.random.rand(n_classes, n_features)
#         bias = np.zeros((1, n_classes))   #先onehot再加偏置项
#         # 将标签转换为one hot 形式
#         _one_hot_fit(y)
#         y_one_hot = _one_hot_transform(y)
#         # 记录损失
#         self.loss_history.clear()  #每次记录一个
#         for i in range(1, self.max_iters + 1):
#             # 预测
#             probs = np.dot(X, weights.T) + bias   #array可以直接加偏置项
#             probs = softmax(probs)
            
#             # 损失
#             loss = cross_entropy(y_one_hot, probs)

#             # 计算梯度
#             dw = np.dot(X.T, (probs - y_one_hot)) / n_samples
#             db = np.sum(probs - y_one_hot, axis=0) / n_samples  #偏置项的梯度别忘了！  bais * probs - y_one_hot

#             # 更新参数
#             weights -= lr * dw.T
#             bias -= lr * db
# #             lr = alpha * lr

#             loss_history.append(loss)

#             if i % 200 == 0:
#                 print('Iteration: {:3d}, loss: {:.4f}'.format(i, loss))


# In[26]:


# def predict(X): #预测结果
       
#     probs = np.dot(X, weights.T) + bias
#     probs = softmax(probs)
#     y_numeric = np.argmax(probs, axis=1)   #取每列最大概率（probs）的下标
#     return np.array([mapping_rev[e] for e in y_numeric])  #返回结果：标签值


# In[27]:


class Softmax:   #定义一个softmax的模型框架的类，增强可读性（bushi）hhh
    def __init__(self, max_iters=10, learning_rate=0.01):
        #先定义一些私有变量
        
        # 迭代次数
        self.max_iters = max_iters
        # 学习率
        self.lr = learning_rate
#         # 学习率衰减率
#         self.alpha = alpha
        # 权重
        self.weights = None
        # 偏置项
        self.bias = None
        
        
        # 映射，由于是多类别，需要转到one hot编码，这里需要记录一个映射
        # 例如总共10个类别，类别为9的样本标签可能映射到一个10维的向量，其中第9个元素为1，其余元素为0
        self.mapping = None
        
        # 反映射，方便逆向转换回去，例如模型最终预测结果（离散化处理后）得到一个类别向量
        # [0, 1, 0, 0, ..., 0]，需要能转换回去该向量对应标签
        self.mapping_rev = None
        
        # 记录训练历史记录
        self.loss_history = []

    def one_hot_fit(self, y):
      
        
        #创建映射字典（enumerate只能在字典迭代），np.unique(y)为每个不同的标签值
        #for i in range(len(sorted(np.unique(y))))
        #优化上面的代码
        self.mapping = {e: i for i, e in enumerate(sorted(np.unique(y)))}    #{y[0]：0, y[1],: 1, y[2]: 2}（真实标签：下标）
        self.mapping_rev = {i: e for e, i in self.mapping.items()}           #{0：y[0], 1: y[1], 2: y[2]}

    def one_hot_transform(self, y):    #OneHot编码，将一维数组y转换为二维的one hot形式
        
        n = len(y)
        y_numeric = [self.mapping[e] for e in y]   # y为20000个   mapping[e]为e值的索引 也就是y_numeric=[{6，5,2...,2,9,5}]
        
        #len(sorted(np.unique(y)))=9  （为有序不重复）
        #len(set(y))=9  无序不重复 这里两种都可以
        one_hot = np.zeros((n, len(set(y))))   #创建0矩阵，shape为（20000，10）
        one_hot[np.arange(n), y_numeric] = 1    #再把0矩阵中的对应位置改为1，这就是y_train的onehot，y_train的每个值都有[0,0....1,...0]对应
        return one_hot

    def train(self, X, y):    #训练模型
      

        # 样本数、特征维度（特征类别）
        n_samples, n_features = X.shape
        n_classes = len(set(y))   #0-9共10个

        # 权重&bias
        self.weights = np.random.rand(n_classes, n_features)
        self.bias = np.zeros((1, n_classes))   #先onehot再加偏置项
        
        # 将标签转换为one hot 形式
        self.one_hot_fit(y)
        y_one_hot = self.one_hot_transform(y)
        
        # 记录损失
        self.loss_history.clear()  #每次记录一个
        
        for i in range(1, self.max_iters + 1):
            # 预测
            probs = np.dot(X, self.weights.T) + self.bias   #array可以直接加偏置项
            probs = softmax(probs)
            
            # 损失
            loss = cross_entropy(y_one_hot, probs)

            # 计算梯度
            dw = np.dot(X.T, (probs - y_one_hot)) / n_samples
            db = np.sum(probs - y_one_hot, axis=0) / n_samples  #偏置项的梯度别忘了！  bais * probs - y_one_hot

            # 更新参数
            self.weights -= self.lr * dw.T
            self.bias -= self.lr * db
#             self.lr = alpha * self.lr  #学习率衰减

            self.loss_history.append(loss)

            if i % 200 == 0:
                print('Iteration: {:3d}, loss: {:.4f}'.format(i, loss))

    def predict(self, X):  #预测结果
      
        probs = np.dot(X, self.weights.T) + self.bias
        probs = softmax(probs)
        y_numeric = np.argmax(probs, axis=1)   #取每列最大概率（probs）的下标
        return np.array([self.mapping_rev[e] for e in y_numeric])  #返回结果：标签值


# In[28]:


# 迭代次数
NUM_ITER = 6000   #调参从2000到6000，loss越来越小，准确率越来越高
# 学习率
LR = 0.125
# #学习率衰减率
# alpha=0.5


# In[29]:


model = Softmax(max_iters=NUM_ITER, learning_rate=LR)
model.train(X_train, y_train)


# In[30]:


plt.figure(figsize=(8, 5))
plt.plot(model.loss_history)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Loss')
plt.show()


# In[31]:


y_train_predict = model.predict(X_train)   #模型评估

acc = (y_train == y_train_predict).mean()
print('准确率为', acc)


# In[32]:


y_predict = model.predict(X_test)
y_pred = pd.Series(index=test['id'].values, data=y_predict)


# In[33]:


with open('submission(softmax).csv', 'w') as f:
    f.write('id,label\n')
    for index in y_pred.index:
        f.write('{},{}\n'.format(index-1, y_pred[index]))


# In[ ]:




