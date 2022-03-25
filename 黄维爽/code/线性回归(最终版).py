#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


train=pd.read_csv(open(r'D:\数智考核任务\sznlpassessment2022\train_.csv'))
train


# In[3]:


train.info()    #发现缺失值


# In[4]:


train.describe()   #发现有异常值


# In[5]:


train.outcome.isna().sum()    #检查发现训练集中有样本的outcome缺失，这些样本不可用，直接删除


# In[6]:


train.dropna(subset=['outcome'], inplace=True)     #删除outcome缺失的列，其他的缺失值可以填充


# In[7]:


test=pd.read_csv(open(r'D:\数智考核任务\sznlpassessment2022\test_.csv'))
test


# In[8]:


test.rename({'Unnamed: 0': 'id'}, axis=1, inplace=True)    #处理训练集和测试集的列名，使其保持一致


# In[9]:


test


# In[10]:


y_train = train['outcome'].values     #取outcome作为y_train    
train.drop(['outcome'], axis=1, inplace=True)


# In[11]:


train.isna().sum()


# In[12]:


test.isna().sum()


# In[13]:


train.describe()   


# In[14]:


columns_name = train.columns[1:]
columns_name


# In[15]:


mean = train[columns_name].mean()
mean


# In[16]:


sigma = train[columns_name].std()
sigma


# In[17]:


lower = mean - 3 * sigma   #3sigma原则，定义上下界
upper = mean + 3 * sigma


# In[21]:


# 只要有一个特征属性不合理，该样本就删除
condition_lower = (train[columns_name] < lower).any(axis=1)
condition_upper = (train[columns_name] > upper).any(axis=1)


# In[22]:


condition_lower,condition_upper


# In[23]:


condition = condition_lower | condition_upper   


# In[24]:


condition


# In[25]:


condition.sum()   #查看要删除的异常值的行的数量


# In[27]:


# train = train[!condition].copy()      !condition语法错误
# y_train = y_train[!condition].copy()  



train = train[~condition].copy()   #~可以把false与ture反转，~为反转运算符
y_train = y_train[~condition].copy()


# In[29]:


y_train


# In[30]:


train.describe()  #查看删除之后的数据，发现特征‘3’全为0，该特征无意义，需要删除(而且不删除的话归一化也会发生错误)


# In[31]:


train.drop(['3'], axis=1, inplace=True)
test.drop(['3'], axis=1, inplace=True)


# In[32]:


train_mean = train.mean()           #使用训练集的样本均值填充训练集和测试集的缺失值
train.fillna(train_mean, inplace=True)
test.fillna(train_mean, inplace=True)   #因为实际情况测试集不可见


# In[33]:


train.isna().sum()


# In[34]:


test.isna().sum()


# In[35]:


X_train = train.drop(['id'], axis=1).values   #去掉id列
X_test = test.drop(['id'], axis=1).values


# In[36]:


X_train


# In[37]:


train_min = X_train.min(axis=0),
train_max = X_train.max(axis=0)
X_train = (X_train - train_min) / (train_max - train_min)    #将X_train的数据进行归一化
X_test = (X_test - train_min) / (train_max - train_min)    #将X_test的数据进行归一化 这个真的特别重要，特别重要特别重要！！！！！！！！别忘了


# In[38]:


X_train


# In[39]:


X_test


# In[40]:


X_train = np.insert(X_train, 0, values=1, axis=1)   #添加一列全为1，也就是偏置项（在0列位置上插入的值为一，按列处理）
X_test = np.insert(X_test, 0, values=1, axis=1)


# In[41]:


X_train


# In[42]:


X_train.shape, X_test.shape,y_train.shape


# In[43]:


# 学习率
lr = 0.001
# 迭代次数
num_iteration = 1000
# l2正则化项参数
C = 10.0
# 训练样本数和特征维度
N, D = X_train.shape
# 初始化权重
weights = np.random.uniform(-0.05, 0.05, D)   #第一个数为下界，第二个数为上界，第三个数为随机数个数


# In[46]:


losses = []
for i in range(1, num_iteration + 1):
    # 预测值
    y_hat = np.dot(X_train,weights)
    # 计算损失
    loss = np.mean(np.square(y_hat - y_train))   #其中这里的正则项可以不写,这个loss的功能是看有没有收敛
    # 计算梯度
    dw = np.dot(y_hat - y_train,X_train)
    # 更新权重
#     weights -= lr * dw + lr * C * weights   #样本数远大于特征数，特征不算复杂，多，不需要正则化，反而正则化会使效果下降（亲测）
    weights -= lr * dw
    if i % 100 == 0:
        print('Iteration {:3d}, loss={:.4f}'.format(i, loss))
    losses.append(loss)


# In[47]:


plt.plot(losses)
plt.title('Loss')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.show()


# In[48]:


y_pred = X_test.dot(weights)
y_pred = pd.Series(index=test['id'].values, data=y_pred)


# In[49]:


y_pred


# In[50]:


with open('submission.csv', 'w') as f:
    f.write('id,outcome\n')
    for index in y_pred.index:
        f.write('{},{}\n'.format(index, y_pred[index]))


# In[ ]:





# In[ ]:




