#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


train_data=pd.read_csv(open(r'D:\数智考核任务\sznlpassessment2022\train_.csv'))
train_data


# In[3]:


train_data.outcome.isna().sum()    #查看测试集outcome缺失值数量


# In[4]:


train_data.dropna(subset=['outcome'], inplace=True)   #缺失的outcome所在的行的数据不可用，要删掉
train_data


# In[5]:


train_data.info()   #查看删掉后的训练集（其他列缺失值依然存在）
train_data


# In[6]:


test_data=pd.read_csv(open(r'D:\数智考核任务\sznlpassessment2022\test_.csv'))
test_data


# In[7]:


test_data.info()  #查看测试集，发现有缺失值


# In[8]:


#将测试集的缺失值用训练集的均值填充（因为实际场景中测试集不可见）

def impute_nan(data, column, mean):     #函数传入数据，想要填充数据的列数，以及该列的均值
    data[column]=data[column].fillna(mean)   
# impute_nan(test_data,'0',test_data['0'].mean())   #错误，不能用测试集
# impute_nan(test_data,'12',test_data['12'].mean())   
                
impute_nan(test_data,'0',train_data['0'].mean()) 
impute_nan(test_data,'12',train_data['12'].mean()) 
              
#改进填充缺失值方法，上面的太繁琐
# test_data.fillna(train_data.mean, inplace=True)
# train_data.fillna(train_data.mean, inplace=True)


# In[9]:


test_data


# In[10]:


train_data


# In[11]:


train_data.isna().sum()


# In[12]:


# impute_nan(train_data,'12',train_data['12'].mean()) #太繁琐
# impute_nan(train_data,'11',train_data['11'].mean()) 
# train_data.fillna(train_data.mean, inplace=True)#用了这个内核查看train_data内核就挂掉了


# In[13]:


impute_nan(train_data,'12',train_data['12'].mean()) #太繁琐，要改进
impute_nan(train_data,'11',train_data['11'].mean()) 
impute_nan(train_data,'10',train_data['10'].mean())
impute_nan(train_data,'9',train_data['9'].mean()) 
impute_nan(train_data,'8',train_data['8'].mean()) 
impute_nan(train_data,'7',train_data['7'].mean()) 
impute_nan(train_data,'6',train_data['6'].mean())
impute_nan(train_data,'5',train_data['5'].mean()) 
impute_nan(train_data,'4',train_data['4'].mean()) 
impute_nan(train_data,'3',train_data['3'].mean()) 
impute_nan(train_data,'2',train_data['2'].mean()) 
impute_nan(train_data,'1',train_data['1'].mean())
impute_nan(train_data,'0',train_data['0'].mean()) 


# In[14]:


train_data.isna().sum()


# In[15]:


test_data.isna().sum()


# In[16]:


train_data=train_data.iloc[:,1:]   #去掉第一列无用数据(id)


# In[17]:


test_data=test_data.iloc[:,1:]   #去掉第一列无用数据


# In[18]:


train_data.info()


# In[19]:


test_data.info()


# In[20]:


outcome=train_data.iloc[:,-1:]  #取出outcome值
outcome


# In[21]:


y_train=outcome.values #将outcome作为y值(一维)


# In[22]:


x_train=train_data.iloc[:,:-1]  #训练集的特征矩阵
x_test=test_data   #测试集的特征矩阵


# In[23]:


describe_df=x_train.describe()
describe_df


# In[24]:


def filter_Outliers(column_name, data):  #处理训练集特征矩阵的异常值
#异常值处理出现异常值，返回True
    std_temp = describe_df.loc['std',column_name]
    mean_temp = describe_df.loc['mean',column_name]
    if (np.abs(data-mean_temp) > 3*std_temp):
        print('Outliers')
        return True
    return False


# In[25]:


for column_name in x_train.columns:
    for index in x_train.index:
        data = x_train.loc[index][column_name]
        Outliers = filter_Outliers(column_name, data)
#出现异常值删除
        if Outliers:
            print(index)
            outcome.drop(index,inplace=True)   #连outcome一起删掉，不然后面会报错（刚写的时候没有这行代码）
            x_train.drop(index, inplace=True)


# In[26]:


# def max_min_normalization(data_value, data_col_max_values, data_col_min_values):#归一化       #这个不太简洁

# #     data_value: 要归一化的数据
# #     data_col_max_values: 数据每列的最大值
# #     data_col_min_values: 数据每列的最小值
#     data_shape = data_value.shape
#     data_rows = data_shape[0]#行数
#     data_cols = data_shape[1]#列数

#     for i in range(0, data_rows, 1):
#         for j in range(0, data_cols, 1):
#             data_value.loc[i][j] = (data_value.loc[i][j] - data_col_min_values[j])/(data_col_max_values[j] - data_col_min_values[j])
train_min, train_max = x_train.min(axis=0), x_train.max(axis=0)
x_train = (x_train - train_min) / (train_max - train_min)
x_test = (x_test - train_min) / (train_max - train_min)##将x_test的数据进行归一化 这个真的特别重要，特别重要特别重要！！！！！！！！别忘了


# In[27]:


# max_min_normalization(x_train, np.max(x_train), np.min(x_train))#将x_train的数据进行归一化


# In[28]:


# max_min_normalization(x_test, np.max(x_test), np.min(x_test))#将x_test的数据进行归一化 这个真的特别重要，特别重要特别重要！！！！！！！！别忘了


# In[29]:


# def Z_ScoreNormalization(data_value):  #标准化（这个一般不用在梯度下降）
    
#     data_shape = data_value.shape
#     data_rows = data_shape[0]
#     data_cols = data_shape[1]
#     for i in range(0, data_rows, 1):
#         for j in range(0, data_cols, 1):
#             data_value.loc[i][j] = (data_value.loc[i][j]-np.average(data_value.loc[j]))/np.std(data_value.loc[j])


# In[30]:


x_train   #这里发现索引与行数不符合，要更改索引


# In[31]:


x_train.index=range(x_train.shape[0])
x_train


# In[32]:


x_test


# In[33]:


x_train.insert(0,0,1)  #插入偏置项
x_test.insert(0,0,1)


# In[34]:


x_train.columns=range(0,x_test.shape[1])   #需要改列名，不然后面列名会重复
x_train


# In[35]:


x_train=x_train.values


# In[36]:


x_test.columns=range(0,x_test.shape[1])
x_test


# In[37]:


x_test=x_test.values


# In[38]:


weights=np.random.random(x_train.shape[1]).reshape(x_train.shape[1],1)#初始化权重
weights.shape   #这里这样处理会出现二维的结果


# In[39]:


# 学习率
lr = 0.001
# 迭代次数
num_iteration = 1000

losses = []


# In[40]:


# 损失函数
def loss_function(w, X, Y):
    loss = np.dot(X, w) - Y  # dot() 数组需要像矩阵那样相乘，就需要用到dot()
    return (1/(2*len(X))) * np.dot(loss.T, loss)  #这个2不影响，是为了数学形式好看，当求导的时候把平方的2消掉
#  + 10*w*w #l2正则化


# In[41]:


#梯度
def gradient(w, X, Y):
    loss = np.dot(X, w) - Y
    return (1/len(X)) *np.dot(X.T, loss) 
# + 10*w   正则


# In[42]:


# 梯度下降迭代
def gradient_descent(X, Y, lr,w):
    gra = gradient(w, X, Y)
    for i in range(num_iteration):
        w = w - lr * gra  #更新权重
        gra = gradient(w, X, Y) #更新梯度
        loss=loss_function(w, X, Y)
        if i % 100 == 0:
            print('Iteration {}, loss={}'.format(i, loss))
        losses.append(loss)
    return w


# In[43]:


# for i in range(1, num_iteration + 1):     #比较直观
#     # 预测
#     y_hat = np.dot(x_train,weights)
#     # 计算损失
#     loss = np.mean(np.square(y_hat - y_train)     #(1/*len(X)) * np.dot((y_hat - y_train).T, y_hat - y_train)
#     # 计算梯度
#     gra = np.dot(y_hat - y_outcome,x_train)
#     # 更新权重
#     weights -= lr * gra
#     if i % 100 == 0:
#         print('Iteration {}, loss={}'.format(i, loss))
#     losses.append(loss)


# In[44]:


x_train


# In[45]:


y_train


# In[50]:


result_weights=gradient_descent(x_train,y_train,lr,weights)  #查看到loss为2维，所以下面的画图会报错，要改为一维


# In[47]:


plt.plot(losses)
plt.title('Loss')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.show()


# In[ ]:


y_pred = np.dot(x_test,result_weights)


# In[ ]:


y_pred = pd.Series(index=range(len(x_test)), data=y_pred)  #把dataframe转化成series


# In[ ]:


with open('submission0.csv', 'w') as f:
    f.write('id,outcome\n')
    for index in y_pred.index:
        f.write('{},{}\n'.format(index, y_pred[index]))

