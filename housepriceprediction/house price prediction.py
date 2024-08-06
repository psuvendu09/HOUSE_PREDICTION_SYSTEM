#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics


# In[2]:


data=pd.read_csv("C:/4thsem stuff/USA_Housing.csv")
data.head()


# In[3]:


data=data.drop(['Address'],axis=1)
data.head()


# In[4]:


sns.heatmap(data.isnull())


# In[5]:


X=data.drop(['Price'],axis=1)
Y=data['Price']

X_train,X_test,Y_train,Y_test =train_test_split(X,Y,test_size=.30)


# In[6]:


model=LinearRegression()
model.fit(X_train,Y_train)


# In[7]:


predictions=model.predict(X_test)


# In[8]:


predictions


# In[9]:


error=np.sqrt(metrics.mean_absolute_error(Y_test,predictions))


# In[10]:


error


# In[ ]:




