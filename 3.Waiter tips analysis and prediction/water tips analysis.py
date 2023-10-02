#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pyforest


# In[2]:


dataset = pd.read_csv("waiter tips analysis.csv")


# In[3]:


dataset.head()


# In[4]:


dataset.info()


# In[5]:


dataset.isnull().sum()


# In[6]:


dataset.shape


# In[7]:


dataset.corr()


# In[8]:


ax = sns.heatmap(dataset.corr(),annot = True)


# In[9]:


dataset.head()


# In[17]:


figure = px.scatter(dataset,x='total_bill',y = "tip",trendline="ols",size = 'size',color = "day")
figure.show()


# In[19]:


figure = px.scatter(dataset,x='total_bill',y = "tip",trendline="ols",size = 'size',color = "sex")
figure.show()


# In[20]:


figure = px.scatter(dataset,x='total_bill',y = "tip",trendline="ols",size = 'size',color = "time")
figure.show()


# In[26]:


figure = px.pie(dataset,names = "day",values = "tip",hole = 0.5)
figure.show()


# In[27]:


figure = px.pie(dataset,names = "sex",values = "tip",hole = 0.5)
figure.show()


# In[28]:


figure = px.pie(dataset,names = "smoker",values = "tip",hole = 0.5)
figure.show()


# In[29]:


figure = px.pie(dataset,names = "time",values = "tip",hole = 0.5)
figure.show()


# In[30]:


dataset = pd.get_dummies(dataset,drop_first = True)


# In[31]:


dataset.head()


# In[32]:


dataset.corr()


# In[33]:


ax = sns.heatmap(dataset.corr(),annot = True,cmap='RdGy')


# In[36]:


x = np.array(dataset.drop(['tip'],1))
y= np.array(dataset['tip'])
print(y)


# In[38]:


x_train,x_test,y_train,y_test = train_test_split(x,y,random_state =0,test_size = 0.2)


# In[39]:


regressor = LinearRegression()
regressor.fit(x_train,y_train)


# In[40]:


regressor.score(x_test,y_test)


# In[42]:


print(x)


# In[45]:


features = np.array([[24.50, 1, 0, 0, 1, 4,1,0]])
regressor.predict(features)


# # Other method

# In[59]:


data  = pd.read_csv('waiter tips analysis.csv')


# In[60]:


data.head()


# In[61]:


data["sex"] = data["sex"].map({"Female": 0, "Male": 1})
data["smoker"] = data["smoker"].map({"No": 0, "Yes": 1})
data["day"] = data["day"].map({"Thur": 0, "Fri": 1, "Sat": 2, "Sun": 3})
data["time"] = data["time"].map({"Lunch": 0, "Dinner": 1})
data.head()


# In[63]:


x1 =data.drop(['tip'],1)
y1= data['tip']
print(x1)


# In[65]:


x_train,x_test,y_train,y_test = train_test_split(x1,y1,test_size=0.2,random_state=42)


# In[66]:


regressor = LinearRegression()
regressor.fit(x_train,y_train)


# In[67]:


regressor.score(x_test,y_test)


# In[68]:


features = np.array([[24.50, 1, 0, 0, 1, 4]])
regressor.predict(features)


# In[ ]:


#So here we can see that my method is better than the given method.😎😎

