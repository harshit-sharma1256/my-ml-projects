#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pyforest


# In[2]:


dataset = pd.read_csv("advertising.csv")


# In[3]:


dataset.head()


# In[4]:


dataset.isnull().sum()


# In[5]:


dataset.corr()


# In[6]:


ax = sns.heatmap(dataset.corr(), annot=True)


# In[7]:


plt.scatter(dataset['TV'],dataset['Sales'] ,
            c ="yellow",
            linewidths = 2,
            marker ="*",
            edgecolor ="red",
            s = 20)
plt.show()


# In[8]:


dataset.drop(['Newspaper'],axis = 1)


# In[9]:


plt.scatter(dataset['Newspaper'],dataset['Sales'] ,
            c ="yellow",
            linewidths = 2,
            marker ="*",
            edgecolor ="red",
            s = 20)
plt.show()


# In[10]:


dataset.describe()


# In[11]:


dataset.info()


# In[15]:


x= np.array(dataset.drop(["Sales"], 1))
y= y = np.array(dataset["Sales"])
#print(x)
#print(y)
x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                test_size=0.2, 
                                                random_state=42)


# In[16]:


regressor = LinearRegression()
regressor.fit(x_train,y_train.reshape(-1,))
print(regressor.score(x_test,y_test))


# In[17]:


figure = px.scatter(data_frame = dataset, x="Sales",
                    y="TV", size="TV", trendline="ols",color="Sales")
figure.show()


# In[19]:


user_input = np.array([[437,42,69]])
print(regressor.predict(user_input))


# In[20]:


user_input = np.array([[43,425,690]])
print(regressor.predict(user_input))


# In[21]:


user_input = np.array([[4,42,6900]])
print(regressor.predict(user_input))


# In[22]:


user_input = np.array([[4370,42,69]])
print(regressor.predict(user_input))

