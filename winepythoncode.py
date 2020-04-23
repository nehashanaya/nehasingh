#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing required packages.
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#Loading dataset
wine = pd.read_csv('C:/Users/admin/Desktop/neha singh/winequality-red.csv')


# In[3]:


#Let's check how the data is distributed
wine.head()


# In[4]:


#Information about the data columns
wine.info()


# In[5]:


#Here we see that fixed acidity does not give any specification to classify the quality.
fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'fixed acidity', data = wine)


# In[6]:


#Here we see that its quite a downing trend in the volatile acidity as we go higher the quality 
fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'volatile acidity', data = wine)


# In[7]:


#Composition of citric acid go higher as we go higher in the quality of the wine
fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'citric acid', data = wine)


# In[8]:


fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'residual sugar', data = wine)


# In[9]:


#Composition of chloride also go down as we go higher in the quality of the wine
fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'chlorides', data = wine)


# In[10]:


fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'free sulfur dioxide', data = wine)


# In[11]:


fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'total sulfur dioxide', data = wine)


# In[12]:


#Sulphates level goes higher with the quality of wine
fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'sulphates', data = wine)


# In[13]:


#Alcohol level also goes higher as te quality of wine increases
fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'alcohol', data = wine)


# In[14]:


#Making binary classificaion for the response variable.
#Dividing wine as good and bad by giving the limit for the quality
bins = (2, 6.5, 8)
group_names = ['bad', 'good']
wine['quality'] = pd.cut(wine['quality'], bins = bins, labels = group_names)


# In[15]:


#Now lets assign a labels to our quality variable
label_quality = LabelEncoder()


# In[16]:


#Bad becomes 0 and good becomes 1 
wine['quality'] = label_quality.fit_transform(wine['quality'])


# In[17]:


wine['quality'].value_counts()


# In[18]:


sns.countplot(wine['quality'])


# In[19]:


#Now seperate the dataset as response variable and feature variabes
X = wine.drop('quality', axis = 1)
y = wine['quality']


# In[20]:


#Train and Test splitting of data 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


# In[21]:


#Applying Standard scaling to get optimized result
sc = StandardScaler()


# In[22]:


X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


# In[23]:


rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X_train, y_train)
pred_rfc = rfc.predict(X_test)


# In[24]:


#Let's see how our model performed
print(classification_report(y_test, pred_rfc))


# In[25]:


#Confusion matrix for the random forest classification
print(confusion_matrix(y_test, pred_rfc))


# In[26]:


sgd = SGDClassifier(penalty=None)
sgd.fit(X_train, y_train)
pred_sgd = sgd.predict(X_test)


# In[27]:


print(classification_report(y_test, pred_sgd))


# In[28]:


print(confusion_matrix(y_test, pred_sgd))


# In[29]:


svc = SVC()
svc.fit(X_train, y_train)
pred_svc = svc.predict(X_test)


# In[30]:


print(classification_report(y_test, pred_svc))


# In[31]:


#Finding best parameters for our SVC model
param = {
    'C': [0.1,0.8,0.9,1,1.1,1.2,1.3,1.4],
    'kernel':['linear', 'rbf'],
    'gamma' :[0.1,0.8,0.9,1,1.1,1.2,1.3,1.4]
}
grid_svc = GridSearchCV(svc, param_grid=param, scoring='accuracy', cv=10)


# In[ ]:




