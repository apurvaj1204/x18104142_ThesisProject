#!/usr/bin/env python
# coding: utf-8

# In[93]:


import pandas as pd
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split


# In[94]:


dataset = pd.read_csv('/Users/apurvajain/Desktop/FinalProject/finaldatathesis20k_1s.csv')


# In[95]:


dataset1 = dataset.values
X = dataset1[:,0:28]
Y = dataset1[:,28]


# In[ ]:


dataset.shape


# In[97]:


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25)


# In[ ]:


from sklearn.preprocessing import MinMaxScaler
scaling = MinMaxScaler(feature_range=(-1,1)).fit(X_train)
X_train = scaling.transform(X_train)
X_test = scaling.transform(X_test)


# In[100]:


from sklearn.svm import SVC 
svm_model_linear = SVC(kernel = 'linear', C = 1).fit(X_train, y_train) 
svm_predictions = svm_model_linear.predict(X_test) 


# In[101]:


accuracy = svm_model_linear.score(X_test, y_test) 


# In[ ]:


accuracy


# In[103]:


cm = confusion_matrix(y_test, svm_predictions) 


# In[ ]:


cm


# In[105]:


from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


# In[ ]:


print(f1_score(y_test, svm_predictions, average="macro"))
print(precision_score(y_test, svm_predictions, average="macro"))
print(recall_score(y_test, svm_predictions, average="macro"))    

