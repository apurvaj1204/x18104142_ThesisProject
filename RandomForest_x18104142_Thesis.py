#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib


# In[ ]:


#Creating Dataset and including the first row by setting no header as input
dataset = pd.read_csv('/Users/apurvajain/Desktop/FinalProject/finaldatathesis1k_1.csv')#Renaming the columns
print('Shape of the dataset: ' + str(dataset.shape))
print(dataset.head())


# In[ ]:


#Creating the dependent variable class
factor = pd.factorize(dataset['final_result'])
dataset.final_result = factor[0]
definitions = factor[1]
print(dataset.final_result.head())
print(definitions)


# In[ ]:


dataset1 = dataset.values
X = dataset1[:,0:28]
Y = dataset1[:,28]
print('The independent features set: ')
print(X[:28,:])
print('The dependent variable: ')
print(Y[:28])


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25)


# In[ ]:


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[ ]:


classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 42)
classifier.fit(X_train, y_train)


# In[ ]:


# Predicting the Test set results
y_pred = classifier.predict(X_test)
#Reverse factorize 
reversefactor = dict(zip(range(3),definitions))
y_test = np.vectorize(reversefactor.get)(y_test)
y_pred = np.vectorize(reversefactor.get)(y_pred)
print(pd.crosstab(y_test, y_pred, rownames=['Actual Result'], colnames=['Predicted Result']))


# In[ ]:


print(list(zip(dataset.columns[0:21], classifier.feature_importances_)))
joblib.dump(classifier, 'randomforestmodel.pkl') 


# In[ ]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)


# In[ ]:


cm = confusion_matrix(y_test, y_pred) 


# In[ ]:


cm = confusion_matrix(y_test, y_pred) 


# In[ ]:


from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


# In[ ]:


precision_score(y_test, y_pred, average='macro')
recall_score(y_test,y_pred, average='macro')
f1_score(y_test, y_pred, average='macro')
accuracy_score(y_test, y_pred, average='macro')


# In[ ]:


recall_score(y_test,y_pred, average='macro'
f1_score(y_test, y_pred, average='macro')
precision_score(y_test, y_pred, average='macro')

