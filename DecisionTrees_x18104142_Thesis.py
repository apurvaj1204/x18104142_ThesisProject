#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
#Import the DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier


# In[ ]:


# load libraries
  from sklearn import datasets
  from sklearn import metrics
  from sklearn.model_selection import train_test_split
  import matplotlib.pyplot as plt    
  
  plt.style.use('ggplot')

  from sklearn import tree


# In[ ]:


dataset = pd.read_csv('/Users/apurvajain/Desktop/FinalProject/finaldatathesis20k_1.csv')


# In[ ]:


dataset.describe(include='all')


# In[ ]:


dataset1 = dataset.values
X = dataset1[:,0:28]
Y = dataset1[:,28]


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25)


# In[ ]:


model = tree.DecisionTreeClassifier(class_weight='balanced', criterion='entropy', max_depth=4,
            max_features=None, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=10,
            min_weight_fraction_leaf=0.2, presort=False, random_state=None,
            splitter='best')


# In[ ]:


model.fit(X_train, y_train)


# In[ ]:


# make predictions
    expected_y  = y_test
    predicted_y = model.predict(X_test)


# In[ ]:


print(expected_y)


# In[ ]:


print(predicted_y)


# In[ ]:


# summarize the fit of the model
    print(); print('tree.DecisionTreeClassifier(): ')
    print(); print(metrics.classification_report(expected_y, predicted_y))
    print(); print(metrics.confusion_matrix(expected_y, predicted_y))


# In[ ]:


print(); print(metrics.confusion_matrix(expected_y, predicted_y))


# In[ ]:


print("Accuracy:",metrics.accuracy_score(expected_y, predicted_y))

