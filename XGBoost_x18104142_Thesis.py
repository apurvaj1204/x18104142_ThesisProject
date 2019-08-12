#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib


# In[ ]:


dataset = pd.read_csv('/Users/apurvajain/Desktop/FinalProject/finaldatathesis20k_1.csv')#Renaming the columns
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


import xgboost as xgb

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)


# In[ ]:


from sklearn.datasets import dump_svmlight_file

dump_svmlight_file(X_train, y_train, 'dtrain.svm', zero_based=True)
dump_svmlight_file(X_test, y_test, 'dtest.svm', zero_based=True)
dtrain_svm = xgb.DMatrix('dtrain.svm')
dtest_svm = xgb.DMatrix('dtest.svm')


# In[ ]:


param = {
    'max_depth': 3,  # the maximum depth of each tree
    'eta': 0.3,  # the training step for each iteration
    'silent': 1,  # logging mode - quiet
    'objective': 'multi:softprob',  # error evaluation for multiclass training
    'num_class': 3}  # the number of classes that exist in this datset
num_round = 50  # the number of training iterations


# In[ ]:


# training and testing - numpy matrices
from sklearn.metrics import precision_score
bst = xgb.train(param, dtrain, num_round)
preds = bst.predict(dtest)
# extracting most confident predictions
best_preds = np.asarray([np.argmax(line) for line in preds])
print ("Numpy array precision:", precision_score(y_test, best_preds, average='macro'))


# In[ ]:


bst.dump_model('dump.raw.txt')
bst_svm.dump_model('dump_svm.raw.txt')
# save the models for later
joblib.dump(bst_svm, 'bst_svm_model.pkl', compress=True)


# In[ ]:


joblib.dump(bst, 'bst_model.pkl', compress=True)


# In[ ]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
accuracy_score(y_test, best_preds)
recall_score(y_test, best_preds, average='macro')
f1_score(y_test, best_preds, average='macro')
precision_score(y_test, best_preds, average='macro')

