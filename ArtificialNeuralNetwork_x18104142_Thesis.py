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


# In[ ]:


dataset = pd.read_csv('/Users/apurvajain/Desktop/FinalProject/cleanedData1k.csv')


# In[ ]:


dataset.head(2)


# In[ ]:


del dataset['id_student'] #removing the original columns without encoding


# In[ ]:


dataset.describe(include='all')


# In[ ]:


sns.heatmap(dataset.corr(), annot=True)


# In[ ]:


df_education = pd.get_dummies(dataset['highest_education'],
                              columns=['A Level or Equivalent', 
                                       'Lower Than A Level', 'HE Qualification',
                                       'Post Graduate Qualification', 'No Formal quals'])


# In[ ]:


df_education.head(2)


# In[ ]:


df_new = pd.concat([dataset, df_education], axis=1)


# In[ ]:


df_new.rename(columns={0: "A Level or Equivalent", 1: "Lower Than A Level", 2: "HE Qualification",
                       3: "Post Graduate Qualification", 4: "No Formal quals"}, inplace=True)


# In[ ]:


cols = list(df_new.columns.values)


# In[ ]:


print(df_new.columns)


# In[ ]:


cols


# In[ ]:


df_imd = pd.get_dummies(dataset['imd_band'])


# In[ ]:


df_newimd = pd.concat([df_new, df_imd], axis=1)


# In[ ]:


df_newimd.rename(columns={0: "lowerIMD", 1: "moderateIMD", 2: "higherIMD" }, inplace=True)


# In[ ]:


cols_1 = list(df_newimd.columns.values)


# In[ ]:


df_age = pd.get_dummies(dataset['age_band'], columns=['0-35', '35-55', '55<='])


# In[ ]:


df_newage = pd.concat([df_newimd, df_age], axis=1)


# In[ ]:


df_newage.rename(columns={0: "0-35", 1: "35-55", 2: "55<=" }, inplace=True)


# In[ ]:


list(df_newage.columns.values)


# In[ ]:


del df_newage['highest_education'] #removing the original columns without encoding
del df_newage['imd_band'] #removing the original columns without encoding
del df_newage['age_band'] #removing the original columns without encoding


# In[ ]:


list(df_newage.columns.values)


# In[ ]:


df_newage = df_newage[['gender',
 'num_of_prev_attempts',
 'studied_credits',
 'disability',
 'date_registration',
 'sum_click',
 'date_submitted',
 'score',
 'weight',
 'module_presentation_length',
 'A Level or Equivalent',
 'Lower Than A Level',
 'HE Qualification',
 'Post Graduate Qualification',
 'No Formal quals',
 'lowerIMD',
 'moderateIMD',
 'higherIMD',
 '0-35',
 '35-55',
 '55<=',
'final_result']] #rearranging the columns


# In[ ]:


list(df_newage.columns.values)


# In[ ]:


df_newage.to_csv("/Users/apurvajain/Desktop/FinalProject/finaldatathesis.csv", encoding='utf-8', index=False)


# In[ ]:


# fix random seed for reproducibility
seed = 7
np.random.seed(seed)


# In[ ]:


df_newage.shape


# In[ ]:


dataset = df_newage.values
X = dataset[:,0:28].astype(float)
Y = keras.utils.to_categorical(dataset[:,28], num_classes=3)


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=seed)


# In[ ]:


#def base_model():
    model=Sequential()
    model.add(Dense(28, input_dim=28, activation='relu', kernel_initializer='random_normal'))
    model.add(Dense(14, activation='relu', kernel_initializer='random_normal'))
    model.add(Dense(7, activation='relu', kernel_initializer='random_normal'))
    model.add(Dense(3, activation='softmax', kernel_initializer='random_normal'))


# In[ ]:


# Compile model
    model.compile(loss='categorical_crossentropy', optimizer= 'adam', metrics=['accuracy'])


# In[ ]:


model.fit(X_train,Y_train, batch_size=16, epochs=100) #fitting the model


# In[ ]:


score, acc = model.evaluate(X_train, Y_train, verbose=1, batch_size=16) #evaluate the model with trained data


# In[ ]:


acc


# In[ ]:


y_pred = model.predict(X_test, batch_size=32, verbose=1, steps=None )


# In[ ]:


score_test, acc =model.evaluate(X_test, Y_test, verbose=1, batch_size=32)


# In[ ]:


score_test, acc


# In[ ]:


from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt


# In[ ]:


n_classes = 3


# In[ ]:


fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(Y_test[:, i], y_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])


# In[ ]:


for i in range(n_classes):
    plt.figure()
    plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f)' % roc_auc[i])
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()


# In[ ]:


y_pred = (y_pred > 0.5)


# In[ ]:


pre_micro = metrics.precision_score(Y_test, y_pred, average="micro")
recall_micro = metrics.recall_score(Y_test, y_pred, average="micro")
f1_micro_scikit = metrics.f1_score(Y_test, y_pred, average="micro")


# In[ ]:


print ("Prec_micro_scikit:", pre_micro)
print ("Rec_micro_scikit:", recall_micro)
print ("f1_micro_scikit:", f1_micro_scikit)

