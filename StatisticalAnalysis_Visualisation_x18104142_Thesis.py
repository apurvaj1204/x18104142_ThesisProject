#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import pandas as pd


# In[5]:


#visualization
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[8]:


def show_basic_info(df):
    print("========================================================================================================")
    print("HEAD:")
    print(df.head(3))
    print("--------------------------------------------------------------------------------------------------------")
    print("SHAPE:")
    print(df.shape)
    print("--------------------------------------------------------------------------------------------------------")
    print("INFO:")
    print(df.info())
    print("--------------------------------------------------------------------------------------------------------")
    print("DESCRIBE:")
    print(df.describe())
    print("--------------------------------------------------------------------------------------------------------")
    print("========================================================================================================")


# In[11]:


assessments_df = pd.read_csv('/Users/apurvajain/Desktop/FinalProject/anonymisedData/assessments.csv')
show_basic_info(assessments_df)


# In[12]:


courses_df = pd.read_csv('/Users/apurvajain/Desktop/FinalProject/anonymisedData/courses.csv')
show_basic_info(courses_df)


# In[13]:


studentAssessment_df = pd.read_csv('/Users/apurvajain/Desktop/FinalProject/anonymisedData/studentAssessment.csv')
show_basic_info(studentAssessment_df)


# In[14]:


studentInfo_df = pd.read_csv('/Users/apurvajain/Desktop/FinalProject/anonymisedData/studentInfo.csv')
show_basic_info(studentInfo_df)


# In[15]:


studentRegistration_df = pd.read_csv('/Users/apurvajain/Desktop/FinalProject/anonymisedData/studentRegistration.csv')
show_basic_info(studentRegistration_df)


# In[16]:


vle_df = pd.read_csv('/Users/apurvajain/Desktop/FinalProject/anonymisedData/vle.csv')
show_basic_info(vle_df)


# In[17]:


#Checking gender distribution
sns.countplot(studentInfo_df.gender);    #this shows that courses data is almost equally distributed on gender


# In[18]:


#Now let's try the same on age
studentInfo_df[['id_student', 'age_band']].groupby(by='age_band').count().plot.bar();    #this shows majority of students fall in age band of 0-35


# In[19]:


#Now let's try the same on region
studentInfo_df[['id_student', 'region']].groupby(by='region').count().plot.bar();


# In[20]:


pd.crosstab(studentInfo_df.region, studentInfo_df.age_band).plot.barh(stacked = True);


# In[21]:


#Finding the outliers
studentInfo_df.drop(['id_student', 'num_of_prev_attempts'], axis=1).boxplot(by = 'region')
plt.xticks(rotation = 90)    #without this, x-labels overlap 


# In[22]:


# plotting boxplot using seaborn
sns.boxplot(x = 'region', y = 'studied_credits', data=studentInfo_df)
plt.xticks(rotation = 90)


# In[23]:


# selecting a subset of cols which are of importance to us and grouping them by student id and aggregating them using median
studentPerformance_df = studentInfo_df[['id_student', 'num_of_prev_attempts', 'studied_credits']].groupby('id_student').median()


# In[24]:


studentPerformance_df.head()


# In[25]:


#indices are random,as we have selected it from df, we need to reset them
studentPerformance_df = studentPerformance_df.reset_index()


# In[26]:


studentPerformance_df.head()


# In[27]:


studentPerformance_df.head()


# In[28]:


studentPerformance_df.num_of_prev_attempts.unique()


# In[29]:


sns.countplot(studentPerformance_df.num_of_prev_attempts); #most students are giving their first attempt


# In[30]:


studentProfile_df = studentInfo_df[['id_student', 'gender', 'region', 'highest_education', 'imd_band', 'age_band']].drop_duplicates()


# In[31]:


show_basic_info(studentProfile_df)


# In[32]:


studentAges_df = studentInfo_df[['id_student', 'age_band']].groupby(['id_student']).count()
studentAges_df = studentAges_df.reset_index()
studentAges_df.age_band.hist();


# In[33]:


sns.countplot(studentInfo_df.code_module)


# In[34]:


pd.crosstab(studentInfo_df.code_module, studentInfo_df.code_presentation).plot.barh(stacked = True);


# In[35]:


studentInfo_df.head(2)


# In[36]:


sns.pairplot(data=studentInfo_df[["code_module","num_of_prev_attempts"]],hue="code_module", dropna=True, size=5);


# In[37]:


studentModuleLengths_df = studentInfo_df.merge(courses_df, on = ['code_module', 'code_presentation'], how='left')
studentModuleLengths_df = studentModuleLengths_df[['id_student', 'module_presentation_length']].groupby('id_student').median()
studentModuleLengths_df = studentModuleLengths_df.reset_index()


# In[38]:


show_basic_info(studentModuleLengths_df)


# In[39]:


sns.countplot(studentModuleLengths_df.module_presentation_length);


# In[40]:


studentRegistration_df['unregistered'] = np.where(pd.isnull(studentRegistration_df.date_unregistration), 0, 1)
studentRegistration_df['registered'] = np.where(pd.isnull(studentRegistration_df.date_unregistration), 0, 1)


# In[41]:


studentRegistration_df['register_days'] = (np.where(pd.isnull(studentRegistration_df.date_registration), 0, 
                                          studentRegistration_df.date_registration)).astype(int)
studentRegistration_df['unregister_days'] = (np.where(pd.isnull(studentRegistration_df.date_unregistration), 0, 
                                            studentRegistration_df.date_unregistration)).astype(int)
studentRegDays_df = studentRegistration_df[['id_student', 'register_days', 
                                   'unregister_days']].groupby(['id_student']).mean()
studentRegDays_df = studentRegDays_df.reset_index()
studentRegDays_df.head()


# In[42]:


studentInterest_df = studentRegistration_df[['id_student', 'registered', 'unregistered']].groupby(['id_student']).sum()
studentInterest_df = studentInterest_df.reset_index()


# In[43]:


show_basic_info(studentInterest_df)


# In[44]:


studentInterest_df[['registered', 'unregistered']].boxplot();


# In[45]:


studentInterest_df.unregistered.hist();


# In[46]:


studentAssessment_df['score'] = (np.where(pd.isnull(studentAssessment_df.score), 0, studentAssessment_df.score)).astype(int)


# In[47]:


studentAssessment_df['assessment_mean'] = studentAssessment_df['score'].groupby(studentAssessment_df['id_assessment']) .transform('mean')


# In[48]:


studentAssessment_df['score_std'] = studentAssessment_df.score/studentAssessment_df.assessment_mean


# In[49]:


studentScoring_df = studentAssessment_df[['id_student', 
                                          'score_std']].groupby(['id_student']).median()
studentScoring_df = studentScoring_df.reset_index()
studentScoring_df.info()


# In[50]:


studentScoring_df.score_std.hist();


# In[51]:


studentVle_df = pd.read_csv('/Users/apurvajain/Desktop/FinalProject/anonymisedData/studentVle.csv')


# In[56]:


show_basic_info(studentVle_df)


# In[57]:


studentVle_df = studentVle_df.merge(vle_df, on = 'id_site', how = 'left')


# In[58]:


sns.countplot(studentVle_df.activity_type)
plt.xticks(rotation = 90)


# In[59]:


studentInteractivity_df = studentVle_df[['id_student', 
                                     'activity_type', 'sum_click']].groupby(['id_student', 'activity_type']).mean()
studentInteractivity_df = studentInteractivity_df.reset_index()
studentInteractivity_df.head()


# In[60]:


import missingno as msno


# In[61]:


studentInteractivity_df = studentInteractivity_df.pivot(index='id_student', 
                                                    columns='activity_type', values='sum_click')
studentInteractivity_df = studentInteractivity_df.reset_index()
msno.matrix(studentInteractivity_df)
studentInteractivity_df = studentInteractivity_df.fillna(0)
studentInteractivity_df.info()


# In[62]:


studentInteractivity_df = studentInteractivity_df[['id_student', 'forumng', 'homepage', 'oucollaborate',
       'oucontent', 'ouwiki', 'page', 'questionnaire', 'quiz',
       'resource', 'subpage', 'url']]


# In[63]:


dataset = studentPerformance_df.merge(studentModuleLengths_df, 
                                    on = 'id_student', how='left')
dataset = dataset.merge(studentInterest_df, 
                                    on = 'id_student', how='left')
dataset = dataset.merge(studentRegistration_df[['id_student', 'register_days']], 
                                    on = 'id_student', how='left')
dataset = dataset.merge(studentScoring_df, 
                                    on = 'id_student', how='left')
dataset = dataset.merge(studentInteractivity_df, 
                                    on = 'id_student', how='left')
dataset.info()


# In[64]:


msno.bar(dataset)
dataset = dataset.fillna(0)


# In[65]:


plt.matshow(dataset.corr())


# In[66]:


from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import k_means
from sklearn.metrics import silhouette_score


# In[67]:


# Scaling the data to bring into one range
sc = RobustScaler()


# In[68]:


dataset.score_std.plot.box()


# In[ ]:




