#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


data=pd.read_csv("udemy_courses.csv.xls")
data.head(2)


# In[3]:


#gettting the dimensinality of the data
data.shape


# In[4]:


# checking for null values, duplicates in the data
data.isnull().sum()
#all columns giving 0 means no null value found in any of the column


# In[5]:


data[data.duplicated()] #getting those specific duplicates fron data -->dataframe


# In[6]:


data=data.drop_duplicates()


# In[7]:


data[data.duplicated()]# nothing will pe  printed


# STEP 1 every time someone enters to the recommender system the System should be able to provide 
# some initial recommendation to them hence 1st step of the model development would 
# be to just get the top courses as the recommendation 
# for the purpose of which  a popularity based recommender system needs to ebe built 
# to which the approach would be to develop an new column in the existing dataframe as pop_score
# and for this case I have used number of subscribers and number of reviews as the 2 parameters to get my 
# pop_score

# In[8]:


def pop_rem(df,top_n=5):
    data['pop_score']=0.6*data['num_subscribers']+ 0.4*data['num_reviews'] # new column created as pop_score
    df_sorted=data.sort_values(by='pop_score',ascending=False)
    recommended_courses=df_sorted[['course_title']].head(top_n)
    return recommended_courses


# In[11]:


pop_rem(data,5)


# In[12]:


import neattext.functions as nfx
data['course_title']
# removing spl characters


# In[13]:


data['course_title']=data['course_title'].apply(nfx.remove_stopwords)
data['course_title']=data['course_title'].apply(nfx.remove_special_characters)
data.sample(5)


# In[15]:


# now forming a new column by concatiinating course_title and subject
data['title_subject']=data['course_title']+" "+ data['subject']


# In[16]:


data.sample(5)


# In[18]:


from sklearn.feature_extraction.text import CountVectorizer #would convert textual data to numeric representation
cv=CountVectorizer(max_features=3000) #jsust top 3000 words
vectors=cv.fit_transform(data['title_subject']).toarray()  #sparse amtrix---> array


# In[19]:


vectors[0]


# In[55]:


# checking the features on the basis of which vectors were created
len(cv.get_feature_names_out())


# In[56]:


# now we have data in the form of vectors in the space we would use cosine similarity 
# to find the similarity with each vector 
from sklearn.metrics.pairwise import cosine_similarity


# In[57]:


similarity=cosine_similarity(vectors)


# In[58]:


#testing tmodel
# getting similarity for the 1st course
sorted(list(enumerate(similarity[0])),reverse=True,key=lambda x:x[1])[1:6]  #enumerate-->returns a tuple of index and value of similarity score  for all courses 
# key=lambda x:x[1]--->means sorting done on the 1st  elemnt


# In[61]:


# defining a recommend function
def recommend(course):
    #getting the index number of this course
    course_index = data[data['course_title']==course].index[0]
    # now finding index of similar courses 
    distances=similarity[course_index]
#     gettting a tuplle of similar courses
    courses_list = sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:7]

    for i in courses_list:
        print(data.iloc[i[0]]['course_title'])

    


# In[62]:


recommend('Complete Investment Banking Course 2017')


# In[63]:


import pickle
pickle.dump(data,open('courses.pkl','wb'))


# In[64]:


data['course_title'].values


# In[67]:


# creating this as a dictionary
pickle.dump(data.to_dict(),open('courses_dict.pkl','wb'))


# In[68]:


pickle.dump(similarity,open("similarity.pkl",'wb'))


# In[ ]:




