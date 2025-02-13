#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


mv = pd.read_csv("D:/Recommendation system/tmdb_5000_movies.csv")
cr = pd.read_csv("D:/Recommendation system/tmdb_5000_credits.csv")


# In[3]:


mv.head(1)


# In[4]:


cr.head(1)


# In[5]:


mv = mv.merge(cr, on='title')


# In[6]:


mv.head(1)


# In[ ]:





# In[7]:


mv = mv[["movie_id", "crew" , "genres" , "title", "cast" , "keywords" , "overview"  ]]


# In[8]:


mv.isnull().sum()


# In[9]:


mv.dropna(inplace = True)


# In[10]:


mv.isnull().sum()


# In[11]:


mv.iloc[0].genres


# In[12]:


import ast


# In[13]:


def catagory(obj):
    list =[]
    for i in ast.literal_eval(obj):   #ast.literal_eval since our list is in string formate we use this to convert string - list
        list.append(i['name'])
    return list


# In[14]:


mv['genres'] = mv['genres'].apply(catagory)


# In[15]:


def name_dir(obj):
    list = []
    for i in ast.literal_eval(obj):
        if i['job'] == "Director":
            list.append(i['name'])
            
    return list


# In[16]:


mv['crew'] = mv['crew'].apply(name_dir)


# In[17]:


mv.rename(columns={'crew':'Director', 'title':'Movie_name'}, inplace = True)


# In[18]:


mv.head(1)


# In[19]:


def cast_name(obj):
    names = []
    counter = 0
    for i in ast.literal_eval(obj):
        names.append(i["character"])
        counter +=1
        if counter == 3 :
            break
    return names


# In[20]:


mv['cast'] = mv['cast'].apply(cast_name)


# In[21]:


mv.head(1)


# In[22]:


mv.iloc[0].keywords


# In[23]:


def keyword(obj):
    key = []
    for i in ast.literal_eval(obj):
        key.append(i['name'])
    return key


# In[24]:


mv['keywords'] = mv['keywords'].apply(keyword)


# In[25]:


mv.head(1)


# In[26]:


mv['genres'] = mv['genres'].apply(lambda x :[i.replace(" ", "") for i in x])
mv['cast']  = mv['cast'].apply(lambda x :[i.replace(" ", "") for i in x])


# In[27]:


mv.head(1)


# In[28]:


mv['Director'] = mv['Director'].apply(lambda x :[i.replace(" ", "") for i in x])


# In[29]:


mv.head(11)


# In[30]:


mv['tags'] = mv['Director'] + mv['genres'] + mv['cast'] + mv['keywords'] 


# In[31]:


mv.head(1)


# In[32]:


df = mv[['movie_id', 'Movie_name', 'tags']]


# In[33]:


df.head(1)


# In[34]:


df['tags']=df['tags'].apply(lambda x:" ".join(x))


# In[35]:


df['tags']=df['tags'].apply(lambda x:x.lower())


# In[36]:


df.head(1)


# In[37]:


#converting texts to vectors to find realation between movies 

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=2000 , stop_words="english")


# In[38]:


vectors =  cv.fit_transform(df['tags']).toarray()


# In[39]:


vectors[0]


# In[40]:


#finding the shortest distance btw movie (using eucledian distance or cosine distance )

from sklearn.metrics.pairwise import cosine_similarity 
 #agar koi movie kisi aur movie se similar hai to distance willl be near 1 else 0 
 # 0-->1
                                         
similarity  = cosine_similarity(vectors)
similarity.shape


# In[41]:


sorted(list(enumerate(similarity[0])) ,reverse = True ,key = lambda x:x[1])[1:6]


# In[53]:


def recommendation(movie):
    if movie not in df['Movie_name'].values:
        print("No such movie")
        return

    # Proceed if the movie exists
    index = df[df['Movie_name'] == movie].index[0]
    dis = similarity[index]
    mvlist = sorted(list(enumerate(dis)), reverse=True, key=lambda x: x[1])[1:6]
    return mvlist
    
    


# In[54]:


list_movie = recommendation("Spider-Man")


# In[57]:


list_movie


# In[55]:


movie_name_list = []
for i in list_movie:
    movie_name_list.append(df.iloc[i[0]]['Movie_name'])


# In[56]:


movie_name_list


# In[46]:


import pickle


# In[47]:


pickle.dump(similarity,open('C:/Users/harsh/Desktop/flask/similarity.pkl','wb'))


# In[48]:


df.to_csv('C:/Users/harsh/Desktop/flask/movie_list.csv', index=False)


# In[49]:


movie_name = []
for i in df['Movie_name'].drop_duplicates():
    movie_name.append(i)
    
len(movie_name)


# In[50]:


arr = np.array(movie_name)


# In[51]:


arr.sort


# In[52]:


df.head()


# In[ ]:




