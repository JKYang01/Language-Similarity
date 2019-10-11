
 #!/usr/bin/env python
# coding: utf-8

## The explaination of the data processing
## In this processing I do not aggregate the data.
# it is dataset of the monthly cosine similarity for each user per month in Gravity Spy. 
# For example, in month 2, I generate the commentdata of the user and the community just depend on the comments in month 2
# thus, the data in month on is not included in the community corpus and user's corpus of month 2. 
# This dataset is in the long format with each row being a user | month,  
# and include all the attributes you listed in the original file you sent. 


## import the packages

# the pacakge for data preprocessing 
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
import re

# package for storing the data 
import pickle

# language modeling sklearn package 
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer


# s_hmonth.pkl is generated after running 'Generate the list of tables(GravitySpy).py'
s_month_pkl = open('s_hmonth.pkl', 'rb')

# list of tables which has the comment splitted with commas 
df_list = pickle.load(s_month_pkl)


# check the catergories 
df_list[0].columns


# the commnuity corpus here is a single string which is the collection of 
# user comments in the rest of the community and join together with white space 
# the processing will go in each row which means each user
# the x here is for the user we are going to processing 
# the feed back would be the comments of the users in the rest of the community expect user x

def corpus(all_comment,x):
    if len(x)==1:
        com_corpus = x
    else: 
        com_corpus = [i for i in all_comment if i not in x]
    
    return ' '.join(com_corpus)


# This fuction is to exclude new words which intriduced by the user in user's corpus
# the filtered comment will be vectorized and compaired to the commnunity corpus 
# the column that store the filtered comment for each user named 'vector'

def exclude_newwords(user_comment,community_corpus):
    user_words = user_comment.split()
    community_words = community_corpus.split()
    new_words = list(set(user_words).difference(set(community_words))) 
    exclude_new_words = [i for i in user_words if i not in new_words]
    return ' '.join(exclude_new_words)
   


# This fuction build the community corpus for each user 
# including newwords, vecotr,community_corpus

def build_dataframe(df):
    df['post']=df['filtered_words2'].apply(lambda x : ' '.join(x.split(',')))
    df['n_post']=df['filtered_words2'].apply(lambda x: len(x.split(',')))
    df['n_unigram'] =  df['post'].apply(lambda x: len(x.split()))
    df['n_unique'] = df['post'].apply(lambda x: len(set(x.split())))
    all_comment = df['post'].tolist()
    df['community_corpus'] = df['post'].apply(lambda x: corpus(all_comment,x))
    df['vector']=[exclude_newwords(x,y) for x,y in zip(df['post'],df['community_corpus'])]
    df['n_newword'] = [len(y.split())-len(x.split())for x,y in zip(df['vector'],df['post'])]
    return df[['comment_user_login', 'comment_user_id', 'month','n_post','n_unigram',
               'n_unique','n_newword','vector','community_corpus']]


# the list of dataframe which has multiple catergories  

list_of_dataframe=[]
for df in df_list:
    ndf = build_dataframe(df)
    list_of_dataframe.append(ndf) 


# check the data frame
list_of_dataframe[-2]

# store the reult in case 
with open ('number&corpus_dataframe.pkl','wb') as f:
    pickle.dump(list_of_dataframe,f)


# # cosim similarity for all comments

# set the word tokens
def tokens(text):
    tokens = text.split()
    return tokens

# the vectorzing algorithom
tf_vect = CountVectorizer(tokenizer = tokens, stop_words = None)

# get the cosine similairty score 
def cosine_similarity_score(x,y):
    train_set = [x,y]
    vect = tf_vect.fit_transform(train_set)
    cos_sim = cosine_similarity(vect[0:1], vect)
    return cos_sim[0][1]


# this function is to generate a new column to store the result
# the name of the column is the cosine_similarity

def append_cosim (table):
    #table.replace('',np.nan,inplace=True)
    table['cosine_similarity'] = [cosine_similarity_score(x,y) for x,y  in zip(table['vector'].astype(str),table['community_corpus'].astype(str))]
    df = table.drop(columns = ['vector','community_corpus']).copy() 
    return df

# list of final dataframe of each month
list_fdf=[]
for df in list_of_dataframe:
    fdf=append_cosim(df)
    list_fdf.append(fdf)



# the final table of all 24 months by concat them together
f_df = pd.concat(list_fdf).reset_index(drop=True)



# check the data 
print(len(f_df))

f_df.head()


# store the the result of data 

f_df.to_csv('GravitySpy_user_in_month.csv',index=False)


