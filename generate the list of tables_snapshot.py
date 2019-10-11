#!/usr/bin/env python
# coding: utf-8
### The preprocessing is also done by python
### the result need to be compaired to the result of the original code which is wirtten by R
# In[1]:


import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
import re 


# import the data from csv file
df1 = pd.read_csv(r'C:\Users\wangtao\Documents\summer_intern_gravity spy\original data\snapshot_prep2.csv')
# this step is import the stopword list from NLTK 
english_stopword = stopwords.words('english')
# see the data strucutre
df1.tail(5)

# check the data and there are 2012-02 to 2016-12
df1['created_at']


# read the file mallet.txt and import the words in this file as stopword_list 
with open(r'C:\Users\wangtao\Documents\summer_intern_gravity spy\original data\mallet.txt','r')as f:
    stopword_list = f.read().split('\n')
    # add in the NLTK stopwords 
    stopword_list.extend(english_stopword)



# just extract the columns we need 
df2 = df1[['user_name','comment','created_at']].copy()



## see wether the table cotains null 
miss = df2.isnull()
for c in miss.columns.values.tolist():
    print(c)
    print(miss[c].value_counts())
    print('\n')


# drop the Nan and which do not have comments
df3=df2.dropna(how='any', axis=0)

## Build the funtion of generating the tables
list_of_date = ['2012\-12\-','2013\-01\-','2013\-02\-','2013\-03\-',
                '2013\-04\-','2013\-05\-','2013\-06\-','2013\-07\-',
                '2013\-08\-','2013\-09\-','2013\-10\-','2013\-11\-',
                '2013\-12-','2014\-01\-','2014\-02\-','2014\-03\-',
                '2014\-04\-','2014\-05\-','2014\-06\-','2014\-07\-',
                '2014\-08\-','2014\-09\-','2014\-10\-','2014\-11\-',
                '2014\-12-','2015\-01\-','2015\-02\-','2015\-03\-',
                '2015\-04\-','2015\-05\-','2015\-06\-','2015\-07\-',
                '2015\-08\-','2015\-09\-','2015\-10\-','2015\-11\-',
                '2015\-12-','2016\-01\-','2016\-02\-']
                
## function to generate table for each months
## collect each user's comment in every moths in the list
## using groupby to summarizing each user's comment every month 
def month_collection(df3,list_of_date,list_of_tables = None):
    list_of_tables = []
    n = 0
    for string in list_of_date:
        month = df3['created_at'].str.contains(string)
        table = df3[month].copy()
        table1 = table.groupby(['user_name'])['comment'].apply(','.join).reset_index()
        n +=1
        table1['month'] = str(n)
        #table1['filtered_words2'].apply(lambda x:''.join(unique(x)))
        list_of_tables.append(table1)
    return list_of_tables
    

## the data of every month is aggragate forexample the data of month2 is two months' data "2016-03"and "2016-04" 

tables = month_collection(df3,list_of_date)


# take month1 as a example
# the result is stored in snap_month.pkl

# check the data 
tables[1].head()
# the last table in the list
len(tables[38])



