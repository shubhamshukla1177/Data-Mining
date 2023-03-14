#!/usr/bin/env python
# coding: utf-8

# In[146]:


import numpy as np
import pandas as pd
import re
import string
import matplotlib.pyplot as plt
from collections import Counter

import nltk
from nltk.tokenize import word_tokenize
nltk.download("stopwords")
from nltk.corpus import stopwords 
from nltk.stem import SnowballStemmer

from sklearn.feature_extraction.text import TfidfVectorizer


# In[147]:


def csv_to_df(file_name,col_name):
    return pd.read_csv(file_name,delimiter=',',header=None,names=col_name,index_col=None,quotechar='"',quoting=0,escapechar=None,doublequote=True,delim_whitespace=False)


# In[148]:


def data_cleaning(col):
    lower_replace= col.lower().replace('\\n',' ').replace('\\',' ')
    rm_numbers= re.sub(r'\d+','', lower_replace) 
    translator= str.maketrans('', '', string.punctuation)
    rm_punctuation= rm_numbers.translate(translator)
    rm_whitespace= " ".join(rm_punctuation.split())
    return rm_whitespace


# In[149]:


def rm_stopwords(col):
    stopword_list = stopwords.words("english")
    #col_tokens= tokenizer.tokenize(col)
    col_tokens= word_tokenize(col)
    col_tokens= [token.strip() for token in col_tokens]
    clean_tokens= [token for token in col_tokens if token.lower() not in stopword_list]    
    return " ".join(clean_tokens)


# In[150]:


def stem_words(col):
    snowballstemmer= SnowballStemmer(language='english')
    stem= " ".join([snowballstemmer.stem(word) for word in col.split()])
    return stem


# In[151]:


def data_preprocess(df,col):
    clean= list(map(data_cleaning,df[col]))
    stopwords= list(map(rm_stopwords,clean))
    stem= list(map(stem_words,stopwords))
    return stem


# In[152]:


train_filename='new_train_data.csv'
train_fileheader=['Sentiment', 'Reviews']
train_df=csv_to_df(train_filename,train_fileheader)
train_df2=data_preprocess(train_df,'Reviews')


# In[153]:


#Creating L2-normalized sparse matrices with TF-IDF values 
#And then Finding out cosine similarity 
#Using both train and test vector matrices
def cosine_similarities(train_col_list, test_col_list):
    vectorizer= TfidfVectorizer(norm = "l2")
    train_matrix= vectorizer.fit_transform(train_col_list)
    test_matrix= vectorizer.transform(test_col_list)
    train_matrix_t= np.transpose(train_matrix)
    cosine= np.dot(test_matrix, train_matrix_t).toarray()
    return cosine


# In[154]:


#argsort function preserve the indices to let training lables be easily referenced
def k_nearest_neighbours(cosine, k):
    return np.argsort(-cosine)[:k]


# In[155]:


def predict_review_sentiment(knn, sentiments):
    p_sentiment= 0
    n_sentiment= 0
    for n in knn:
        if int(sentiments[n]) == 1:
            p_sentiment += 1
        else:
            n_sentiment += 1
    
    if p_sentiment >= n_sentiment:
        return "+1"
    else:
        return "-1"


# In[158]:


train_filename='new_train_data.csv'
train_fileheader=['Sentiment', 'Reviews']
test_filename='new_test_data.csv'
test_fileheader=['Reviews']

train_df= csv_to_df(train_filename,train_fileheader)
test_df= csv_to_df(test_filename,test_fileheader)

train_preprocess= data_preprocess(train_df,'Reviews')
test_preprocess= data_preprocess(test_df,'Reviews')

#np.sqrt(18000) is 134.16
k = 135
train_sentiment= list(train_df['Sentiment'])
test_sentiment= list()

cosine = cosine_similarities(train_preprocess, test_preprocess)

for x in cosine:
    knn= k_nearest_neighbours(x, k)
    prediction= predict_review_sentiment(knn, train_sentiment)
    test_sentiment.append(prediction)


# In[200]:


#writing sentiment prediction to a csv file
write_sentiments= open("output.csv", "w")
write_sentiments.writelines("%s\n" % i for i in test_sentiment)
write_sentiments.close()


# In[202]:


sentiments= Counter(test_sentiment).keys()
sentiments_count= Counter(test_sentiment).values()
plt.pie(sentiments_count,labels= sentiments_count,autopct= "%1.2f%%")
plt.legend(sentiments,title= "Sentiments",loc= "center",bbox_to_anchor= (1, 0, 1, 2))
plt.title("Yelp Data Reviews Sentiments")
plt.show()


# In[ ]:




