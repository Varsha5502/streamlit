import os

from utils.b2 import B2
from dotenv import load_dotenv
import streamlit as st

import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from itertools import islice

from nltk.tokenize import RegexpTokenizer
import string

import matplotlib.pyplot as plt
import pandas as pd

REMOTE_DATA = 'twitter_subset.csv'

load_dotenv()

# load Backblaze connection
b2 = B2(endpoint=os.environ['B2_ENDPOINT'],
        key_id=os.environ['B2_keyID'],
        secret_key=os.environ['B2_applicationKey'])

def get_data():
    # collect data frame of reviews and their sentiment
    b2.set_bucket(os.environ['B2_BUCKETNAME'])
    df = b2.get_df(REMOTE_DATA)
    
    return df

df = get_data()
st.title("Let's analyze the top 10 words that are being used frequently by users on twitter")

data = df.copy()
data.columns=["target","ids","date","flag","user","text"]

data = data[:10000]
data['text']=data['text'].str.lower()
stopwords_list = stopwords.words('english')

STOPWORDS = set(stopwords_list)
def cleaning_stopwords(text):
    return " ".join([word for word in str(text).split() if word not in STOPWORDS])

data['text'] = data['text'].apply(lambda text: cleaning_stopwords(text))

def cleaning_email(data):
    return re.sub('@[^\s]+', ' ', data)
data['text']= data['text'].apply(lambda x: cleaning_email(x))

def cleaning_URLs(data):
    return re.sub('((www\.[^\s]+)|(https?://[^\s]+))',' ',data)
data['text'] = data['text'].apply(lambda x: cleaning_URLs(x))

english_punctuations = string.punctuation
def cleaning_punctuations(text):
    translator = str.maketrans('', '', english_punctuations)
    return text.translate(translator)
data['text'] = data['text'].apply(lambda text: cleaning_punctuations(text))

def cleaning_numbers(data):
    return re.sub('[0-9]+', '', data)
data['text'] = data['text'].apply(lambda x: cleaning_numbers(x))

tokenizer = RegexpTokenizer(r'\w+')
data['text'] = data['text'].apply(tokenizer.tokenize)

word_list = data["text"].values

freq_dict = {}

for i in word_list:
    for j in i:
        if j in freq_dict:
            freq_dict[j] += 1
        else:
            freq_dict[j] = 1

sorted_dict = dict(sorted(freq_dict.items(), key=lambda x: x[1], reverse=True))
top_10 = dict(islice(sorted_dict.items(), 10))

fig = plt.figure(figsize=(10, 6))
plt.bar(top_10.keys(), top_10.values(), color='orange')
plt.title('Overall Top 10 Most Frequent Words')
plt.xlabel('Words')
plt.ylabel('Frequency')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

st.pyplot(fig)

st.dataframe(df.head(25))