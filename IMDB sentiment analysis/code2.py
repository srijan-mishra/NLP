import numpy as np
import pandas as pd
from bs4 import BeautifulSoup #to remove html markups
import re #to remove anything other than letters
from nltk.corpus import stopwords  #to remove stopwords like a, an, the, is etc.
from functions import review_to_wordlist
from functions import review_to_sentences
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
import nltk.data
import logging
from gensim.models import word2vec






train = pd.read_csv( "labeledTrainData.tsv", header=0, delimiter="\t", quoting=3 )
test = pd.read_csv( "testData.tsv", header=0, delimiter="\t", quoting=3 )
unlabeled_train = pd.read_csv( "unlabeledTrainData.tsv", header=0, delimiter="\t", quoting=3 )

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

sentences = []
for review in train["review"]:
    sentences += review_to_sentences(review, tokenizer)

for review in unlabeled_train["review"]:
    sentences += review_to_sentences(review, tokenizer)


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',level=logging.INFO)


# Set values for various parameters
num_features = 300    # Word vector dimensionality
min_word_count = 50   # Minimum word count
num_workers = 4       # Number of threads to run in parallel
context = 10          # Context window size
downsampling = 1e-3   # Downsample setting for frequent words


model = word2vec.Word2Vec(sentences, workers=num_workers,size=num_features, min_count = min_word_count, window = context, sample = downsampling)


model.init_sims(replace=True)
model_name = "300features_50minwords_10context"
model.save(model_name)
