import numpy as np
import pandas as pd
from bs4 import BeautifulSoup #to remove html markups
import re #to remove anything other than letters
from nltk.corpus import stopwords  #to remove stopwords like a, an, the, is etc.
from functions import review_to_words
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier




train = pd.read_csv("labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)

num_reviews = train["review"].size
clean_train_reviews = []
for i in xrange( 0, num_reviews ):
    if( (i+1)%1000 == 0 ): print "Review %d of %d\n" % ( i+1, num_reviews )
    clean_train_reviews.append( review_to_words( train["review"][i] ))


# Initialize the "CountVectorizer" object, which is scikit-learn's bag of words tool.
vectorizer = CountVectorizer(analyzer = "word",tokenizer = None, preprocessor = None,stop_words = None, max_features = 5000)
train_data_features = vectorizer.fit_transform(clean_train_reviews)
# fit_transform() does two functions: First, it fits the model
# and learns the vocabulary; second, it transforms our training data
# into feature vectors. The input to fit_transform should be a list of
# strings.
train_data_features = train_data_features.toarray() #converting to arrays


#to see vocab list
vocab = vectorizer.get_feature_names()
dist = np.sum(train_data_features, axis=0)
for tag, count in zip(vocab, dist):
    print count, tag


#Training
forest = RandomForestClassifier(n_estimators = 100)
forest = forest.fit( train_data_features, train["sentiment"] )


test = pd.read_csv("testData.tsv", header=0, delimiter="\t",quoting=3 )

num_reviews = len(test["review"])
clean_test_reviews = []

for i in xrange( 0, num_reviews ):
    if( (i+1)%1000 == 0 ): print "Review %d of %d\n" % ( i+1, num_reviews )
    clean_test_reviews.append( review_to_words( test["review"][i] ))

test_data_features = vectorizer.transform(clean_test_reviews)
test_data_features = test_data_features.toarray()
result = forest.predict(test_data_features)
output = pd.DataFrame( data={"id":test["id"], "sentiment":result} )

output.to_csv( "Bag_of_Words_model.csv", index=False, quoting=3 )
