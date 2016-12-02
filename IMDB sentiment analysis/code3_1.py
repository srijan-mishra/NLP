from gensim.models import Word2Vec
from functions import review_to_wordlist
from functions import getAvgFeatureVecs
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

#using model for analysis
model = Word2Vec.load("300features_40minwords_10context")

#The number of rows in syn0 is the number of words in the model's vocabulary,
#and the number of columns corresponds to the size of the feature vector, which
#we set in Part 2.  Setting the minimum word count to 40 gave us a total vocabulary
#of 16,492 words with 300 features apiece. Individual word vectors can be accessed
#in the following way:
# model["flower"]


num_features=300

train = pd.read_csv( "labeledTrainData.tsv", header=0, delimiter="\t", quoting=3 )
test = pd.read_csv( "testData.tsv", header=0, delimiter="\t", quoting=3 )

clean_train_reviews = []
for review in train["review"]:
    clean_train_reviews.append( review_to_wordlist( review,remove_stopwords=True ))

trainDataVecs = getAvgFeatureVecs( clean_train_reviews, model, num_features )

clean_test_reviews = []
for review in test["review"]:
    clean_test_reviews.append( review_to_wordlist( review,remove_stopwords=True ))

testDataVecs = getAvgFeatureVecs( clean_test_reviews, model, num_features )


forest = RandomForestClassifier( n_estimators = 100 )
forest = forest.fit( trainDataVecs, train["sentiment"] )
result = forest.predict( testDataVecs )
output = pd.DataFrame( data={"id":test["id"], "sentiment":result} )
output.to_csv( "Word2Vec_AverageVectors.csv", index=False, quoting=3 )
