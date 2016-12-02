from gensim.models import Word2Vec
from functions import review_to_wordlist
from functions import getAvgFeatureVecs
import pandas as pd
from sklearn.cluster import KMeans
import time
from functions import create_bag_of_centroids

#using model for analysis
model = Word2Vec.load("300features_40minwords_10context")

#The number of rows in syn0 is the number of words in the model's vocabulary,
#and the number of columns corresponds to the size of the feature vector, which
#we set in Part 2.  Setting the minimum word count to 40 gave us a total vocabulary
#of 16,492 words with 300 features apiece. Individual word vectors can be accessed
#in the following way:
# model["flower"]




start = time.time()
word_vectors = model.syn0
num_clusters = word_vectors.shape[0] / 5

kmeans_clustering = KMeans( n_clusters = num_clusters )
idx = kmeans_clustering.fit_predict( word_vectors )
end = time.time()
elapsed = end - start
print "Time taken for K Means clustering: ", elapsed, "seconds."


word_centroid_map = dict(zip( model.index2word, idx ))



clean_train_reviews = []
for review in train["review"]:
    clean_train_reviews.append( review_to_wordlist( review,remove_stopwords=True ))

clean_test_reviews = []
for review in test["review"]:
    clean_test_reviews.append( review_to_wordlist( review,remove_stopwords=True ))


train_centroids = np.zeros( (train["review"].size, num_clusters), dtype="float32" )

counter = 0
for review in clean_train_reviews:
    train_centroids[counter] = create_bag_of_centroids( review, word_centroid_map )
    counter += 1

test_centroids = np.zeros(( test["review"].size, num_clusters),dtype="float32" )

counter = 0
for review in clean_test_reviews:
    test_centroids[counter] = create_bag_of_centroids( review, word_centroid_map )
    counter += 1


forest = RandomForestClassifier(n_estimators = 100)
forest = forest.fit(train_centroids,train["sentiment"])
result = forest.predict(test_centroids)
output = pd.DataFrame(data={"id":test["id"], "sentiment":result})
output.to_csv( "BagOfCentroids.csv", index=False, quoting=3 )
