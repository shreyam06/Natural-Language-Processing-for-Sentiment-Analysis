
import nltk
import random
from nltk.corpus import movie_reviews

import pickle

from sklearn import naive_bayes

documents= []
for category in movie_reviews.categories():
    for filed in movie_reviews.fileids(category):
        words = list(movie_reviews.words(filed))
        documents.append((words,category))


print(documents[1])

random.shuffle(documents)

all_words = []
for w in movie_reviews.words():
    all_words.append(w.lower())

all_words = nltk.FreqDist(all_words)
print(all_words.most_common(15))

word_features = list(all_words.keys())[: 3000]

def find_features (documents):
    words = set(documents)
    features = {}
    for w in word_features:
        features[w] = ( w in words ) 
    return features 

print((find_features(movie_reviews.words("neg/cv019_16117.txt"))))
featuresets = [(find_features(rev), category) for (rev, category) in documents]

featuresets = []
for (rev , category ) in documents:
    rev_of = find_features(rev)
    featuresets.append((rev_of,category))


training_set = featuresets[:1900]  


testing_set =  featuresets[1900:]


classifier = nltk.NaiveBayesClassifier.train(training_set)
# classifier_f =  open("naivebayes.pickle","rb")

# classifier = pickle.load(classifier_f)

# classifier_f.close()

print("classifier accuracy " , (nltk.classify.accuracy(classifier,testing_set))*100)

classifier.show_most_informative_features(15)


# save_classifier = open("naivebayes.pickle","wb")
# pickle.dump(classifier, save_classifier)
# save_classifier.close()