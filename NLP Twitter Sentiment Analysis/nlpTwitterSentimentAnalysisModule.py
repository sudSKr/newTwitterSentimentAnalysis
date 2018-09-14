'''@author: SuD
   This module is used for loading our classifiers and data sets and to use them in out module where we live stream twitter data
'''
import random
import pickle
from nltk.tokenize import word_tokenize
from nltk.classify import ClassifierI
from statistics import mode

documents = [] # Will store our categorized tweets
all_words = [] # Will store the tokenized words from our categorized tweets
# the class VoteClassifier is used for classifing the tweets and finding the confidence 
class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers
    def classify(self, features): #classified by taking the votes and taking the classification for which most voted
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)
    def confidence(self, features): #calculated by picking the majority votes and dividing them with the total votes
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf    
# here we load our tweets from our pickle          
tweets_f = open("tweets.pickle","rb")
documents = pickle.load(tweets_f)
tweets_f.close()
# here we load our word features from our pickle
word_features_f = open("word_features5k.pickle","rb")
word_features = pickle.load(word_features_f)
word_features_f.close()
# we convert the words into features which is basically create a array of words 
def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    return features
#we convert our tweets into features
featuresets = [(find_features(rev), category) for (rev, category) in documents]
random.shuffle(featuresets) # we shuffle the data so that we don't make our data bias
# We have 10664 categorized tweets so we take 9000 as training data and the rest as testing data
training_set = featuresets[:9000]
testing_set = featuresets[9000:]
# here we load our stored original naive bayes classifier from our pickle
ONB_classifier_f = open("originalnaivebayes.pickle","rb")
ONB_classifier = pickle.load(ONB_classifier_f)
ONB_classifier_f.close()
# here we load our stored multinomial naive bayes classifier from our pickle
MNB_classifier_f = open("multinomialnaivebayes.pickle","rb")
MNB_classifier = pickle.load(MNB_classifier_f)
MNB_classifier_f.close()
# here we load our stored bernoulli naive bayes classifier from our pickle
BNB_classifier_f = open("bernoullinaivebayes.pickle","rb")
BNB_classifier = pickle.load(BNB_classifier_f)
BNB_classifier_f.close()
# here we load our stored logistic regressionn classifier from our pickle
LogisticRegression_classifier_f = open("logisticregression.pickle","rb")
LogisticRegression_classifier = pickle.load(LogisticRegression_classifier_f)
LogisticRegression_classifier_f.close()
# here we load our stored stochastic gradient descent classifier from our pickle
SGDClassifier_classifier_f = open("stochasticgradientdescent.pickle","rb")
SGDClassifier_classifier = pickle.load(SGDClassifier_classifier_f)
SGDClassifier_classifier_f.close()
# here we load our stored linear Suport Vector Machines classifier from our pickle
LinearSVC_classifier_f = open("linearsupportvectormachines.pickle","rb")
LinearSVC_classifier = pickle.load(LinearSVC_classifier_f)
LinearSVC_classifier_f.close()
# here we load our stored nu Suport Vector Machines classifier from our pickle
NuSVC_classifier_f = open("nusupportvectormachines.pickle","rb")
NuSVC_classifier = pickle.load(NuSVC_classifier_f)
NuSVC_classifier_f.close()

voted_classifier = VoteClassifier(ONB_classifier,
                                  MNB_classifier,
                                  BNB_classifier,
                                  LogisticRegression_classifier,
                                  SGDClassifier_classifier,
                                  LinearSVC_classifier,
                                  NuSVC_classifier)
#this method sentiment is used for finding the sentiment which currently has only two classification pos and neg and to return the confidence
def sentiment(text):
    feats = find_features(text)
    return voted_classifier.classify(feats),voted_classifier.confidence(feats)






