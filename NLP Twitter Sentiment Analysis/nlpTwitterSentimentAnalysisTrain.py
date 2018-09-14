'''@author: SuD
   This module is used for training our classifed tweets and pickling our classifiers and data sets for use in applications
'''
import nltk
import random
import pickle
from nltk.tokenize import word_tokenize
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.svm import LinearSVC, NuSVC
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
# Here we call our aquired tweets which has already been categorized into positive and negative
short_pos = open("short_reviews/positive.txt","r").read()
short_neg = open("short_reviews/negative.txt","r").read()
# We can restrict the word types according to our preference such as adverb
allowed_word_types = ["J","R","V"]
# Append our categorized tweets to documents
for r in short_pos.split('\n'):
    documents.append( (r, "pos") )
    words = word_tokenize(r)
    pos = nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())
for r in short_neg.split('\n'):
    documents.append( (r, "neg") )
    words = word_tokenize(r)
    pos = nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())
# We store our processed tweets in a pickle so that we don't have to process this again when we require it           
save_tweets = open("tweets.pickle","wb")
pickle.dump(documents, save_tweets)
save_tweets.close()
# Here we take our categorized tweets and tokenize them into words 
short_pos_words = word_tokenize(short_pos)
short_neg_words = word_tokenize(short_neg)
for w in short_pos_words:
    all_words.append(w.lower())
for w in short_neg_words:
    all_words.append(w.lower())
# using nltk.FreqDist we calculate the frequency of all the words which we have in all words
all_words = nltk.FreqDist(all_words)
word_features = []
all_words = all_words.most_common(5000) # you can change the value 5000 to whatever you want - my pc cannot handle huge data and for that maater high processing
for key,value in all_words: 
    word_features.append(key)
# We store our processed word features in a pickle so that we don't have to process this again when we require it  
save_word_features = open("word_features5k.pickle","wb")
pickle.dump(word_features, save_word_features)
save_word_features.close()
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
# We use original naive bayes classifier
ONB_classifier = nltk.NaiveBayesClassifier.train(training_set)
ONB_classifier.show_most_informative_features(15)
print("NaiveBayesClassifier accuracy percent:",(nltk.classify.accuracy(ONB_classifier, testing_set))*100)
# We store original naive bayes classifier in a pickle so that we don't have to train this again when we require it
save_classifier = open("originalnaivebayes.pickle","wb")
pickle.dump(ONB_classifier, save_classifier)
save_classifier.close()
# We use multinomial naive bayes classifier 
MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print("MultinomialNB accuracy percent:",nltk.classify.accuracy(MNB_classifier, testing_set)*100)
# We store multinomial naive bayes classifier in a pickle so that we don't have to train this again when we require it
save_classifier = open("multinomialnaivebayes.pickle","wb")
pickle.dump(MNB_classifier, save_classifier)
save_classifier.close()
# We use bernoulli naive bayes classifier 
BNB_classifier = SklearnClassifier(BernoulliNB())
BNB_classifier.train(training_set)
print("BernoulliNB accuracy percent:",nltk.classify.accuracy(BNB_classifier, testing_set)*100)
# We store our bernoulli naive bayes classifier in a pickle so that we don't have to train this again when we require it
save_classifier = open("bernoullinaivebayes.pickle","wb")
pickle.dump(BNB_classifier, save_classifier)
save_classifier.close()
# We use logistic regression classifier 
LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
print("LogisticRegression_classifier accuracy percent:", (nltk.classify.accuracy(LogisticRegression_classifier, testing_set))*100)
# We store our logistic regression classifier in a pickle so that we don't have to train this again when we require it
save_classifier = open("logisticregression.pickle","wb")
pickle.dump(LogisticRegression_classifier, save_classifier)
save_classifier.close()
# We use stochastic gradient descent classifier 
SGDClassifier_classifier = SklearnClassifier(SGDClassifier(max_iter=1000))
SGDClassifier_classifier.train(training_set)
print("SGDClassifier_classifier accuracy percent:", (nltk.classify.accuracy(SGDClassifier_classifier, testing_set))*100)
# We store our stochastic gradient descent classifier in a pickle so that we don't have to train this again when we require it
save_classifier = open("stochasticgradientdescent.pickle","wb")
pickle.dump(SGDClassifier_classifier , save_classifier)
save_classifier.close()
# We use linear Suport Vector Machines classifier 
LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print("LinearSVC_classifier accuracy percent:", (nltk.classify.accuracy(LinearSVC_classifier, testing_set))*100)
# We store our linear Suport Vector Machines classifier in a pickle so that we don't have to train this again when we require it
save_classifier = open("linearsupportvectormachines.pickle","wb")
pickle.dump(LinearSVC_classifier, save_classifier)
save_classifier.close()
# We use nu Suport Vector Machines classifier
NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(training_set)
print("NuSVC_classifier accuracy percent:", (nltk.classify.accuracy(NuSVC_classifier, testing_set))*100)
# We store our nu Suport Vector Machines classifier in a pickle so that we don't have to train this again when we require it
save_classifier = open("nusupportvectormachines.pickle","wb")
pickle.dump(NuSVC_classifier, save_classifier)
save_classifier.close()

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

print("voted_classifier accuracy percent:", (nltk.classify.accuracy(voted_classifier, testing_set))*100)
# two examples were used for checking the working of the code above
print(sentiment("This movie was awesome! The acting was great, plot was wonderful, and there were pythons...so yea!"))
print(sentiment("This movie was utter junk. There were absolutely 0 pythons. I don't see what the point was at all. Horrible movie, 0/10"))







