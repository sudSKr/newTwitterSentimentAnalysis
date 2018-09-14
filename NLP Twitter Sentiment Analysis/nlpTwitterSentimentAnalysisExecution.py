'''@author: SuD
   This module is used for live streaming twitter data and to save the outout in a file
'''
from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import json
import nlpTwitterSentimentAnalysisModule as TSAM
# below we are supposed to store our keys and tokens we can get after creating out twiiter application at twiiter.com
ckey=""
csecret=""
atoken=""
asecret=""
#this class will get us our data from twitter
class listener(StreamListener):
    def on_data(self, data):
        all_data = json.loads(data)
        tweet = ascii(all_data["text"]) # this is the tweet which we are going to get from twitter
        sentiment_value, confidence = TSAM.sentiment(tweet) # we process the tweets here to find the clasification and the confidence
        print(tweet, sentiment_value, confidence)
        if confidence*100 >= 70: # we have used 7 classifiers so we are only going to save the tweets if atleast 5 out of 7 of them agree i.e. atleast 71% 
            output = open("twitterOutput.txt","a") # saving them to our file
            output.write(sentiment_value) # we are just saving the tweets's classification for our next process -you can save all if you want
            output.write('\n')
            output.close()
        return True
#  to show any error we get - hope we don't   
    def on_error(self, status):
        print(self,status)
# as i am planning to keep the keys and token please fill your own keys and tokens
if(ckey=="" or csecret=="" or atoken=="" or asecret==""):
    print("Kindly insert the consumer key, consumer secret, access token, access secret.")
else:
    auth = OAuthHandler(ckey, csecret)
    auth.set_access_token(atoken, asecret)
    twitterStream = Stream(auth, listener()) # our twitter stream
    twitterStream.filter(track=["the predator"]) # we filter our tweets here from the twitter stream