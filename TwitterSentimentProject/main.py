from tweepy import OAuthHandler
from tweepy import Stream
from tweepy.streaming import StreamListener
import tweepy
import json
import nltk
# nltk.download('stopwords')
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import warnings
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression



#Keys you get with twitter api
consumer_key = ""
consumer_secret = ""
access_key = ""
access_secret = ""

file = open(r"C:\Users\sevgi\Desktop\Tweet-Duygu-Analizi\haziran.txt", 'a', encoding="utf-8")


class StreamListener(tweepy.StreamListener):
    tweets = []
    tweet_batch_size = []

    def on_status(self, status):

        tweet_data = json.dumps(status._json)
        tweet_data = json.loads(tweet_data)
        if "extended_tweet" in tweet_data:
            tweet = tweet_data['extended_tweet']['full_text']
            print(tweet)
            if 'RT' not in tweet:
                file.write(tweet + "\n")



if __name__ == '__main__':
    # complete authorization and initialize API endpoint
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_key, access_secret)
    api = tweepy.API(auth, parser=tweepy.parsers.JSONParser())

    streamListener = StreamListener()
    stream = tweepy.Stream(auth=api.auth, listener=streamListener, tweet_mode='extended')

    tags = ['stayhome', 'stay home', 'covid', "Aşıhayatkurtarır", "aşıolun", "Covid19Aşısı", "corona",
            "pandemic", "pandemi", "COVID-19", "CoronaVaccine"]
    stream.filter(track=tags, languages=["tr"])


data = pd.read_excel(r'C:\Users\sevgi\Desktop\Tweet-Duygu-Analizi\etiket.xlsx', engine='openpyxl')
data.head()

# etiketleme
data["sentiment"].replace(1, value="positive", inplace=True)
data["sentiment"].replace(-1, value="negative", inplace=True)
data["sentiment"].replace(0, value="neutral", inplace=True)

labels = Counter(data['sentiment']).keys()
sum_ = Counter(data['sentiment']).values()
df = pd.DataFrame(zip(labels, sum_), columns=['sentiment', 'Toplam'])
print(df)

# grafik
df.plot(x='sentiment', y='Toplam', kind='bar', legend=False, grid=True, figsize=(15, 10))
plt.title('Kategori Sayılarının Görselleştirilmesi', fontsize=25)
plt.xlabel('Kategoriler', fontsize=15)
plt.ylabel('Toplam', fontsize=15);
plt.show()


#data preprocessing
WPT = nltk.WordPunctTokenizer()
stop_word_list = nltk.corpus.stopwords.words('turkish')

docs = data['tweets']
docs = docs.map(lambda x: re.sub('[,\.!?();:$%&#"]', '', x))
docs = docs.map(lambda x: x.lower())
docs = docs.map(lambda x: x.strip())


# to remove stopwords
def token(values):
    filtered_words = [word for word in values.split() if word not in stop_word_list]
    not_stopword_doc = " ".join(filtered_words)
    return not_stopword_doc


docs = docs.map(lambda x: token(x))
data['tweets'] = docs

#print(data.head(20))

data.groupby("sentiment").count()
dataDoc = data['tweets'].values.tolist()
dataClass = data['sentiment'].values.tolist()
#Separation of data as test-train
x_train, x_test, y_train, y_test = train_test_split(dataDoc, dataClass, test_size=0.2, random_state=42)

#CountVectorizer (Converting data to numeric data)
vect = CountVectorizer()
vect.fit(x_train)
print("Vocabulary size: {}".format(len(vect.vocabulary_)))

vect = CountVectorizer().fit(x_train)
x_train_count = vect.transform(x_train)
print("x_train_count:\n{}".format(repr(x_train_count)))

#LogisticRegression
model = cross_val_score(LogisticRegression(), x_train_count, y_train, cv=5)
print("Logistic Regression Accuracy: {:.2f}".format(np.mean(model)))

lr = LogisticRegression()
lr.fit(x_train_count,y_train)

#Build Text Classification Pipeline
lr_pipeline = make_pipeline(vect,lr)

#save the list of prediction classes
classes = list(lr_pipeline.classes_)

print(pd.DataFrame(lr_pipeline.predict_proba(['corona yüzünden evde kalmaktan sıkıldım bu süreçte güzel hobiler edindim',]), columns=classes))


#We make our model permanent for the web app.
import joblib
joblib.dump(stop_word_list,'stopwords.pkl')
joblib.dump(lr,'model.pkl')
joblib.dump(vect,'vectorizer.pkl')


