# TwitterSentimentAnalysis
 Tweet sentiment analysis with Twitter API and simple web app with Flask
 
 In this project, sentiment analysis was performed on tweets in Turkish.
 
 I got real time tweets with Twitter API using Tweepy library.
 "You need to create a developer account to use the Twitter API.--> https://developer.twitter.com/"
 https://help.twitter.com/en/rules-and-policies/twitter-api

These tweets are written to a text file. I chose these tweets myself and wrote the selected tweets to an excel file. My choices were tweets written more clearly.

In the Excel file, I labeled the sentences according to their mood.
"positive=1 , negative=-1, neutral=0"

I did classification using Logistic Regression.
Then I saved my models.
I used the models in a simple web app I wrote with Flask.
