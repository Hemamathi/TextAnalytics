from lxml import html  
import requests
from requests.packages.urllib3.exceptions import InsecureRequestWarning
from dateutil import parser as dateparser
from time import sleep
import sys
import codecs
from sklearn.naive_bayes import MultinomialNB
import nltk
from nltk.corpus import stopwords
from nltk.classify import SklearnClassifier
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import sent_tokenize,word_tokenize
from nltk.stem import PorterStemmer
import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile
from textblob import TextBlob
import re
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score
file = open('C:/Users/WELCOME/Desktop/testfile.xls', 'r+')
file.truncate()
file.write("Reviews \n")
file.close()
for n in range(1,6000):
    print ("Extracting Page No " + str(n))
    myurl = 'https://www.amazon.com/Kindle-Wireless-Reader-Wifi-Graphite/product-reviews/B002Y27P3M/ref=cm_cr_arp_d_paging_btm_2?ie=UTF8&reviewerType=all_reviews&pageNumber=' + str(n)
    print (myurl)
    myurl_page = requests.get(myurl,verify=False)
    myurl_text = myurl_page.text
    parser = html.fromstring(myurl_text)
    REVIEW_COMMENT = './/span[@data-hook="review-body"] //text() '
    raw_review = parser.xpath(REVIEW_COMMENT)
    reviews = str(raw_review)
    limiter = "\', \""
    final_review = reviews.replace(limiter,"\n").encode('utf-8').decode('ascii','ignore')
    file = open("C:/Users/WELCOME/Desktop/testfile.xls","a")
    file.write(final_review)
    file.close()
    #OUTPUT:WEBSCRAPING
    #Extracting Page No 1
    #https: // www.amazon.com / Kindle - Wireless - Reader - Wifi - Graphite / product - reviews / B002Y27P3M / ref = cm_cr_arp_d_paging_btm_2?ie = UTF8 & reviewerType = all_reviews & pageNumber = 1
    #C:\Users\WELCOME\PycharmProjects\textanalytics\venv\lib\site - packages\urllib3\connectionpool.py:847: InsecureRequestWarning: Unverified
    #HTTPS request is being made.Adding  certificate  verification is strongly advised.See: https: // urllib3.readthedocs.io / en / latest / advanced - usage.html  # ssl-warnings
    #InsecureRequestWarning)
    #using pandas to store data in a dataframe
    pandas=pd.read_excel("C:/Users/WELCOME/Desktop/reviews.xlsx")
    len(pandas)
    #5258
    nltk.download()
    #showing info https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/index.xml
    ##tokenization and stopwords, steming
    words_stopped=set(stopwords.words("english"))
    words=word_tokenize(str(pandas))
    stemmer=PorterStemmer()
    sent_tokens=sent_tokenize(str(pandas))
    sent_filtered=[]
    for s in sent_tokens:
        sent_filtered.append(s)
    reviews_filtered=[]
    for i in words:
        if i not in words_stopped:
                reviews_filtered.append(stemmer.stem(i))
    #output:
    #['review', '0',  '[',  "'updat",  'novemb',  '2011',   ':', "'", ',',"'mi", 'review','ov', '...', '1','for','reason',....]
    amazon_review=pd.DataFrame(sent_filtered,columns=['Reviews'])
    amazon_review.head(20)
    x = amazon_review['Reviews']
    x.head(10)
    #output 0... 1 Of cou...\n3 If you're trying to choose be... 2 If yo...\n11 Kindle and Nook  have otherfea... 3It's lightweight, even with the attached co......4
    # ", '1.5 Buttonsare...\n184.6E - bookpricing, thoughAmazon haslittle co......7That's a ...\n22    Overall, I have to give th... 8For the
    #Kind...\n25A:  ThenewKindle repl...
    #Name: Reviews, dtype: object
    trans1=CountVectorizer(words).fit(x)
    len(trans1.vocabulary_)
    #296
    #reviews2 = x[1]
    #trans_ = trans1.transform([reviews2])
    #print(trans_)
    #print(trans_.shape)
    #print(trans1.get_feature_names()[75])
    #best
    #print(trans.get_feature_names()[254])
    #things
    x=trans1.transform(x)
    x
    density = (100.0 * x.nnz / (x.shape[0] * x.shape[1]))
    #density: 7.156019656019656
    def senti_analyze(word):
        sentiment_analysis = TextBlob(word)
        if sentiment_analysis.sentiment.polarity > 0:
            return 1
        elif sentiment_analysis.sentiment.polarity == 0:
            return 0
        else:
            return -1
amazon_review['Sentiment_Polarity'] = np.array([ senti_analyze(word) for word in amazon_review['Reviews']])
positive_review = [ review for index, review in enumerate(amazon_review['Reviews']) if amazon_review['Sentiment_Polarity'][index] > 0]
neutral_review = [ review for index, review in enumerate(amazon_review['Reviews']) if amazon_review['Sentiment_Polarity'][index] == 0]
negative_review = [ review for index, review in enumerate(amazon_review['Reviews']) if amazon_review['Sentiment_Polarity'][index] < 0]
print("positive reviews: {}%".format(len(positive_review)*100/len(amazon_review['Reviews'])))#77.27272727272727%
print("neutral reviews: {}%".format(len(neutral_review)*100/len(amazon_review['Reviews'])))#18.181818181818183%
print("negative reviews: {}%".format(len(negative_review)*100/len(amazon_review['Reviews'])))#4.545454545454546%
polarity=amazon_review["Sentiment_Polarity"]
k_train, k_test, b_train, b_test = train_test_split(trans1, polarity, test_size=0.2)
naivebayes=MultinomialNB()
naivebayes.fit(k_train,b_train)
prediction_model=naivebayes.predict(k_test)
print(confusion_matrix(k_test, prediction_model))
print('\n')
print(classification_report(b_test, prediction_model))
accuracy_score(b_test, prediction_model, normalize = True)

