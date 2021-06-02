import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
data = pd.read_csv("D:\data.csv")
print(data.head(5))
data.drop(columns=["URLs", "Body"],inplace=True, axis=0)
print (data)
#data cleaning and preeprocessing
import re
import nltk
nltk.download('stopwords') 
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
Corpus = []
for i in range (0, len(data)):
    news = re.sub('[^a-zA-Z]',' ', data['Headline'][i])
    news = news.lower()
    news = news.split()
    news = [ps.stem(word) for word in news if not word in stopwords.words('english')]
    news = ' '.join(news)
    Corpus.append(news)
#print(Corpus)
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(data.Headline,data.Label,test_size=0.1)

#creating bag of words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features= 5000)
x_train = cv.fit_transform(xtrain)
"""The fit_transform() method does both fit and transform."""
x_test = cv.transform(xtest)
"""The transform(data) method is used to perform scaling using mean and std dev 
calculated using the .fit() method."""

#applying SVM algroithm
from sklearn.svm import SVC
Classifier = SVC(kernel = 'rbf' ,random_state=0)
Classifier.fit(x_train, ytrain)
"""The fit(data) method is used to compute the mean and std dev for a given feature 
so that it can be used further for scaling."""

y_pred = Classifier.predict(x_test)

# classification_report(y_test, y_pred)
# print("Classification Report of the MoDEL is:", classification_report)

print(Classifier.score(x_test,y_pred))

from sklearn.metrics import accuracy_score
print("Accuracy:", accuracy_score(ytest,y_pred)*100)

from sklearn.metrics import confusion_matrix
print("Confusion Matrix is :", confusion_matrix(ytest,y_pred))

#Rxtracting TP , FP , FN , TN
TP , FP ,FN ,TN = confusion_matrix(ytest,y_pred).ravel()
print(TP, FP, FN, TN)

from sklearn.metrics import classification_report
matrix = classification_report(ytest,y_pred)
print("CLASSIFICATION REPORT: \n", matrix)

