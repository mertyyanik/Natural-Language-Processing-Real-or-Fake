import pandas as pd
import numpy as np
import re
import random
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.metrics import accuracy_score

# Reading Data
input1Data = open('clean_real-Train.txt', 'r+')
input2Data = open('clean_fake-Train.txt', 'r+')
testData = open('test.txt')
testDataForComparison = pd.read_csv('test.csv').iloc[:, 1:]

input1List = []
input2List = []
testList = []

read = input1Data.readline()
read2 = input2Data.readline()
read3 = testData.readline()
rowCount = 0
while read != "":
    input1List.append(read)
    read = input1Data.readline()
    
while read2 != "":
    input2List.append(read2)
    read2 = input2Data.readline()

while read3 != "":
    testList.append(read3)
    read3 = testData.readline()
    
# Standard

sc = StandardScaler()
    
# Real and Fake data combined.    
liste = ['real' for i in range(len(input1List))]
realInput = np.column_stack((input1List, liste))


liste = ['fake' for i in range(len(input2List))]
fakeInput = np.column_stack((input2List, liste))

mainInput = realInput.tolist() + fakeInput.tolist()
random.shuffle(mainInput)

sentenceInput = []
targetInput = []
targetInputFloat = []
for i in range(len(mainInput)):
    sentenceInput.append(mainInput[i][0])
    
for i in range(len(mainInput)):
    targetInput.append(mainInput[i][1])

from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()
targetInputFloat = lb.fit_transform(targetInput)

# Stop Words
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

# Stemmer
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

sentenceList = []
for i in range(len(sentenceInput)):
    # Regular Expressions
    comment = re.sub('[^a-zA-Z]', ' ', sentenceInput[i])
    # Upper Case Problem
    comment = comment.lower()
    # Splitting
    comment = comment.split()
    
    # Stemming and Stop Words
    comment = [ps.stem(kelime) for kelime in comment if not kelime in set(stopwords.words('english'))]
    
    comment = ' '.join(comment)
    sentenceList.append(comment)
    

# Vectorized All Data
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(sentenceList)

result = pd.DataFrame(data=X.toarray())
##resultStandart = sc.fit_transform(result)


# PCA - Principle Component Analysis
from sklearn.decomposition import PCA
pca = PCA(n_components=1900)
result2 = pca.fit_transform(result)

# Naive Bayes Prediction
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()

nb.fit(result2, targetInput)


# KNeighbors
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5,metric='minkowski')

knn.fit(result2, targetInput)


# Logistic Regression
from sklearn.linear_model import LogisticRegression

logr = LogisticRegression(random_state = 1, solver='lbfgs')
logr.fit(result2, targetInput)

# Test Data
testSentenceList = []
for i in range(len(testList)):
    # Regular Expressions
    comment = re.sub('[^a-zA-Z]', ' ', testList[i])
    # Upper Case Problem
    comment = comment.lower()
    # Splitting
    comment = comment.split()
    
    # Stemming and Stop Words
    comment = [ps.stem(kelime) for kelime in comment if not kelime in set(stopwords.words('english'))]
    
    comment = ' '.join(comment)
    testSentenceList.append(comment)

X = vectorizer.transform(testSentenceList)

testResult = pd.DataFrame(data=X.toarray())
# PCA for Test
testResult2 = pca.transform(testResult)

##testResultStandard = sc.transform(testResult)

# Prediction
predictionNaive = nb.predict(testResult2)
predictionKNN = knn.predict(testResult2)
predictionLogistic = logr.predict(testResult2)

# Accuracy

acScoreNaive = accuracy_score(testDataForComparison,predictionNaive)
acScoreKNN = accuracy_score(testDataForComparison,predictionKNN)
acScoreLogistic = accuracy_score(testDataForComparison, predictionLogistic)

print("Accuracy Naive  : "  + str(acScoreNaive))
print("Accuracy KNN : "  + str(acScoreKNN))
print("Accuracy Logistic : "  + str(acScoreLogistic))

# AUC


probaNaive = nb.predict_proba(testResult2)
probaLogistic = logr.predict_proba(testResult2)
probaKNN = knn.predict_proba(testResult2)

print("AUC Naive " + str(metrics.roc_auc_score(testDataForComparison, probaNaive[:, 1])))
print("AUC KNN " + str(metrics.roc_auc_score(testDataForComparison, probaKNN[:, 1])))
print("AUC Logistic " + str(metrics.roc_auc_score(testDataForComparison, probaLogistic[:, 1])))

# Logistic Regresyon Cross-Validation
from sklearn.model_selection import cross_val_score

basari = cross_val_score(estimator=logr, X=result2, y=targetInput, cv=4)
print("Logistic Cross-Validation: "+str(basari.mean()))
print("Logistic Cross-Validation Error:"+str(basari.std()))


# Neural Network
from keras.layers import Dropout
from keras.models import Sequential
from keras.layers import Dense, LSTM

result2 = result2.reshape((2777, 1, 1900))
testResult2 = testResult2.reshape((489, 1, 1900))

classifier = Sequential()

#classifier.add(Dense(1500, init='uniform', activation='sigmoid', input_dim=1900))
classifier.add(LSTM(1500, input_shape=(1, 1900), activation='sigmoid'))
classifier.add(Dropout(0.50))

classifier.add(LSTM(1500, activation='sigmoid'))
classifier.add(Dropout(0.50))

classifier.add(LSTM(1500, activation='sigmoid'))
classifier.add(Dropout(0.50))

classifier.add(Dense(1, init='uniform', activation='sigmoid'))
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

classifier.fit(result2, targetInputFloat, epochs=7)  # 7 times.
prediction = classifier.predict(testResult2)

prediction = (prediction > 0.5)

from sklearn.metrics import accuracy_score

acScoreNN = accuracy_score(lb.fit_transform(testDataForComparison), prediction)
print("Accuracy : "+str(acScoreNN))

probaNN = classifier.predict_proba(testResult2)
print("NN AUC " + str(metrics.roc_auc_score(lb.fit_transform(testDataForComparison), probaNN)))


# XGBoost
from xgboost import XGBClassifier
classifierXG = XGBClassifier()
classifierXG.fit(result2, targetInput)

predictionXGBoost = classifierXG.predict(testResult2)

acScoreXGBoost = accuracy_score(testDataForComparison, predictionXGBoost)
print("Accuracy XGBoost: " + str(acScoreXGBoost))


# Saving the best model with Pickle (Neural %82)
import pickle
pickle.dump(classifier, open("NeuralNews", 'wb'))

loading = pickle.load(open("NeuralNews", 'rb'))
predictionPickleNeural = loading.predict(testResult2)

predictionPickleNeural = (predictionPickleNeural > 0.5)

acScorePickleNeural = accuracy_score(lb.fit_transform(testDataForComparison), predictionPickleNeural)
print("Accuracy Pickle Neural : " + str(acScorePickleNeural))


