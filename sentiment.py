from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer
from stemming.porter2 import stem
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

import time

corpus = []
annotation = []

arquivo = open("sentiment_corpus.txt")
for line in arquivo:
    line_list = line.split('\t')
    corpus.append(line_list[1])
    annotation.append(line_list[0])

cv = CountVectorizer()
X = cv.fit_transform(corpus).toarray()
print len(X[0]) #muitas features 2132

list_stopwords = []

for stopword in open("stopwords_en.txt"):
    list_stopwords.append(stopword.replace('\n','').replace('\r',''))
list_stopwords.sort()

new_corpus = []

for phrase in corpus:
    new_phrase = ''
    for word in phrase.split(' '):
        if not word in list_stopwords:
            new_phrase += ' '+word
    new_corpus.append(new_phrase)

cv = CountVectorizer()
X = cv.fit_transform(new_corpus).toarray()
print len(X[0]) #ainda muitas features 2007


X_train, X_test, y_train, y_test = train_test_split(X,annotation,test_size=0.3)

print "quantidade de treino: ", len(X_train)

cls1 = MultinomialNB()
cls2 = SVC()


time1 = time.time()
cls1.fit(X_train,y_train)
print "Tempo para treinar o Naive Bayes: ", str(time.time() - time1)

time1 = time.time()
# cls2.fit(X_train,y_train)
print "Tempo para treinar o SVM: ", str(time.time() - time1)

ypred_1 = cls1.predict(X_test)
# ypred_2 = cls2.predict(X_test)

print 'accuracy NB: ',accuracy_score(ypred_1, y_test)
# print 'accuracy SVM: ',accuracy_score(ypred_2, y_test)

while(True):
	text = raw_input("What do you have in mind?\n")
	vector = cv.transform([text]).toarray()
	if cls1.predict(vector)[0] == '0':
		print "Negative\n"
	else:
		print "Positive\n"