# -*- coding: utf-8 -*-
from sklearn.preprocessing import Normalizer, StandardScaler, RobustScaler
from sklearn import svm
from sklearn import cross_validation
import numpy as np

f = open("crx.data")

target = []
data = []

#entendendo as features
for line in f:
    l = line.split(" ")

    t = l[-1].replace("\n","")
    target.append(t)
    for i in xrange(len(l[:-1])):
        try:
            l[i] = float(l[i])
        except:
            pass
    data.append(l[:-1])

# print len(data[0])
# print len(target)
# print target

X = np.array(data)
for item in X.T:
    # print len(set(item))
    pass

#posicao 4 do vetor é a única feature realmente numérica

hashing_features = []
for i in xrange(len(X.T)):
    if i != 4:
        hashing_features.append(list(set(X.T[i])))
    if i == 4:
        hashing_features.append([])

# print hashing_features

newX = []
for line in X:
    new_line = []
    for i in xrange(len(line)):
        aux = [0]*len(hashing_features[i])
        if i != 4:
            aux[hashing_features[i].index(line[i])] = 1
        else:
            aux = [float(line[i])]
        new_line += aux
    newX.append(new_line)

# print newX
X = np.array(newX)
y = target
print len(X[0])

#aprendendo com SVM
print np.mean(cross_validation.cross_val_score(svm.SVC(), X, y, cv=10))

exit(0)
#é possível melhorar?
sc = StandardScaler()
X_new = sc.fit_transform(X)
print np.mean(cross_validation.cross_val_score(svm.SVC(), X_new, y, cv=10))