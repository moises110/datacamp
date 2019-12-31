# -*- coding: utf-8 -*-
"""
Created on Tue Dec 31 10:04:33 2019

@author: Moisés
"""
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score


A = load_iris()
X = pd.DataFrame(A.data, columns = A.feature_names)
y = pd.Series(A.target)


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2)

clf = KNeighborsClassifier(n_neighbors = 5)

clf.fit(X_train,y_train)

y_predict = clf.predict(X_test)

clf.score(X_test,y_test)


cross_val_score(clf, X, y, cv = 5).mean()











