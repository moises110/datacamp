# -*- coding: utf-8 -*-
"""
Created on Tue Dec 31 10:37:36 2019

@author: Moisés
"""
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
#import seaborn as sns

A = load_diabetes()

X = pd.DataFrame(A.data, columns = A.feature_names)
y = pd.Series(A.target,name = 'glucose')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)


#sns.heatmap(X.corr(), cmap = 'coolwarm', annot = True)

reg = LinearRegression()

reg.fit(X_train,y_train)

y_predict = pd.Series(reg.predict(X_test))

reg.score(X_test,y_test)





 










