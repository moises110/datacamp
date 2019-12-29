# -*- coding: utf-8 -*-
"""
Created on Sun Dec 29 10:45:04 2019

@author: Moisés
"""

from sklearn.datasets import load_boston
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score




A = load_boston()

X = pd.DataFrame(A['data'],columns = A['feature_names'])

y = pd.Series(A['target'], name = 'price')


# here we analize the correlation betweem all the features variables
#import seaborn as sns
#sns.heatmap(X.corr(), cmap='coolwarm', annot = True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

reg = LinearRegression()

reg.fit(X_train,y_train)

y_pred = reg.predict(X_test)

reg.score(X_test,y_test)

score = pd.Series(cross_val_score(reg, X, y, cv  = 5)).mean()






























