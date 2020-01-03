#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 09:56:39 2020

@author: moises
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split




A = load_boston()

X = pd.DataFrame(A.data, columns = A.feature_names)
y = pd.Series(A.target)


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2)

#sns.heatmap(X.corr(), cmap = 'coolwarm', annot = True)

# vamos a seleccionar los elementos mas importantes 

lasso = Lasso(alpha = 0.1, normalize = True)
lasso.fit(X_train,y_train)
y_pred = lasso.predict(X_test)
print(lasso.score(X_test,y_test))


#coef = lasso.coef_
#coef = pd.Series(coef)
#coef.index = X.columns

#plt.plot(X.columns, coef)
#plt.show()





















