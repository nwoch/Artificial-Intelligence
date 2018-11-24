#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 07:29:43 2018

@author: richard
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RANSACRegressor
from sklearn.model_selection import train_test_split
import scipy as sp
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


df = pd.read_csv('./housing.txt', sep='\s+')

df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 
              'NOX', 'RM', 'AGE', 'DIS', 'RAD', 
              'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
df.head()



import seaborn as sns
cols = ['RM', 'LSTAT', 'MEDV']
sns.pairplot(df[cols], height=2.5)
plt.tight_layout()
plt.show()


#####################################################

cm = np.corrcoef(df[cols].values.T)
hm = sns.heatmap(cm,
                 cbar=True,
                 annot=True,
                 square=True,
                 fmt='.2f',
                 annot_kws={'size': 15},
                 yticklabels=cols,
                 xticklabels=cols)

plt.tight_layout()
plt.show()


# ## Modeling nonlinear relationships in the Housing Dataset



X = df[['RM', 'LSTAT', 'CRIM', 'TAX', 'DIS', 'INDUS', 'PTRATIO', 'CHAS', 'ZN']].values
y = df['MEDV'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0)

X_train = np.sqrt(X_train)
X_test = np.sqrt(X_test)
sc_X = StandardScaler() #center + standardize data
sc_y = StandardScaler()
X_train_std = sc_X.fit_transform(X_train)
y_train_std = sc_y.fit_transform(y_train[:, np.newaxis]).flatten()
X_test_std = sc_X.transform(X_test)
y_test_std = sc_y.transform(y_test[:, np.newaxis]).flatten()

regr = LinearRegression()

# create quadratic features
quadratic_train = PolynomialFeatures(degree=2)
cubic_train = PolynomialFeatures(degree=3)
quadratic_test = PolynomialFeatures(degree=2)
cubic_test = PolynomialFeatures(degree=3)
X_train_quad = quadratic_train.fit_transform(X_train)
X_train_cubic = cubic_train.fit_transform(X_train)
X_test_quad = quadratic_test.fit_transform(X_test)
X_test_cubic = cubic_test.fit_transform(X_test)

# fit features
X_train_fit = np.arange(X_train_std.min(), X_train_std.max(), 1)[:, np.newaxis]
X_test_fit = np.arange(X_test_std.min(), X_test_std.max(), 1)[:, np.newaxis]

regr = regr.fit(X_train_std, y_train_std)
y_train_pred = regr.predict(X_train_std)
y_test_pred = regr.predict(X_test_std)
linear_train_r2 = r2_score(y_train_std, y_train_pred)
linear_test_r2 = r2_score(y_test_std, y_test_pred)
print("Linear train R2", linear_train_r2)
print("Linear test R2", linear_test_r2)

regr = regr.fit(X_train_quad, y_train)
quadratic_train_r2 = r2_score(y_train, regr.predict(X_train_quad))
quadratic_test_r2 = r2_score(y_test, regr.predict(X_test_quad))
print("Quad train R2", quadratic_train_r2)
print("Quad test R2", quadratic_test_r2)

# # Using regularized methods for regression

regr = regr.fit(X_train_cubic, y_train)
cubic_train_r2 = r2_score(y_train, regr.predict(X_train_cubic))
cubic_test_r2 = r2_score(y_test, regr.predict(X_test_cubic))
print("Cubic train R2", cubic_train_r2)
print("Cubic test R2", cubic_test_r2)
