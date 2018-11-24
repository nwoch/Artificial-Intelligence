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



X = df[['RM', 'LSTAT', 'DIS', 'TAX', 'AGE', 'CRIM', 'ZN']].values
y = df['MEDV'].values

X = np.sqrt(X)
sc_x = StandardScaler() #center + standardize data
sc_y = StandardScaler()
X_std = sc_x.fit_transform(X)
y_std = sc_y.fit_transform(y[:, np.newaxis]).flatten()

def lin_regplot(X, y, model):
    plt.scatter(X, y, c='steelblue', edgecolor='white', s=70)
    plt.plot(X, model.predict(X), color='black', lw=2)    
    return 

dtr = DecisionTreeRegressor(max_depth=None, max_features=None)
dtr.fit(X_std, y_std)
y_pred = dtr.predict(X_std)
print("Max features, n features, n outputs", dtr.max_features_, dtr.n_features_, dtr.n_outputs_)
print('MSE train: %.3f, test: %.3f' % (
        mean_squared_error(y_std, y_pred),
        mean_squared_error(y_std, y_pred)))
print('R^2 train: %.3f, test: %.3f' % (
        r2_score(y_std, y_pred),
        r2_score(y_std, y_pred)))




########################################

X = df[['RM', 'LSTAT', 'DIS', 'TAX', 'AGE', 'CRIM', 'ZN']].values
#X = df.iloc[:, :-1].values $all
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


dtr = DecisionTreeRegressor(max_depth=None, max_features=None)
dtr.fit(X_train_std, y_train_std)
y_train_pred = dtr.predict(X_train_std)
y_test_pred = dtr.predict(X_test_std)





ary = np.array(range(100000))



plt.scatter(y_train_pred,  y_train_pred - y_train_std,
            c='steelblue', marker='o', edgecolor='white',
            label='Training data')
plt.scatter(y_test_pred,  y_test_pred - y_test_std,
            c='limegreen', marker='s', edgecolor='white',
            label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, color='black', lw=2)
plt.xlim([-10, 50])
plt.tight_layout()

plt.show()





print('MSE train: %.3f, test: %.3f' % (
        mean_squared_error(y_train_std, y_train_pred),
        mean_squared_error(y_test_std, y_test_pred)))
print('R^2 train: %.3f, test: %.3f' % (
        r2_score(y_train_std, y_train_pred),
        r2_score(y_test_std, y_test_pred)))

