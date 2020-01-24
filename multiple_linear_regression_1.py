# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 12:46:30 2020

@author: Dolley
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values#independent Values
y = dataset.iloc[:, 4].values#dependent Values

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding the Dummy Variable Trap
X = X[:, 1:]
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

#linear regression taining set
from sklearn.linear_model import LinearRegression
linearRegression=LinearRegression()
linearRegression.fit(X_train,y_train)

#testing set
y_pred=linearRegression.predict(X_test)


import statsmodels.api as sm
X=np.append(arr=np.ones((50,1)).astype(int), values=X, axis=1)
X_opt=X[:,[0,1,2,3,4,5]]
sm_ols = sm.OLS(y_train, X_train).fit()
sm_ols.summary()
X_opt=X[:,[0,3]]
sm_ols = sm.OLS(y_train, X_train).fit()
#plotting of the graph is not done yet
plt.scatter(X_train,y_train,color='black')
plt.plot(X_train,linearRegression.predict(X_train),color='green')
plt.xlabel("Years of exp")
plt.ylabel("salry")
plt.title("salary v/s years of exp")
plt.show()