# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 16:22:27 2019

@author: Estelle
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics


dataset_train = pd.read_csv('C:\\Users\\Estelle\\Documents\\train.csv', engine='python', encoding = "utf-8-sig")

dataset_train.head() #to get a high-level view 

print(dataset_train)


#Divide data into 'attributes' and 'label' sets.
#Then divided data into training and test sets.

X = dataset_train.iloc[1:, 2:].values
y = dataset_train.iloc[1:, 1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

"""
Scale our data
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
"""

regressor = RandomForestRegressor(n_estimators=20, random_state=0)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)


#evaluate our algorithm
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))




