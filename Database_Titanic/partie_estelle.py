# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 14:11:54 2019

@author: Estelle
"""

# import os
# import time
# import json
# import pickle
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# import matplotlib.mlab as mlab
# from scipy import stats
# from pylab import rcParams
from sklearn.utils import check_random_state
from sklearn.datasets import load_digits
from sklearn.datasets import fetch_mldata
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, \
     GradientBoostingClassifier
    
    
    
max_queried = 500
train_size = 250
initial_train_size = 100
final_train_size = 451



def download():
    data = pd.read_csv("train.csv")
    cols_throw = ['Name', 'Ticket', 'Cabin', 'Sex', 'Embarked','PassengerId']
    data = data.drop(cols_throw, axis=1)
    data.dropna(inplace=True)
    print('Download progressing ...')
    print(data.shape)
    X = data[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']].values
    y = data['Survived']
    return X,y



def split(X, y):                 #On commence avec un subset du train_set
    X_train_initial = X[:train_size]
    y_train_initial = y[:train_size]
    X_left = X[initial_train_size:final_train_size]
    y_left = y[initial_train_size:final_train_size]
    X_test = X[train_size:]
    y_test = y[train_size:]
    return (X_train_initial, y_train_initial, X_left, y_left, X_test, y_test)

    

def randomForest(X_train_initial, y_train_initial, X_left, y_left):
    """
    print('----------------------------')
    print('Evaluation of the algorithm')
    print('----------------------------')
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_predict))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_predict))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_predict)))
    print('Confusion Matrix:', confusion_matrix(y_test,y_predict))
    print('Accurancy Score:', accuracy_score(y_test, y_predict))
    print('Proba Predict:', prediction)
    print('Random Forest launch ...')
    """
    random_forest = RandomForestClassifier(random_state = 0, n_jobs = -1)
    random_forest.fit(X_train_initial, y_train_initial)
    y_predict = random_forest.predict(X_left)
    prediction = random_forest.predict_proba(X_left)
    accuracy = accuracy_score(y_left, y_predict)
    return (y_predict, prediction, accuracy)


    
    
    
    