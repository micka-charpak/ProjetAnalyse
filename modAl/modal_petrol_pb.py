# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 21:26:11 2019

@author: Estelle
"""

"""
In this example the use of ActiveLearner is demonstrated on the iris dataset in a pool-based sampling setting.
For more information on the iris dataset, see https://en.wikipedia.org/wiki/Iris_flower_data_set
For its scikit-learn interface, see http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from modAL.models import ActiveLearner
import csv

# loading the petrol consumption database
with open('petrol_consumption.csv', 'r') as f: #ourvir le fichier qui se trouve au meme emplacement que ce fichier .py
    reader = csv.reader(f) #retourne un objet "lecteur" qui va iterer sur les lignes dans le fichier csv donne.
    dataset = np.array(list(reader)) #transforme en liste
    dataset = dataset[1:,:] # suprime le head
    dataset = dataset[1:,:].astype(np.float)#transforme tous les elements du tableau (string) en float
data = dataset[:,:4]
target = dataset[:,4]

for i in range(len(target)):
    if target[i] < 550:
        target[i] = 1
    elif 550 <= target[i] < 700:
        target[i] = 2
    else :
        target[i] = 3

print(data)
print(target)




with plt.style.context('seaborn-white'):
    pca = PCA(n_components=2).fit_transform(data)
    plt.figure(figsize=(7, 7))
    plt.scatter(x=pca[:, 0], y=pca[:, 1], c=target, cmap='viridis', s=50)
    plt.title('The petrol database')
    plt.show()

# initial training data
train_idx = [0,20,40]      # index des éléments initiaux du training set
X_train = data[train_idx]
y_train = target[train_idx]



# generating the pool
X_pool = np.delete(data, train_idx, axis=0)
y_pool = np.delete(target, train_idx)

# initializing the active learner
learner = ActiveLearner(
    estimator=RandomForestClassifier(),
    X_training=X_train, y_training=y_train
)

# visualizing initial prediction
with plt.style.context('seaborn-white'):
    plt.figure(figsize=(7, 7))
    prediction = learner.predict(data)
    plt.scatter(x=pca[:, 0], y=pca[:, 1], c=prediction, cmap='viridis', s=50)
    plt.title('Initial accuracy: %f' % learner.score(data, target))
    plt.show()

print('Accuracy before active learning: %f' % learner.score(data, target))

# pool-based sampling
n_queries = 30
for idx in range(n_queries):
    query_idx, query_instance = learner.query(X_pool)
    learner.teach(
        X=X_pool[query_idx].reshape(1, -1),
        y=y_pool[query_idx].reshape(1, )
    )
    # remove queried instance from pool
    X_pool = np.delete(X_pool, query_idx, axis=0)
    y_pool = np.delete(y_pool, query_idx)
    print('Accuracy after query no. %d: %f' % (idx+1, learner.score(data, target)))

# plotting final prediction
with plt.style.context('seaborn-white'):
    plt.figure(figsize=(7, 7))
    prediction = learner.predict(data)
    plt.scatter(x=pca[:, 0], y=pca[:, 1], c=prediction, cmap='viridis', s=50)
    plt.title('Classification accuracy after %i queries: %f' % (n_queries, learner.score(data, target)))
    plt.show()