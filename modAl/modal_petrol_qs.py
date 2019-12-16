# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 11:47:28 2019

@author: Estelle
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 21:06:25 2019

@author: Estelle
"""

import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from modAL.models import ActiveLearner, Committee


# loading the petrol consumption database
with open('petrol_consumption.csv', 'r') as f: #ourvir le fichier qui se trouve au meme emplacement que ce fichier .py
    reader = csv.reader(f) #retourne un objet "lecteur" qui va iterer sur les lignes dans le fichier csv donne.
    dataset = np.array(list(reader)) #transforme en liste
    dataset = dataset[1:,:] # suprime le head
    dataset = dataset[1:,:].astype(np.float)#transforme tous les elements du tableau (string) en float
data = dataset[:,:4]
target = dataset[:,4]







# visualizing the classes
with plt.style.context('seaborn-white'):
    plt.figure(figsize=(7, 7))
    pca = PCA(n_components=2).fit_transform(data) #decomposer en 2 paramètres regroupant plusieurs caracteristiques
    plt.scatter(x=pca[:, 0], y=pca[:, 1], c=target, cmap='viridis', s=50) #c correspond à la couleur en fonction de la plante
    plt.title('The petrol dataset')
    plt.show()

# generate the pool
X_pool = deepcopy(data) #copie recursive de l'objet initial
y_pool = deepcopy(target)

# initializing Committee members
n_members = 2
learner_list = list()

for member_idx in range(n_members):
    # initial training data
    n_initial = 5
    train_idx = np.random.choice(range(X_pool.shape[0]), size=n_initial, replace=False) #random choice
    X_train = X_pool[train_idx]
    y_train = y_pool[train_idx]

    # creating a reduced copy of the data with the known instances removed
    X_pool = np.delete(X_pool, train_idx, axis=0)
    y_pool = np.delete(y_pool, train_idx)

    # initializing learner
    learner = ActiveLearner(
        estimator=RandomForestClassifier(),
        X_training=X_train, y_training=y_train
    )
    learner_list.append(learner)

# assembling the committee
committee = Committee(learner_list=learner_list)

# visualizing the Committee's predictions per learner

with plt.style.context('seaborn-white'):
    plt.figure(figsize=(n_members*7, 7))
    for learner_idx, learner in enumerate(committee):
        plt.subplot(1, n_members, learner_idx + 1)
        plt.scatter(x=pca[:, 0], y=pca[:, 1], c=learner.predict(data), cmap='viridis', s=50)
        plt.title('Learner no. %d initial predictions' % (learner_idx + 1))
    plt.show()

# visualizing the initial predictions
with plt.style.context('seaborn-white'):
    plt.figure(figsize=(7, 7))
    prediction = committee.predict(data)
    plt.scatter(x=pca[:, 0], y=pca[:, 1], c=prediction, cmap='viridis', s=50)
    plt.title('Committee initial predictions, accuracy = %1.3f' % committee.score(data, target))
    plt.show()

# query by committee
n_queries = 10
for idx in range(n_queries):
    query_idx, query_instance = committee.query(X_pool)
    committee.teach(
        X=X_pool[query_idx].reshape(1, -1),
        y=y_pool[query_idx].reshape(1, )
    )
    # remove queried instance from pool
    X_pool = np.delete(X_pool, query_idx, axis=0)
    y_pool = np.delete(y_pool, query_idx)

# visualizing the final predictions per learner
with plt.style.context('seaborn-white'):
    plt.figure(figsize=(n_members*7, 7))
    for learner_idx, learner in enumerate(committee):
        plt.subplot(1, n_members, learner_idx + 1)
        plt.scatter(x=pca[:, 0], y=pca[:, 1], c=learner.predict(data), cmap='viridis', s=50)
        plt.title('Learner no. %d predictions after %d queries' % (learner_idx + 1, n_queries))
    plt.show()

# visualizing the Committee's predictions
with plt.style.context('seaborn-white'):
    plt.figure(figsize=(7, 7))
    prediction = committee.predict(data)
    plt.scatter(x=pca[:, 0], y=pca[:, 1], c=prediction, cmap='viridis', s=50)
    plt.title('Committee predictions after %d queries, accuracy = %1.3f'
              % (n_queries, committee.score(data, target)))
    plt.show()