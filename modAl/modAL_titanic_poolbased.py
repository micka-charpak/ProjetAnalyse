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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from modAL.models import ActiveLearner
import csv
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier


train_size = 200
initial_train_size = 10
# loading the titanic database
def download():
    database = pd.read_csv("train.csv")
    cols_throw = ['Name', 'Ticket', 'Cabin', 'Sex', 'Embarked','PassengerId']
    database = database.drop(cols_throw, axis=1)
    database.dropna(inplace=True)
    database.reset_index(drop=True, inplace=True)
    print('Download progressing ...')
    print(database.shape)
    fulldata = database[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']].values
    fulltarget = database['Survived']
    return (fulldata, fulltarget)



def split(X, y):                 #On commence avec un subset du train_set
    data = X[:train_size]
    target = y[:train_size]
    X_test = X[train_size:]
    y_test = y[train_size:]
    return data,target.values, X_test, y_test



if __name__ == '__main__':
    X, y = download()
    data, target, X_test, y_test = split(X,y)
    random_forest = RandomForestClassifier(random_state=0, n_jobs=-1)
    random_forest.fit(data, target)
    y_predict = random_forest.predict(data)
    y_predict_2 = random_forest.predict(X_test)
    accuracy = accuracy_score(y_predict, target)
    accuracy_2 = accuracy_score(y_predict_2,y_test)

    with plt.style.context('seaborn-white'):
        pca = PCA(n_components=2).fit_transform(data)
        plt.figure(figsize=(7, 7))
        plt.scatter(x=pca[:, 0], y=pca[:, 1], c=target, cmap='viridis', s=50)
        plt.title('The titanic database')
        plt.show()

    # initial training data
    train_idx = np.random.choice(range(len(data)), initial_train_size, replace=False)    # index des éléments initiaux du training se
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
    print('Accuracy on test before active learning: %f' % learner.score(X_test, y_test))


    # pool-based sampling
    n_queries = 40
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
        print('Accuracy on test after query no. %d: %f' % (idx + 1, learner.score(X_test, y_test)))
        print('reference accuracy on training set:', accuracy)
        print('reference accuracy on test set:', accuracy_2)

    # plotting final prediction
    with plt.style.context('seaborn-white'):
        plt.figure(figsize=(7, 7))
        prediction = learner.predict(data)
        plt.scatter(x=pca[:, 0], y=pca[:, 1], c=prediction, cmap='viridis', s=50)
        plt.title('Classification accuracy after %i queries: %f' % (n_queries, learner.score(data, target)))
        plt.show()