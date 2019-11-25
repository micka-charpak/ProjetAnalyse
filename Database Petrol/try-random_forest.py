# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 19:11:50 2019

@author: Estelle
"""

#import pandas as pd
import numpy as np
import csv
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


with open('petrol_consumption.csv', 'r') as f: #ourvir le fichier qui se trouve au meme emplacement que ce fichier .py
    reader = csv.reader(f) #retourne un objet "lecteur" qui va iterer sur les lignes dans le fichier csv donne. 
    dataset = np.array(list(reader)) #transforme en liste
    dataset = dataset[1:,:] # suprime le head
    dataset = dataset[1:,:].astype(np.float)#transforme tous les elements du tableau (string) en float

#Division des donnees en attributs et etiquettes
X = dataset[:, 0:4] #recupere toutes les lignes de la 0e à la 4e colonnes
y = dataset[:, 4] #recupere toutes les lignes de la 4e colonne

#print(X)
#print(y)

#Donnees resultantes divisees en ensembles d'entrainement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


#Mettre les donnees a l'echelle 
#Ex : Average_Income de l'ordre du millier
#     Petrol_tax de l'ordre de la dizaine
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Mise en place du random forest
regressor = RandomForestRegressor(n_estimators=20, random_state=0) #resout des problemes de regression via une foret aleatoire
#n_estimators definit le nombre d'arbre de la foret aleatoire
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)



#Evaluer la performance de son algorithme
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))



print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test, y_pred))