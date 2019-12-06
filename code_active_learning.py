# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 14:13:47 2019

@author: Valentin Marconnier
"""

import panda as pd

initial_train_size = 100
# data

# model

# strategy




# split random test forever
data_test = pd.read_csv("test.csv")
cols_throw = ['Name', 'Ticket', 'Cabin', 'Sex', 'Embarked']
# cols_keep = [col for col in data.columns and col not in cols_throw]
# print(cols_keep)
data_test = data_test.drop(cols_throw, axis=1)


data_test.dropna(inplace=True)
print(data_test.shape)

X_testpool = data_test[['Pclass', 'Age', 'SibSp', 'Parch','Fare']].values
y_testpool = data_test['Survived']




if __name__ == '__main__':
    

while n < 500:
    (X_train, y_train , X_test, y_test) = split_data()
    resultats = model(X_train, y_train , X_test, y_test, random_parameter) 
    proba = resultats[0]
    chosen_pool = strategy(proba)
    X_train.append(X_test[i] for i in chosen_pool)
    Y_train.append(Y_train[i] for i in chosen_pool)
    
    
    
    
# sample
# train
# proba
# sampling
# label
# recommence

# score tracking
