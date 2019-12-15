import pandas as pd
import random
import partie_estelle as pe
import numpy as np
import pandas as pd
# data

# model

pool_size = 3

def strategy(probas, pool_size):
    selection = np.random.choice(probas.shape[0], pool_size, replace=False)

    #     print('uniques chosen:',np.unique(selection).shape[0],'<= should be equal to:',initial_labeled_samples)

    return selection

# strategy


if __name__ == '__main__':
    # print(pe.download()[1])
    # print(pe.download()[1][4])
    (X_train, y_train, X_left, y_left, X_test, y_test) = pe.split(pe.download()[0],pe.download()[1])
    X_train= pd.DataFrame(X_train)
    y_train = pd.DataFrame(y_train)
    X_left = pd.DataFrame(X_left)
    y_left = pd.DataFrame(y_left)
    X_test = pd.DataFrame(X_test)
    y_test = pd.DataFrame(y_test)


    while pe.final_train_size - len(X_train) > pool_size :
        resultats = pe.randomForest(X_train, y_train, X_left, y_left)
        probas = resultats[1]
        print(1)


        # on choisit des echantillons à labelliser
        chosen_pool = strategy(probas, pool_size)
        # labbeliser et ajouter les échantillons au modele
        print(chosen_pool)
        print(X_train)
        for i in chosen_pool:
            X_train.append([X_left.iloc[i]])
            y_train.append([y_train.iloc[i]])
        for i in chosen_pool:
            X_left.drop(i,0)
            y_left.drop(i, 0)
    print(pe.randomForest(X_train, y_train, X_test, y_test)[2])
    print(pe.randomForest(pe.download()[0][:pe.final_train_size],pe.download()[1][:pe.final_train_size],X_test,y_test)[2])

# sample
# train
# proba
# sampling
# label
# recommence

# score tracking

# sample
# train
# proba
# sampling
# label
# recommence

# score tracking

