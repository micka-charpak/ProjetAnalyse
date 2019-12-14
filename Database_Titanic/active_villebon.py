import pandas as pd
import random
import partie_estelle as pe
import numpy as np
# data

# model

pool_size = 3

def strategy(probas, pool_size):
    random_state = check_random_state(0)
    selection = np.random.choice(probas.shape[0], pool_size, replace=False)

    #     print('uniques chosen:',np.unique(selection).shape[0],'<= should be equal to:',initial_labeled_samples)

    return selection

# strategy


if __name__ == '__main__':
    # print(pe.download()[1])
    # print(pe.download()[1][4])
    (X_train, y_train, X_left, y_left, X_test, y_test) = pe.split(pe.download()[0],pe.download()[1])
    while len(X_train) - pe.final_train_size > pool_size :
        print(pe.randomForest(X_train, Y_train, X_test, Y_test))
        resultats = pe.randomForest(X_train, y_train, X_left, y_left)
        probas = resultats[5]

        # on choisit des echantillons à labelliser
        chosen_pool = strategy(proba, pool_size)

        # labbeliser et ajouter les échantillons au modele
        X_train.append(X_left[i] for i in chosen_pool)
        Y_train.append(Y_train[i] for i in chosen_pool)
        for i in chosen_pool:
            np.delete(X_left, i, 0)
        for i in chosen_pool:
            np.delete(Y_test, i, 0)
    print(pe.randomForest(X_train, y_train, X_test, y_test))
    print(pe.randomForest(pe.download()[0][:pe.final_train_size],pe.download()[1][:pe.final_train_size],X_test,y_test))

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

