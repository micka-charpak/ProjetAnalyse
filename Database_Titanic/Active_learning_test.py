# !/usr/bin/python
# -*- coding: utf-8 -*-


import os
import time
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from scipy import stats
from pylab import rcParams
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
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, \
    GradientBoostingClassifier

max_queried = 500

trainset_size = 714 # ie., testset_size = 10000


def download():
    data = pd.read_csv("train.csv")
    cols_throw = ['Name', 'Ticket', 'Cabin', 'Sex', 'Embarked']
    # cols_keep = [col for col in data.columns and col not in cols_throw]
    # print(cols_keep)
    data = data.drop(cols_throw, axis=1)

    data.dropna(inplace=True)
    print(data.shape)

    X = data[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']].values
    y = data['Survived']
    return (X,y)



def split(train_size):                 #On commence avec un subset du train_set
    X_train_full = X[:train_size]
    y_train_full = y[:train_size]
    X_test = X[train_size:]
    y_test = y[train_size:]
    return (X_train_full, y_train_full, X_test, y_test)


class BaseModel(object):

    def __init__(self):
        pass

    def fit_predict(self):
        pass


class SvmModel(BaseModel):
    model_type = 'Support Vector Machine with linear Kernel'

    def fit_predict(self, X_train, y_train, X_val, X_test, c_weight):
        print('training svm...')
        self.classifier = SVC(C=1, kernel='linear', probability=True,
                              class_weight=c_weight)
        self.classifier.fit(X_train, y_train)
        self.test_y_predicted = self.classifier.predict(X_test)
        self.val_y_predicted = self.classifier.predict(X_val)
        return (X_train, X_val, X_test, self.val_y_predicted,
                self.test_y_predicted)


class LogModel(BaseModel):
    model_type = 'Multinominal Logistic Regression'

    def fit_predict(self, X_train, y_train, X_val, X_test, c_weight):
        print('training multinomial logistic regression')
        train_samples = X_train.shape[0]
        self.classifier = LogisticRegression(
            C=50. / train_samples,
            multi_class='multinomial',
            penalty='l1',
            solver='saga',
            tol=0.1,
            class_weight=c_weight,
        )
        self.classifier.fit(X_train, y_train)
        self.test_y_predicted = self.classifier.predict(X_test)
        self.val_y_predicted = self.classifier.predict(X_val)
        return (X_train, X_val, X_test, self.val_y_predicted,
                self.test_y_predicted)



class RfModel(BaseModel):
    model_type = 'Random Forest'

    def fit_predict(self, X_train, y_train, X_val, X_test, c_weight):    # X_val --> reste du train_set ?
        print('training random forest...')
        self.classifier = RandomForestClassifier(n_estimators=500, class_weight=c_weight)
        self.classifier.fit(X_train, y_train)
        self.test_y_predicted = self.classifier.predict(X_test)
        self.val_y_predicted = self.classifier.predict(X_val)
        return (X_train, X_val, X_test, self.val_y_predicted, self.test_y_predicted)


# entrainement et visualisation de la performance du model sur un set d'entrainement.


class TrainModel:



    def __init__(self, model_object):
        self.accuracies = []
        self.model_object = model_object()

    def print_model_type(self):
        print(self.model_object.model_type)

    # we train normally and get probabilities for the validation set. i.e., we use the probabilities to select the most uncertain samples

    def train(self, X_train, y_train, X_val, X_test, c_weight):
        print('Train set:', X_train.shape, 'y:', y_train.shape)
        print('Val   set:', X_val.shape)
        print('Test  set:', X_test.shape)
        t0 = time.time()
        (X_train, X_val, X_test, self.val_y_predicted,
         self.test_y_predicted) = \
            self.model_object.fit_predict(X_train, y_train, X_val, X_test, c_weight)
        self.run_time = time.time() - t0
        return (
        X_train, X_val, X_test)  # we return them in case we use PCA, with all the other algorithms, this is not needed.

    # we want accuracy only for the test set

    def get_test_accuracy(self, i, y_test):
        classif_rate = np.mean(self.test_y_predicted.ravel() == y_test.ravel()) * 100
        self.accuracies.append(classif_rate)
        print('--------------------------------')
        print('Iteration:', i)
        print('--------------------------------')
        print('y-test set:', y_test.shape)
        print('Example run in %.3f s' % self.run_time, '\n')
        print("Accuracy rate for %f " % (classif_rate))
        print("Classification report for classifier %s:\n%s\n" % (
        self.model_object.classifier, metrics.classification_report(y_test, self.test_y_predicted)))
        print("Confusion matrix:\n%s" % metrics.confusion_matrix(y_test, self.test_y_predicted))
        print('--------------------------------')


# classe regroupant différentes fonctions de sléection des échantillons


class BaseSelectionFunction(object): #classe de base de la fonction de selection des echantillons à labelliser

    def __init__(self):
        pass

    def select(self):
        pass


class RandomSelection(BaseSelectionFunction):

    @staticmethod
    def select(probas_val, initial_labeled_samples):
        random_state = check_random_state(0)
        selection = np.random.choice(probas_val.shape[0], initial_labeled_samples, replace=False)

        #     print('uniques chosen:',np.unique(selection).shape[0],'<= should be equal to:',initial_labeled_samples)

        return selection


class EntropySelection(BaseSelectionFunction):

    @staticmethod
    def select(probas_val, initial_labeled_samples):
        e = (-probas_val * np.log2(probas_val)).sum(axis=1)
        selection = (np.argsort(e)[::-1])[:initial_labeled_samples]
        return selection


class MarginSamplingSelection(BaseSelectionFunction):

    @staticmethod
    def select(probas_val, initial_labeled_samples):
        rev = np.sort(probas_val, axis=1)[:, ::-1]
        values = rev[:, 0] - rev[:, 1]
        selection = np.argsort(values)[:initial_labeled_samples]
        return selection


# normaliser pour réduire l'espace de variation des valeurs

class Normalize(object):

    def normalize(self, X_train, X_val, X_test):
        self.scaler = MinMaxScaler()
        X_train = self.scaler.fit_transform(X_train)
        X_val = self.scaler.transform(X_val)
        X_test = self.scaler.transform(X_test)
        return (X_train, X_val, X_test)

    def inverse(self, X_train, X_val, X_test):
        X_train = self.scaler.inverse_transform(X_train)
        X_val = self.scaler.inverse_transform(X_val)
        X_test = self.scaler.inverse_transform(X_test)
        return (X_train, X_val, X_test)


# fonction pour obtenir un échantillon aléatoire de données à partir du pool de données non étiquetées

def get_k_random_samples(initial_labeled_samples, X_train_full,
                         y_train_full):
    random_state = check_random_state(0)
    permutation = np.random.choice(trainset_size,
                                   initial_labeled_samples,
                                   replace=False)
    print()
    print('initial random chosen samples', permutation.shape),
    #            permutation)
    X_train = X_train_full[permutation]
    y_train = y_train_full[permutation]
    X_train = X_train.reshape((X_train.shape[0], -1))
    bin_count = np.bincount(y_train.astype('int64'))
    unique = np.unique(y_train.astype('int64'))
    print(
        'initial train set:',
        X_train.shape,
        y_train.shape,
        'unique(labels):',
        bin_count,
        unique,
    )
    return (permutation, X_train, y_train)


# classe principale qui initialise le processus d'apprentissage actif ( sélection d'échantillons aléatoires 'k' -> formation du modèle -> sélection des échantillons les plus 'informatifs' -> on les retire de l'ensemble de validation -> on vérifie l'étiquette -> on recycle jusqu'à obtenir ce qu'on vise )

class TheAlgorithm(object):
    accuracies = []

    def __init__(self, initial_labeled_samples, model_object, selection_function):
        self.initial_labeled_samples = initial_labeled_samples
        self.model_object = model_object
        self.sample_selection_function = selection_function

    def run(self, X_train_full, y_train_full, X_test, y_test):
        # initialize process by applying base learner to labeled training data set to obtain Classifier

        (permutation, X_train, y_train) = \
            get_k_random_samples(self.initial_labeled_samples,
                                 X_train_full, y_train_full)
        self.queried = self.initial_labeled_samples
        self.samplecount = [self.initial_labeled_samples]

        # permutation, X_train, y_train = get_equally_k_random_samples(self.initial_labeled_samples,classes)

        # assign the val set the rest of the 'unlabelled' training data

        X_val = np.array([])
        y_val = np.array([])
        X_val = np.copy(X_train_full)
        X_val = np.delete(X_val, permutation, axis=0)
        y_val = np.copy(y_train_full)
        y_val = np.delete(y_val, permutation, axis=0)
        print('val set:', X_val.shape, y_val.shape, permutation.shape)
        print()

        # normalize data

        normalizer = Normalize()
        X_train, X_val, X_test = normalizer.normalize(X_train, X_val, X_test)

        self.clf_model = TrainModel(self.model_object)
        (X_train, X_val, X_test) = self.clf_model.train(X_train, y_train, X_val, X_test, 'balanced')
        active_iteration = 1
        self.clf_model.get_test_accuracy(1, y_test)

        # fpfn = self.clf_model.test_y_predicted.ravel() != y_val.ravel()
        # print(fpfn)
        # self.fpfncount = []
        # self.fpfncount.append(fpfn.sum() / y_test.shape[0] * 100)

        while self.queried < max_queried:
            active_iteration += 1

            # get validation probabilities

            probas_val = \
                self.clf_model.model_object.classifier.predict_proba(X_val)
            print('val predicted:',
                  self.clf_model.val_y_predicted.shape,
                  self.clf_model.val_y_predicted)
            print('probabilities:', probas_val.shape, '\n',
                  np.argmax(probas_val, axis=1))

            # select samples using a selection function

            uncertain_samples = \
                self.sample_selection_function.select(probas_val, self.initial_labeled_samples)

            # normalization needs to be inversed and recalculated based on the new train and test set.

            X_train, X_val, X_test = normalizer.inverse(X_train, X_val, X_test)

            # get the uncertain samples from the validation set

            print('trainset before', X_train.shape, y_train.shape)
            X_train = np.concatenate((X_train, X_val[uncertain_samples]))
            y_train = np.concatenate((y_train, y_val[uncertain_samples]))
            print('trainset after', X_train.shape, y_train.shape)
            self.samplecount.append(X_train.shape[0])

            bin_count = np.bincount(y_train.astype('int64'))
            unique = np.unique(y_train.astype('int64'))
            print(
                'updated train set:',
                X_train.shape,
                y_train.shape,
                'unique(labels):',
                bin_count,
                unique,
            )

            X_val = np.delete(X_val, uncertain_samples, axis=0)
            y_val = np.delete(y_val, uncertain_samples, axis=0)
            print('val set:', X_val.shape, y_val.shape)
            print()

            # normalize again after creating the 'new' train/test sets
            normalizer = Normalize()
            X_train, X_val, X_test = normalizer.normalize(X_train, X_val, X_test)

            self.queried += self.initial_labeled_samples
            (X_train, X_val, X_test) = self.clf_model.train(X_train, y_train, X_val, X_test, 'balanced')
            self.clf_model.get_test_accuracy(active_iteration, y_test)

        print('final active learning accuracies',
              self.clf_model.accuracies)


# téléchargement des donnée pour les valider et les tester, cela sur l'ensemble des algorithmes d'entrainement et de nos fonctions de sélection


(X, y) = download()
(X_train_full, y_train_full, X_test, y_test) = split(trainset_size)
print('train:', X_train_full.shape, y_train_full.shape)
print('test :', X_test.shape, y_test.shape)
classes = len(np.unique(y))
print('unique classes', classes)


def pickle_save(fname, data):
    filehandler = open(fname, "wb")
    pickle.dump(data, filehandler)
    filehandler.close()
    print('saved', fname, os.getcwd(), os.listdir())


def pickle_load(fname):
    print(os.getcwd(), os.listdir())
    file = open(fname, 'rb')
    data = pickle.load(file)
    file.close()
    print(data)
    return data


def experiment(d, models, selection_functions, Ks, repeats, contfrom):
    algos_temp = []
    print('stopping at:', max_queried)
    count = 0
    for model_object in models:
        if model_object.__name__ not in d:
            d[model_object.__name__] = {}

        for selection_function in selection_functions:
            if selection_function.__name__ not in d[model_object.__name__]:
                d[model_object.__name__][selection_function.__name__] = {}

            for k in Ks:
                d[model_object.__name__][selection_function.__name__][str(k)] = []

                for i in range(0, repeats):
                    count += 1
                    if count >= contfrom:
                        print('Count = %s, using model = %s, selection_function = %s, k = %s, iteration = %s.' % (
                        count, model_object.__name__, selection_function.__name__, k, i))
                        alg = TheAlgorithm(k,
                                           model_object,
                                           selection_function
                                           )
                        alg.run(X_train_full, y_train_full, X_test, y_test)
                        d[model_object.__name__][selection_function.__name__][str(k)].append(alg.clf_model.accuracies)
                        fname = 'Active-learning-experiment-' + str(count) + '.pkl'
                        pickle_save(fname, d)
                        if count % 5 == 0:
                            print(json.dumps(d, indent=2, sort_keys=True))
                        print()
                        print('---------------------------- FINISHED ---------------------------')
                        print()
    return d


max_queried = 500

repeats = 1

models = [SvmModel, RfModel, LogModel]

selection_functions = [RandomSelection, MarginSamplingSelection, EntropySelection]

Ks = [250, 125, 50, 25, 10]

d = {}
stopped_at = -1

# print('directory dump including pickle files:', os.getcwd(), np.sort(os.listdir()))
# d = pickle_load('Active-learning-experiment-' + str(stopped_at) + '.pkl')
# print(json.dumps(d, indent=2, sort_keys=True))

d = experiment(d, models, selection_functions, Ks, repeats, stopped_at + 1)
print(d)
results = json.loads(json.dumps(d, indent=2, sort_keys=True))
print(results)


# on affiche les performances avec des graphes


def performance_plot(fully_supervised_accuracy, dic, models, selection_functions, Ks, repeats):
    fig, ax = plt.subplots()
    ax.plot([0, 500], [fully_supervised_accuracy, fully_supervised_accuracy], label='algorithm-upper-bound')
    for model_object in models:
        for selection_function in selection_functions:
            for idx, k in enumerate(Ks):
                x = np.arange(float(Ks[idx]), 500 + float(Ks[idx]), float(Ks[idx]))
                Sum = np.array(dic[model_object][selection_function][k][0])
                for i in range(1, repeats):
                    Sum = Sum + np.array(dic[model_object][selection_function][k][i])
                mean = Sum / repeats
                ax.plot(x, mean, label=model_object + '-' + selection_function + '-' + str(k))
    ax.legend()
    ax.set_xlim([50, 500])
    ax.set_ylim([40, 100])
    ax.grid(True)
    plt.show()


models_str = ['SvmModel', 'RfModel', 'LogModel']
selection_functions_str = ['RandomSelection', 'MarginSamplingSelection', 'EntropySelection']
Ks_str = ['250', '125', '50', '25', '10']
repeats = 1
random_forest_upper_bound = 97.
svm_upper_bound = 94.
log_upper_bound = 92.47
total_experiments = len(models_str) * len(selection_functions_str) * len(Ks_str) * repeats

print('So which is the better model? under the stopping condition and hyper parameters - random forest is the winner!')
performance_plot(random_forest_upper_bound, d, ['RfModel'], selection_functions_str, Ks_str, 1)
performance_plot(svm_upper_bound, d, ['SvmModel'], selection_functions_str, Ks_str, 1)
performance_plot(log_upper_bound, d, ['LogModel'], selection_functions_str, Ks_str, 1)

# nous affiche la meilleure fonction de sélection des données ainsi que le meilleur hyper paramètre

print('So which is the best sample selection function? margin sampling is the winner!')
performance_plot(random_forest_upper_bound, d, ['RfModel'], selection_functions_str, Ks_str, 1)
print()
print('So which is the best k? k=10 is the winner')
performance_plot(random_forest_upper_bound, d, ['RfModel'], ['MarginSamplingSelection'], Ks_str, 1)