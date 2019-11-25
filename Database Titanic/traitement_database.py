import numpy as np
import csv


data = []
fichier = open('train.csv')
csv_f = csv.reader(fichier)
for row in csv_f:
    data.append(row)

n_data = np.array(data)[1:,:]
T_data = n_data[400:, :]

unlabelled_data = np.delete(T_data, 1, axis = 1)
labelled_data = n_data[:400, :]






if __name__ == '__main__':
    print(unlabelled_data)
    print(labelled_data)



