import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing

data = pd.read_csv("train.csv")
cols_throw = ['Name', 'Ticket', 'Cabin', 'Sex', 'Embarked']
# cols_keep = [col for col in data.columns and col not in cols_throw]
# print(cols_keep)
data = data.drop(cols_throw, axis=1)


data.dropna(inplace=True)
print(data.shape)

X = data[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare','sex']].values
y = data['Survived']

# #Create an encoder
# sex_encoder = preprocessing.LabelEncoder()
#
# # Fit the encoder to the train data so it knows that male = 1
# sex_encoder.fit(X['Sex'])
#
# # Apply the encoder to the training data
# X['male'] = sex_encoder.transform(data['Sex'])




X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

random_forest = RandomForestClassifier(random_state=0, n_jobs=-1)
random_forest.fit(X_train, y_train)
y_predict = random_forest.predict(X_test)

M = confusion_matrix(y_test, y_predict)
print('matrice de confusion')
print(M)










if __name__ == '__main__':



