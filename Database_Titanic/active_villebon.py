import panda as pd


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


# sample
# train
# proba
# sampling
# label
# recommence

# score tracking

