import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('train.csv')
data.head()
columm = ['Survived']
columm_train = ['Age', 'Pclass', 'Sex', 'Fare']
x = data[columm_train]
y = data[columm]
x['Sex'].isnull().sum()
x['Age'].isnull().sum()
x['Pclass'].isnull().sum()
x['Fare'].isnull().sum()
x['Age']= x['Age'].fillna(x['Age'].median())
x['Age'].isnull().sum()
#no Nan values
#encode the values
d = {'male':0, 'female':1}
x['Sex'] = x['Sex'].apply(lambda x:d[x])
x['Sex'].head()
x
from sklearn import svm
clf = svm.LinearSVC()
clf.fit(x,y)
print(clf.predict(x[0:10]))
print(clf.score(x,y))
test_data = pd.read_csv('test.csv')
col = ['Age', 'Pclass', 'Sex', 'Fare']
cc = ['Survived']
x_test = data[col]
y_test = data[cc]
x_test['Sex'].isnull().sum()
x_test['Pclass'].isnull().sum()
x_test['Age'].isnull().sum()
x_test['Age']=x['Age'].fillna(x_test['Age'].median())

x_test['Fare'].isnull().sum()
#clf.predict(x_test)
x_test['Sex'] = x_test['Sex'].apply(lambda x:d[x])
clf.predict(x_test)
clf.score(x_test,y_test)