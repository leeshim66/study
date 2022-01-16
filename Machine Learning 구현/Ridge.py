import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

spam = pd.read_csv('spam.csv')
X = spam.drop(['col57'], axis=1)
y = spam['col57']
X_train, X_test, y_train, y_test = train_test_split(X,y)


class Ridge() :
    def __init__(self, lamda = 0.1):
        self.lamda = lamda

    def fit(self, x, y) :
        x = x.reset_index(drop=True)
        y = y.reset_index(drop=True)
        x0 = pd.Series([1] * len(x))
        x = pd.concat([x0,x], axis=1)
        x_t = x.T
        I = np.identity(x.shape[1])
        self.beta_hat = np.dot(np.dot(np.linalg.inv(np.dot(x_t,x)+self.lamda*I),x_t),y)

        return self.beta_hat

    def predict(self, x):
        x = x.reset_index(drop=True)
        x0 = pd.Series([1] * len(x))
        x = pd.concat([x0,x], axis=1)

        return np.dot(x,self.beta_hat)

my_ridge = Ridge()
my_ridge.fit(X_train, y_train)
my_ridge_pred = my_ridge.predict(X_test)
mean_squared_error(y_test, my_ridge_pred)
