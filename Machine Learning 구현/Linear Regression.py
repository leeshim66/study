import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

spam = pd.read_csv('spam.csv')
X = spam.drop(['col57'], axis=1)
y = spam['col57']
X_train, X_test, y_train, y_test = train_test_split(X,y)


class Linear_Model() :
    def __init__(self):
        return

    def fit(self, x, y) :
        x = x.reset_index(drop=True)
        y = y.reset_index(drop=True)
        x0 = pd.Series([1] * len(x))
        x = pd.concat([x0,x], axis=1)
        x_t = x.T
        self.beta_hat = np.dot(np.dot(np.linalg.inv(np.dot(x_t,x)),x_t),y)

        return self.beta_hat

    def predict(self, x):
        x = x.reset_index(drop=True)
        x0 = pd.Series([1] * len(x))
        x = pd.concat([x0,x], axis=1)

        return np.dot(x,self.beta_hat)

my_lm = Linear_Model()
my_lm.fit(X_train,y_train)
my_lm_pred = my_lm.predict(X_test)
mean_squared_error(y_test, my_lm_pred)


# scikit-learn에 구현된 LinearRegression과 결과 비교
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train,y_train)
lm_pred = lm.predict(X_test)
mean_squared_error(y_test,lm_pred)
