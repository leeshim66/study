import pandas as pd
import numpy as np

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
