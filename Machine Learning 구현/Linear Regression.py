import numpy as np
import pandas as pd

class linear_model() :
    def __init__(self):
        return

    def fit(self, x, y) :
        x0 = pd.Series([1] * len(x))
        x = pd.concat([x0,x], axis=1)
        x_t = x.T
        self.beta_hat = np.dot(np.dot(np.linalg.inv(np.dot(x_t,x)),x_t),y)

        return self.beta_hat

    def predict(self, x):
        x0 = pd.Series([1] * len(x))
        x = pd.concat([x0,x], axis=1)

        return np.dot(x,self.beta_hat)
