import pandas as pd
import numpy as np

class knn_alg() :
    def __init__(self, n_neighbors):
        self.k = n_neighbors

    def euclidean(self,x1,x2) :
        euclid = np.sqrt(np.sum((x1-x2)**2))
        return euclid

    def euclidean_matrix(self,x_test) :
        n_train = len(self.x)
        n_test = len(x_test)
        self.mat = pd.DataFrame(index=range(0,n_train), columns=range(0,n_test))
        for i in range(n_train):
            for j in range(n_test):
                self.mat[j][i] = self.euclidean(x_test.iloc[j,:],self.x.iloc[i,:])
        return self.mat

    def fit(self,x,y):
        self.x = x
        self.y = y

    def predict(self,x_test):
        y_hat = []
        euc = self.euclidean_matrix(x_test)
        for i in range(len(euc)):

        return y_hat