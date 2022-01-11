import pandas as pd
import numpy as np

class knn_alg() :
    def __init__(self, n_neighbors):
        self.k = n_neighbors

    def euclidean(self,x1,x2) :
        euclid = np.sqrt(np.sum((x1-x2)**2))
        return euclid

    def euclidean_matrix(self,test_x) :
        n_train = len(self.x)
        n_test = len(test_x)
        self.mat = pd.DataFrame(index=range(0,n_train), columns=range(0,n_test))
        for i in range(n_train):
            for j in range(n_test):
                self.mat[j][i] = self.euclidean(test_x.iloc[j,:],self.x.iloc[i,:])
        self.mat.index = self.x.index
        return self.mat

    def fit(self,x,y):
        self.x = x
        self.y = y

    def predict(self,x_test):
        y_hat = []
        euc = self.euclidean_matrix(x_test)
        rank_mat = euc.rank(ascending=False).astype(int)
        for i in rank_mat.columns:
            idx = rank_mat[i].sort_values().index[:self.k]
            if np.mean(self.y[idx])>0.5 :
                y_hat.append(1)
            else :
                y_hat.append(0)
        return y_hat
