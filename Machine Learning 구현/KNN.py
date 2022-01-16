import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

prostate = pd.read_csv('prostate.csv')
prostate_train = prostate[prostate['train']=='T'].drop('train',axis=1)
prostate_test = prostate[prostate['train']=='F'].drop('train',axis=1)

X_train = prostate_train.drop('lpsa', axis=1)
X_test = prostate_test.drop('lpsa', axis=1)
y_train = prostate_train['lpsa']
y_test = prostate_test['lpsa']

scale = StandardScaler()
X_train_scaled = scale.fit_transform(X_train)
X_test_scaled = scale.transform(X_test)


class KNN() :
    def __init__(self, n_neighbors):
        self.k = n_neighbors

    def euclidean_matrix(self,x_test) : # x_train과 x_test 사이의 유클리드 거리 행렬 생성
        n_train = len(self.x)
        n_test = len(x_test)
        self.mat = pd.DataFrame(index=range(0,n_train), columns=range(0,n_test))
        for i in range(n_train):
            for j in range(n_test):
                self.mat[j][i] = np.sqrt(sum((x_test.iloc[j]-self.x.iloc[i])**2))
        self.mat.index = self.x.index # x_train과 y_train 인덱스 통일
        return self.mat

    def fit(self,x_train,y_train): # KNN은 비모수 학습
        self.x = x_train
        self.y = y_train

    def predict(self,x_test):
        y_hat = []
        euc = self.euclidean_matrix(x_test.reset_index(drop=True))
        rank_mat = euc.rank().astype(int) # 열별 x_train과의 거리 오름차순
        for i in rank_mat.columns:
            idx = rank_mat[i].sort_values().index[:self.k] # 열별 정렬 후 최솟값(가장 가까운) 인덱스 추출
            y_hat.append(round(np.mean(self.y[idx]),10))
        return y_hat

my_knn = KNN(5)
my_knn.fit(X_train_scaled,y_train)
my_knn_pred = my_knn.predict(X_test_scaled)
mean_squared_error(y_test, my_knn_pred)


# scikit-learn에 구현된 KNearestNeighbors과 결과 비교
from sklearn.neighbors import KNeighborsRegressor

KNN = KNeighborsRegressor(5)
KNN.fit(X_train_scaled,y_train)
knn_pred = KNN.predict(X_test_scaled)
mean_squared_error(y_test,knn_pred)