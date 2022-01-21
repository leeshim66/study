import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

boston = pd.read_csv('boston.csv')
X = boston.drop('target', axis=1)
y = boston['target']
X_train,X_test,y_train,y_test = train_test_split(X,y)


class RandomForest():
    def __init__(self, n_estimators=100):
        self.ntree = n_estimators

    def fit(self,x,y):
        self.save_model = []
        n_row = len(x)
        n_col = x.shape[1]
        self.model_col = []
        for i in range(self.ntree):
            random_sample = list(set(np.random.choice(n_row, size=n_row, replace=True))) # 랜덤하게 일부 데이터 선택
            random_column = np.random.choice(n_col, size=int(np.sqrt(n_col)), replace=False) # 랜덤하게 일부 변수 선택
            sample_X = x.iloc[random_sample,random_column]
            sample_y = y.iloc[random_sample]
            tree = DecisionTreeRegressor()
            tree.fit(sample_X,sample_y)
            self.save_model.append(tree) # 각 결정트리 모델 저장
            self.model_col.append(random_column) # 결정트리별 선택된 변수 저장
        return self.save_model, self.model_col

    def predict(self,x_test):
        res = []
        for i in range(len(self.save_model)):
            x = x_test.iloc[:,self.model_col[i]]
            pred = self.save_model[i].predict(x)
            res.append(pred)
        res = pd.DataFrame(res)
        return np.mean(res)

my_rf = RandomForest(n_estimators=100)
my_rf.fit(X_train,y_train)
my_rf_pred = my_rf.predict(X_test)
mean_squared_error(y_test,my_rf_pred)


# sklearn에 구현된 RandomForest와 결과 비교
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=100)
rf.fit(X_train,y_train)
rf_pred = rf.predict(X_test)
mean_squared_error(y_test,rf_pred)
