import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

boston = pd.read_csv('boston.csv')
X = boston.drop('target', axis=1)
y = boston['target']
X_train,X_test,y_train,y_test = train_test_split(X,y)


class GradientBoosting():
    def __init__(self, n_estimators=100, alpha=0.1):
        self.n_estimators = n_estimators
        self.lamda = alpha

    def fit(self,x,y):
        self.mean_y = np.mean(y)
        update_y = y.copy()
        n_row = len(x)
        resi_sum = [self.mean_y] * n_row
        update_y = y - resi_sum
        #     result = [np.mean(y)] * len(x)
        self.save_model = []

        for i in range(self.n_estimators) :
            tree = DecisionTreeRegressor()
            tree.fit(x,update_y)
            pred = tree.predict(x)
            resi_sum = resi_sum + self.lamda*pred
            update_y = y - resi_sum
            self.save_model.append(tree)
        #         result = result + alpha*pred
        return self.save_model

    def predict(self,x_test):
        result = [[self.mean_y] * len(x_test)]
        for i in range(len(self.save_model)):
            pred = self.save_model[i].predict(x_test)
            result.append(self.lamda*pred)
        return np.sum(pd.DataFrame(result))

my_GB = GradientBoosting(n_estimators=100, alpha=0.01)
my_GB.fit(X_train,y_train)
my_GB_pred = my_GB.predict(X_test)
mean_squared_error(y_test, my_GB_pred)


# sklearn에 구현된 GradientBoostingRegressor와 결과 비교
from sklearn.ensemble import GradientBoostingRegressor
gb = GradientBoostingRegressor(n_estimators=100, alpha=0.1)
gb.fit(X_train,y_train)
gb_pred = gb.predict(X_test)
mean_squared_error(y_test,gb_pred)
