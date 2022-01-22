import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt

bone_mineral = pd.read_csv('Bone Mineral Density.csv')
male = bone_mineral[bone_mineral['gender']=='male'].groupby('age').mean().reset_index()
female = bone_mineral[bone_mineral['gender']=='female'].groupby('age').mean().reset_index()
X_male = male['age']
y_male = male['spnbmd']
X_female = female['age']
y_female = female['spnbmd']



class Smoothing_Spline():
    def __init__(self, lamda=0):
        self.lamda = lamda

    def fit(self,x,y):
        num_row = len(x)
        N = pd.DataFrame(index=range(0,num_row), columns=range(0,num_row)).fillna(0) # Natural cubic spline
        N[0] = 1
        N[1] = x
        N_pp = pd.DataFrame(index=range(0,num_row), columns=range(0,num_row)).fillna(0) # N의 2차미분 matrix
        Xi_K = x[num_row-1] # 크사이_K
        Xi_K_1 = x[num_row-2] # 크사이_K-1

        for k in range(num_row-2):
            Xi_k = x[k]
            for j in range(num_row):
                if x[j] > Xi_k:
                    N.iloc[j,k+2] = (x[j]-Xi_k)**3 / (Xi_K-Xi_k)
                    N_pp.iloc[j,k+2] = 6*(x[j]-Xi_k)/(Xi_K-Xi_k)
                if x[j] > Xi_K_1:
                    N.iloc[j,k+2] = N.iloc[j,k+2] - (x[j]-Xi_K_1)**3 / (Xi_K-Xi_K_1)
                    N_pp.iloc[j,k+2] = N_pp.iloc[j,k+2] - 6*(x[j]-Xi_K_1)/(Xi_K-Xi_K_1)

        omega = pd.DataFrame(index=range(0,num_row), columns=range(0,num_row)).fillna(0)
        for k in range(num_row):
            for j in range(num_row):
                omega.iloc[j,k] = np.dot(N_pp.iloc[:,j], N_pp.iloc[:,k])

        if self.lamda==0: # lamda값 미설정시 최적 lamda값 탐색
            final_gcv = sys.maxsize
            lamda = 0.1
            for i in range(1000):
                theta = np.dot(np.dot(np.linalg.inv(np.dot(N.T,N) + lamda*omega),N.T),y) # smoothing spline
                result = np.dot(N,theta)
                S_lamda = np.dot(np.dot(N, np.linalg.inv(np.dot(N.T,N)+lamda*omega)),N.T)
                my_gcv = sum(((y-result)/(1-sum(np.diag(S_lamda))/num_row))**2)/num_row
                if final_gcv > my_gcv :
                    final_gcv = my_gcv
                    best_lamda = lamda
                    final_result = result
                lamda += 0.1
            print('best lambda : {}'.format(best_lamda))
            return final_result
        else :
            theta = np.dot(np.dot(np.linalg.inv(np.dot(N.T,N) + self.lamda*omega),N.T),y)
            result = np.dot(N,theta)
            return result


my_spline_male = Smoothing_Spline().fit(X_male,y_male)
my_spline_female = Smoothing_Spline().fit(X_female,y_female)

plt.plot(X_male,my_spline_male)
plt.plot(X_female,my_spline_female)
plt.scatter(X_male,y_male)
plt.scatter(X_female,y_female)


# scipy에 구현된 Spline과 결과 비교
from scipy.interpolate import UnivariateSpline
plt.plot(X_male,UnivariateSpline(X_male,y_male)(X_male))
plt.plot(X_female,UnivariateSpline(X_female,y_female)(X_female))
plt.scatter(X_male,y_male)
plt.scatter(X_female,y_female)
