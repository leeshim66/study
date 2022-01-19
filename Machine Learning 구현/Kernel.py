import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

bone_mineral = pd.read_csv('Bone Mineral Density.csv')
male = bone_mineral[bone_mineral['gender']=='male'].sort_values('age')
X_male = male['age']
y_male = male['spnbmd']


class Kernel():
    def __init__(self, alpha=1, kernel='gaussian'):
        self.lamda = alpha
        self.kernel = eval('self.{}'.format(kernel))

    def gaussian(self, t):
        return 1/np.sqrt(2*np.pi)*np.exp(-t**2/2)

    def epanechnikov(self, t):
        if abs(t)<=1 :
            return 3/4*(1-t**2)
        else :
            return 0

    def tri_cube(self, t):
        if abs(t)<=1 :
            return (1-abs(t)**3)**3
        else :
            return 0

    def fit(self, x,y):
        self.x = x
        self.y = y

    def predict(self, x_test):
        y_hat = []
        x_test = x_test.reset_index(drop=True)
        for i in range(len(x_test)):
            sik = pd.Series(abs(self.x-x_test[i])/self.lamda)
            K_lamda = sik.apply(lambda x:self.kernel(x))
            aa = np.dot(K_lamda,self.y) / sum(K_lamda)
            y_hat.append(round(aa,10))
        return y_hat

my_kernel_gaussian = Kernel(alpha=1, kernel='gaussian')
my_kernel_gaussian.fit(X_male,y_male)
my_kernel_gaussian_pred = my_kernel_gaussian.predict(X_male)
my_kernel_epanechnikov = Kernel(alpha=1, kernel='epanechnikov')
my_kernel_epanechnikov.fit(X_male,y_male)
my_kernel_epanechnikov_pred = my_kernel_epanechnikov.predict(X_male)
my_kernel_tricube = Kernel(alpha=1, kernel='tri_cube')
my_kernel_tricube.fit(X_male,y_male)
my_kernel_tricube_pred = my_kernel_tricube.predict(X_male)


plt.figure(figsize=(10,6))
plt.scatter(X_male,y_male, s=20)
plt.plot(X_male,my_kernel_gaussian_pred, color='yellow', label='gaussian')
plt.plot(X_male,my_kernel_epanechnikov_pred, color='red', label='epanechnikov')
plt.plot(X_male,my_kernel_tricube_pred, color='green', label='tri_cube')
plt.legend()
plt.show()
