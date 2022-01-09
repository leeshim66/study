import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

x_train = np.array([1., 2., 3., 4., 5., 6.])
y_train = np.array([9., 16., 23., 30., 37., 44.])

W = 0.0
b = 0.0

n_data = len(x_train)

epochs = 1500
learning_rate = 0.01

for i in range(epochs):
    hypothesis = x_train * W + b
    cost = np.sum((hypothesis - y_train) ** 2) / n_data
    gradient_w = np.sum((W * x_train - y_train + b) * 2 * x_train) / n_data
    gradient_b = np.sum((W * x_train - y_train + b) * 2) / n_data

    W -= learning_rate * gradient_w
    b -= learning_rate * gradient_b

    if i % 100 == 0:
        print('Epoch ({:10d}/{:10d}) cost: {:10f}, W: {:10f}, b:{:10f}'.format(i, epochs, cost, W, b))

print('W: {:10f}'.format(W))
print('b: {:10f}'.format(b))
print('result : ', x_train * W + b)


# 데이터 로드
fashion_mnist = tf.keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test,y_test) = fashion_mnist.load_data()

X_valid,X_train = X_train_full[:10000]/255, X_train_full[10000:]/255
y_valid,y_train = y_train_full[:10000], y_train_full[10000:]
X_test = X_test/255

class_names = ['T-shirt/top', 'Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=[28,28]))
model.add(tf.keras.layers.Dense(300, activation='relu'))
model.add(tf.keras.layers.Dense(100, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=10, validation_data=(X_valid,y_valid))
model.evaluate(X_test,y_test)

# 손실함수, 정확도 시각화
pd.DataFrame(history.history).plot(figsize=(8,5))
plt.grid(True)
plt.gca().set_ylim(0,1)
plt.show()


# 회귀 문제에 딥러닝 적용
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

housing = fetch_california_housing()

X_train_full, X_test, y_train_full, y_test = train_test_split(housing.data, housing.target)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.fit_transform(X_valid)
X_test = scaler.fit_transform(X_test)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(30, activation='relu', input_shape=X_train.shape[1:]))
model.add(tf.keras.layers.Dense(1))
model.compile(loss='mean_squared_error',optimizer='sgd')
history = model.fit(X_train, y_train, epochs=20, validation_data=(X_valid,y_valid))
model.evaluate(X_test, y_test)

y_pred = model.predict(X_test)


### 비순차적 신경망 - 와이드&딥 신경망 : 입력의 일부가 출력층에 바로 연결
input_ = tf.keras.layers.Input(shape=X_train.shape[1:])
hidden1 = tf.keras.layers.Dense(30, activation='relu')(input_) # input_을 함수처럼 호출하므로 함수형 API라 한다.
hidden2 = tf.keras.layers.Dense(30, activation='relu')(hidden1)
concat = tf.keras.layers.Concatenate()([input_,hidden2])
output = tf.keras.layers.Dense(1)(concat)
model = tf.keras.Model(inputs=[input_], outputs=[output])