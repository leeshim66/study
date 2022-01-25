import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

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
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)

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


# input 2개, output 2개(보조 출력. ex.출력1:얼굴 표정, 출력2:안경 여부)
X_train_A, X_train_B = X_train[:,:5], X_train[:,2:] # 1~5 column을 첫 번째 input으로, 2~7column을 두 번째 input으로 학습
X_valid_A, X_valid_B = X_valid[:,:5], X_valid[:,2:]
X_test_A, X_test_B = X_test[:,:5], X_test[:,2:]
X_new_A, X_new_B = X_test_A[:3], X_test_B[:3]

input_A = tf.keras.layers.Input(shape=[5], name='wide_input')
input_B = tf.keras.layers.Input(shape=[6], name='deep_input')
hidden1 = tf.keras.layers.Dense(30, activation='relu')(input_B)
hidden2 = tf.keras.layers.Dense(30, activation='relu')(hidden1)
concat = tf.keras.layers.Concatenate()([input_A,hidden2]) # input_A는 얕은 학습, input_B는 깊은 학습
output = tf.keras.layers.Dense(1, name='output')(concat)
aux_output = tf.keras.layers.Dense(1, name='aux_output')(hidden2) # 보조 ouput
model = tf.keras.Model(inputs=[input_A,input_B], outputs=[output, aux_output])
model.compile(loss=['mse','mse'], loss_weights=[.9,.1], optimizer=tf.keras.optimizers.SGD(lr=1e-3))

history = model.fit((X_train_A, X_train_B), [y_train,y_train],
                     epochs=10, validation_data=((X_valid_A,X_valid_B),(y_valid,y_valid)))
total_loss,main_loss,aux_loss = model.evaluate([X_test_A,X_test_B],[y_test,y_test])

y_pred_main, y_pred_aux = model.predict([X_new_A,X_new_B])

# 모델 저장 및 로드
model.save('my_keras_model.h5')
mod = tf.keras.models.load_model('my_keras_model.h5')


# 콜백
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint('my_keras_model.h5', save_best_only=True)
early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True) # 지정한 patience 횟수 이상 모델 개선이 없으면 학습 중지
history = model.fit([X_train_A,X_train_B],[y_train,y_train], epochs=100,
                    validation_data=((X_valid_A,X_valid_B),(y_valid,y_valid)),
                    callbacks=[checkpoint_cb,early_stopping_cb])
model = tf.keras.models.load_model('my_keras_model.h5') # save_best_only=True이므로 최상의 모델로 복원


# scikit-learn 추정기 형태로 keras 객체 사용
def build_model(n_hidden=1, n_neurons=30, learning_rate=3e-3, input_shape=[8]):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=input_shape))
    for layer in range(n_hidden):
        model.add(tf.keras.layers.Dense(n_neurons,activation='relu'))
    model.add(tf.keras.layers.Dense(1))
    optimizer = tf.keras.optimizers.SGD(lr=learning_rate)
    model.compile(loss='mse', optimizer=optimizer)
    return model

keras_reg = tf.keras.wrappers.scikit_learn.KerasRegressor(build_model) # 사이킷런 추정기처럼 객체 사용 가능

keras_reg.fit(X_train,y_train,epochs=100,validation_data=(X_valid,y_valid),
              callbacks=[tf.keras.callbacks.EarlyStopping(patience=5)])
mse = keras_reg.score(X_test,y_test)
y_pred = keras_reg.predict(X_test)


# Randomized Search를 이용하여 최적 파라미터 탐색
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error

param_distribs = {'n_hidden':[1,2,3,4,5], 'n_neurons':range(1,100)}  # 하이퍼파라미터의 수가 많으므로 Grid Search 대신 사용

rnd_search_cv = RandomizedSearchCV(keras_reg, param_distribs, n_iter=10, cv=5) # parameter를 랜덤으로 설정후 최적해 도출, n_iter=학습횟수
rnd_search_cv.fit(X_train, y_train, epochs=100, validation_data=(X_valid,y_valid),
                  callbacks=[tf.keras.callbacks.EarlyStopping(patience=5)]) # cv를 사용하므로 valid set은 eraly stopping에만 사용됨
model = rnd_search_cv.best_estimator_.model

print(rnd_search_cv.best_params_)
print(rnd_search_cv.best_score_)

model = rnd_search_cv.estimator.model
mean_squared_error(y_test,model.predict(X_test))