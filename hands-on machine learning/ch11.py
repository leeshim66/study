import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# 가중치 초기화, 배치 정규화
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=[28,28]),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(300, activation='elu', kernel_initializer='he_normal'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(100, activation='elu', kernel_initializer='he_normal'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(10, activation='softmax')
])
model.summary()

# 활성화함수 이전에 배치 정규화 적용





# 전이 학습
model_A = tf.keras.models.load_model('model.h5')
model_A_clone = tf.keras.models.clone_model(model_A) # model_A의 정보를 유지하기 위해 복사
model_A_clone.set_weights(model_A.get_weights()) # clone_model()은 가중치를 복제하지 않으므로 설정을 해줘야함

model_B_on_A = tf.keras.models.Sequential(model_A.layers[:-1]) # 출력층만 제거
model_B_on_A.add(tf.keras.models.Dense(1, activation='sigmoid'))


