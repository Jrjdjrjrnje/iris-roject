import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers

iris = load_iris()
X, y = iris.data, iris.target


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = keras.Sequential([
    layers.Dense(10, activation='relu', input_shape=(4,)),
    layers.Dense(10, activation='relu'),                   
    layers.Dense(3, activation='softmax')                  
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

print("Начинаем обучение нейросети...")
model.fit(X_train, y_train, epochs=50, batch_size=8, verbose=0)
print("Обучение завершено.")

loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Точность модели на тесте: {accuracy:.2f}")

model.save("iris_model.keras")
print("Модель сохранена в файл iris_model.keras")
