# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 13:13:40 2024

@author: gilberto.radecki_sol
"""
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Dropout, BatchNormalization, Flatten
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
import tensorflow as tf

# Fixar aleatoriedade para reproducibilidade
np.random.seed(33)
tf.random.set_seed(33)

# Parâmetros gerais
height = 96
width = 96
path = r"C:\Users\gilberto.radecki_sol\.spyder-py3\genderAgeFaces"
images, gender = [], []

# Carregar imagens
for img in os.listdir(path):
    genders = int(img.split("_")[1])
    img_path = os.path.join(path, img)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Converter para escala de cinza
    img = cv2.equalizeHist(img)  # Equalização do histograma
    img = cv2.resize(img, (height, width))
    img = img / 255.0  # Normalização entre 0 e 1
    images.append(img)
    gender.append(genders)

# Converter para numpy arrays
images = np.array(images).reshape(-1, height, width, 1)  # Adicionar canal 1 (grayscale)
gender = np.array(gender)

# Dividir os dados
x_train, x_test, y_train, y_test = train_test_split(images, gender, test_size=0.2, random_state=33)

# Augmentação de dados
from tensorflow.keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
)
datagen.fit(x_train)

# Definir modelo otimizado
def build_gender_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(height, width, 1)))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2, 2)))
    
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2, 2)))
    
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2, 2)))
    
    model.add(Flatten())
    model.add(Dropout(0.4))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid', name='gender'))
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Construir o modelo
gender_model = build_gender_model()

# Callbacks
GenderCheckpoint = ModelCheckpoint(
    'best_gender_model.keras', monitor='val_accuracy', save_best_only=True, mode='max', verbose=1
)
EarlyStoppingGender = EarlyStopping(monitor='val_accuracy', patience=5, mode='max', verbose=1)
ReduceLROnPlateauGender = ReduceLROnPlateau(
    monitor='val_accuracy', factor=0.5, patience=3, min_lr=1e-6, verbose=1
)

# Treinamento
history_gender = gender_model.fit(
    datagen.flow(x_train, y_train, batch_size=32),
    validation_data=(x_test, y_test),
    epochs=50,
    callbacks=[GenderCheckpoint, EarlyStoppingGender, ReduceLROnPlateauGender]
)

# Avaliação do modelo
best_model = tf.keras.models.load_model('best_gender_model.keras')
predictions = best_model.predict(x_test)
y_pred = (predictions >= 0.5).astype(int).flatten()

accuracy = accuracy_score(y_test, y_pred)
print("Acurácia Final: {:.2f}%".format(accuracy * 100))

# Matriz de confusão
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Male", "Female"])
disp.plot()
plt.show()
