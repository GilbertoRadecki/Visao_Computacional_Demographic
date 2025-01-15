# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 13:18:28 2024

@author: gilberto.radecki_sol
"""

# https://youtu.be/rdjWDAYt98s
"""
Train deep learning models to predict age and gender.

Datase from here: https://susanqq.github.io/UTKFace/

24.109 imagens, inicialmente.
No entanto, para evitar colapso decorrente do esgotamento de memória, removi do dataset 1813 imagens (crianças de até 2 anos)

22.295 imagens restantes.
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
from keras.losses import MeanAbsoluteError
mae = MeanAbsoluteError

from keras.models import Sequential,load_model,Model
from keras.layers import Conv2D,MaxPool2D,Dense,Dropout,BatchNormalization,Flatten,Input
from sklearn.model_selection import train_test_split
from keras.callbacks import History
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from keras_visualizer import visualizer
from keras import models, layers

history = History()

heigth=96
width=96
path = r"C:\Users\gilberto.radecki_sol\.spyder-py3\genderAgeFaces"
images = []
age = []
gender = []
for img in os.listdir(path):
  ages = img.split("_")[0]
  genders = int(img.split("_")[1])
  img = cv2.imread(str(path)+"/"+str(img))
  img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
  img = cv2.resize(img, (heigth,width), interpolation=cv2.INTER_AREA)
  img = img / 255.0
  images.append(np.array(img))
  age.append(np.array(ages))
  gender.append(np.array(genders))
  
age = np.array(age,dtype=np.int64)
images = np.array(images)   
gender = np.array(gender,np.uint64)

x_train_age, x_test_age, y_train_age, y_test_age = train_test_split(images, age, test_size=0.2, random_state=33)

x_train_gender, x_test_gender, y_train_gender, y_test_gender = train_test_split(images, gender, test_size=0.2, random_state=33)


##################################################
#Define age model and train. 
##################################################

age_model = Sequential()
#age_model.add(Conv2D(64, padding='same', kernel_size=3, activation='relu', input_shape=(heigth,width,3)))
age_model.add(Conv2D(128, padding='same', kernel_size=3, activation='relu', input_shape=(heigth,width,3)))
age_model.add(MaxPool2D(pool_size=3, padding='same', strides=2))

#age_model.add(Conv2D(128, padding='same', kernel_size=3, activation='relu'))
age_model.add(Conv2D(256, padding='same', kernel_size=3, activation='relu'))
age_model.add(MaxPool2D(pool_size=3, padding='same', strides=2))
              
#age_model.add(Conv2D(256, padding='same', kernel_size=3, activation='relu'))
age_model.add(Conv2D(512, padding='same', kernel_size=3, activation='relu'))
age_model.add(MaxPool2D(pool_size=3, padding='same', strides=2))

#age_model.add(Conv2D(512, padding='same', kernel_size=3, activation='relu'))
age_model.add(Conv2D(1024, padding='same', kernel_size=3, activation='relu'))
age_model.add(MaxPool2D(pool_size=3, padding='same', strides=2))

age_model.add(Flatten())
age_model.add(Dropout(0.2))
age_model.add(Dense(512, activation='relu'))

age_model.add(Dense(1, activation='linear', name='age'))

print(age_model.summary())
#visualizer(age_model, file_name="age_model", file_format='png', view=True)

age_model.compile(optimizer='adam', loss='mse' , metrics=['mae'])
          

AgeCheckpoint = ModelCheckpoint(
    'best_age_model.keras', 
    monitor='val_mae',  # Monitorando o erro médio absoluto na validação
    save_best_only=True,  # Apenas salva os melhores pesos
    mode='min',  # Salvando quando houver o menor erro médio absoluto
    verbose=1
)

EarlyStoppingAge = EarlyStopping(
    monitor='val_mae',  # Monitorando o erro médio absoluto na validação
    patience=5,  # Para de treinar após 5 épocas sem melhora
    mode='min',  # Procurando minimizar o erro
    verbose=1
)

history_age = age_model.fit(
    x_train_age, y_train_age,
    validation_data=(x_test_age, y_test_age),
    epochs=50,
    callbacks=[AgeCheckpoint, EarlyStoppingAge]
)

################################################################
#Define gender model and train
################################################################
gender_model = Sequential()

#gender_model.add(Conv2D(36, padding='same', kernel_size=3, activation='relu', input_shape=(heigth,width,3)))
gender_model.add(Conv2D(72, padding='same', kernel_size=3, activation='relu', input_shape=(heigth,width,3)))

gender_model.add(MaxPool2D(pool_size=3, padding='same', strides=2))
#gender_model.add(Conv2D(64, padding='same', kernel_size=3, activation='relu'))
gender_model.add(Conv2D(128, padding='same', kernel_size=3, activation='relu'))
gender_model.add(MaxPool2D(pool_size=3, padding='same', strides=2))

#gender_model.add(Conv2D(128, kernel_size=3, padding='same', activation='relu'))
gender_model.add(Conv2D(256, kernel_size=3, padding='same', activation='relu'))
gender_model.add(MaxPool2D(pool_size=3, padding='same', strides=2))

#gender_model.add(Conv2D(256, kernel_size=3, padding='same', activation='relu'))
gender_model.add(Conv2D(512, kernel_size=3, padding='same', activation='relu'))
gender_model.add(MaxPool2D(pool_size=3, padding='same', strides=2))

gender_model.add(Conv2D(512, kernel_size=3, padding='same', activation='relu'))
gender_model.add(MaxPool2D(pool_size=3, padding='same', strides=2))

gender_model.add(Flatten())
gender_model.add(Dropout(0.2))
#gender_model.add(Dense(512, activation='relu'))
gender_model.add(Dense(1024, activation='relu'))
gender_model.add(Dense(1, activation='sigmoid', name='gender'))

#from keras_visualizer import visualizer
#visualizer(gender_model, file_format='png', view=True) 

gender_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

GenderCheckpoint = ModelCheckpoint(
    'best_gender_model.keras', 
    monitor='val_accuracy',  # Monitorando a acurácia na validação
    save_best_only=True,  # Apenas salva os melhores pesos
    mode='max',  # Salvando quando houver a maior acurácia
    verbose=1
)

EarlyStoppingGender = EarlyStopping(
    monitor='val_accuracy',  # Monitorando a acurácia na validação
    patience=5,  # Para de treinar após 5 épocas sem melhora
    mode='max',  # Procurando maximizar a acurácia
    verbose=1
)

history_gender = gender_model.fit(
    x_train_gender, y_train_gender,
    validation_data=(x_test_gender, y_test_gender),
    epochs=50,
    callbacks=[GenderCheckpoint, EarlyStoppingGender]
)
############################################################

# history = history_age

# #plot the training and validation accuracy and loss at each epoch
# loss = history.history['loss']
# val_loss = history.history['val_loss']
# epochs = range(1, len(loss) + 1)
# plt.plot(epochs, loss, 'y', label='Training loss')
# plt.plot(epochs, val_loss, 'r', label='Validation loss')
# plt.title('Training and validation loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()

# acc = history.history['accuracy']
# #acc = history.history['accuracy']
# val_acc = history.history['val_accuracy']
# #val_acc = history.history['val_accuracy']

# plt.plot(epochs, acc, 'y', label='Training acc')
# plt.plot(epochs, val_acc, 'r', label='Validation acc')
# plt.title('Training and validation accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.show()

from keras.models import load_model

####################################################################
#Test the gender model
gender_model = load_model(r"best_gender_model.keras", compile=True)

predictions = gender_model.predict(x_test_gender)
y_pred_gender = (predictions>= 0.5).astype(int)[:,0]

from sklearn import metrics
print ("Acurácia (Gênero) = ", metrics.accuracy_score(y_test_gender, y_pred_gender))

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, mean_absolute_error

cm = confusion_matrix(y_test_gender,y_pred_gender, labels=[0, 1])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Male","Female"])
disp.plot()
plt.show()

####################################################################

#Test the age model
my_model_age = load_model(r"Solvis_15.keras", compile=True)

predictions = my_model_age.predict(x_test_age)
y_pred_age = (predictions>= 0.5).astype(int)[:,0]
y_pred_age = predictions[:,0]
y_pred_age = y_pred_age.astype(int)
print ("Erro médio absoluto (Idade) = ", metrics.mean_absolute_error(y_test_age, y_pred_age))

#Confusion Matrix - verify accuracy of each class
from sklearn.metrics import confusion_matrix
import seaborn as sns
cm=confusion_matrix(y_test_age, y_pred_age)  
sns.heatmap(cm)
plt.show()
