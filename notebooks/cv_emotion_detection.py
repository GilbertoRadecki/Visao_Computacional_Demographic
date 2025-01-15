# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 16:12:22 2024

@author: gilberto.radecki_sol

Treinar um modelo de deep learning para classificação de emoções em imagens

https://www.youtube.com/watch?v=P4OevrwTq78

Dataset from: https://www.kaggle.com/msambare/fer2013

angry		3995
disgust		436
fear		4097
happy		7215
neutral		4965
sad		    4830
surprise	3171
Total:		28.709

"""

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten
from keras.layers import Conv2D,MaxPooling2D
import os
from matplotlib import pyplot as plt
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint

IMG_HEIGHT=96 
IMG_WIDTH = 96
batch_size=32

train_data_dir='emotion/train/'
validation_data_dir='emotion/test/'

train_datagen = ImageDataGenerator(
					rescale=1./255,
					rotation_range=30,
					shear_range=0.3,
					zoom_range=0.3,
					horizontal_flip=True,
					fill_mode='nearest')

validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
					train_data_dir,
					color_mode='grayscale',
					target_size=(IMG_HEIGHT, IMG_WIDTH),
					batch_size=batch_size,
					class_mode='categorical',
					shuffle=True)

validation_generator = validation_datagen.flow_from_directory(
							validation_data_dir,
							color_mode='grayscale',
							target_size=(IMG_HEIGHT, IMG_WIDTH),
							batch_size=batch_size,
							class_mode='categorical',
							shuffle=True)

#Verify our generator by plotting a few faces and printing corresponding labels
class_labels=['Angry','Disgust', 'Fear', 'Happy','Neutral','Sad','Surprise']

img, label = train_generator.__next__()

import random

i=random.randint(0, (img.shape[0])-1)
image = img[i]
labl = class_labels[label[i].argmax()]
plt.imshow(image[:,:,0], cmap='gray')
plt.title(labl)
plt.show()
##########################################################


###########################################################
# Create the model
emotion_model = Sequential()

emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(96,96,1)))

emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.1))

emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.1))

emotion_model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.1))

emotion_model.add(Flatten())
emotion_model.add(Dense(512, activation='relu'))
emotion_model.add(Dropout(0.2))

emotion_model.add(Dense(7, activation='softmax'))

emotion_model.compile(optimizer = 'adam', loss='categorical_crossentropy', metrics=['accuracy'])
print(emotion_model.summary())

EmotionCheckpoint = ModelCheckpoint('best_emotion_model.keras', 
                             monitor='val_accuracy',    # Monitorando a acurácia na validação
                             save_best_only=True,       # Apenas salva os melhores pesos
                             mode='max',                # Salvando quando houver a maior acurácia
                             verbose=1)

train_path = "data/train/"
test_path = "data/test"

num_train_imgs = 0
for root, dirs, files in os.walk(train_path):
    num_train_imgs += len(files)
    
num_test_imgs = 0
for root, dirs, files in os.walk(test_path):
    num_test_imgs += len(files)


epochs=50

history=emotion_model.fit(train_generator,
                steps_per_epoch=num_train_imgs//batch_size,
                epochs=epochs,
                validation_data=validation_generator,
                validation_steps=num_test_imgs//batch_size, callbacks=[EmotionCheckpoint])


emotion_model.save('Solvis_001.keras')

#plot the training and validation accuracy and loss at each epoch

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

acc = history.history['accuracy']
#acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
#val_acc = history.history['val_accuracy']

plt.plot(epochs, acc, 'y', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

####################################################################
from keras.models import load_model

#Test the model
emotion_model = load_model('best_emotion_model.keras', compile=True)
#emotion_model = load_model(r"C:\Users\gilberto.radecki_sol\.spyder-py3\Solvis_001.keras")
####################################################################
# #Generate a batch of images
# test_img, test_lbl = validation_generator.__next__()
# predictions=my_model.predict(test_img)

# predictions = np.argmax(predictions, axis=1)
# test_labels = np.argmax(test_lbl, axis=1)

# from sklearn import metrics
# print ("Accuracy = ", metrics.accuracy_score(test_labels, predictions))

# #Confusion Matrix - verify accuracy of each class
# from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# cm = confusion_matrix(test_labels, predictions)
# #print(cm)
# import seaborn as sns
# sns.heatmap(cm, annot=True)

# class_labels=['Angry','Disgust', 'Fear', 'Happy','Neutral','Sad','Surprise']
# #Check results on a few select images
# n=random.randint(0, test_img.shape[0] - 1)
# image = test_img[n]
# orig_labl = class_labels[test_labels[n]]
# pred_labl = class_labels[predictions[n]]
# plt.imshow(image[:,:,0], cmap='gray')
# plt.title("Original label is:"+orig_labl+" Predicted is: "+ pred_labl)
# plt.show()
####################################################################

# Obtendo as previsões do modelo para o conjunto de validação
validation_generator.reset()  # Garante que o gerador esteja no início
y_pred = emotion_model.predict(validation_generator, steps=num_test_imgs // batch_size)
y_pred_classes = np.argmax(y_pred, axis=1)  # Classes preditas

# Obtendo as classes reais do conjunto de validação
y_true = validation_generator.classes  # Classes verdadeiras

# Criando a matriz de confusão
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(y_true[:len(y_pred_classes)], y_pred_classes)

# Exibindo a matriz de confusão
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
disp.plot(cmap='viridis', xticks_rotation=45)
plt.show()
