# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 17:02:15 2024

@author: gilberto.radecki_sol
"""

import tensorflow as tf

model = tf.keras.models.load_model(r"C:\Users\gilberto.radecki_sol\.spyder-py3\gender_model\Solvis_16.keras")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open('Solvis_16.tflite', 'wb') as f:
    f.write(tflite_model)