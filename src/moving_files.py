# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 13:10:43 2024

@author: gilberto.radecki_sol
"""
# ANGRY:    DONE
# DISGUST:  DONE
# FEAR:     DONE
# HAPPY:    DONE
# NEUTRAL:  DONE
# SAD:      DONE
# SURPRISE: TODO

'''
Alegria: Elevação das bochechas. Comissura dos lábios retraída e elevada. 
Rugas na pele debaixo da pálpebra inferior. Ruga entre o nariz e o lábio superior e na zona externa dos olhos.

Nojo: Elevação do lábio superior. Geralmente assimétrica. Rugas no nariz e na área próxima ao lábio superior. 
Rugas na testa. Elevação das bochechas enrugando as pálpebras inferiores.

Raiva: Sobrancelhas baixas, contraídas e em disposição oblíqua. Pálpebra inferior tensa. 
Lábios tensos ou abertos em gesto de grito. Olhar proeminente.

Medo: Elevação e contração das sobrancelhas. Pálpebras superiores e inferiores elevadas. Lábios em tensão. 
Em algumas ocasiões, a boca está aberta.

Surpresa: Elevação das sobrancelhas, dispostas em posição circular. Estiramento da pele debaixo das sobrancelhas. 
Pálpebras abertas (superior elevada e inferior descendente). Descenso da mandíbula.

Tristeza: Ângulos inferiores dos olhos para baixo. Pele das sobrancelhas em forma de triângulo. 
Descenso das comissuras dos lábios que, inclusive, podem estar tremendo.
'''
import os
from matplotlib import pyplot as plt
import numpy as np
import cv2

#Solicitar Diretório
path = input("Informe o diretório\n")
path = os.path.normcase(path)

#Listar os arquivos do diretório
for i in os.listdir(path):
    #Mostrar a primeira figura
    img = cv2.imread(path + '/'+ i)
    plt.imshow(img[:,:,0], cmap='gray')
    plt.show()
    #Perguntar o novo diretório
    new_path = input("Nova pasta:\nAnger - Sad - Fear - Surprise - Neutral - Disgust - Happy\n")
    try:
        print(i)
        os.rename(path + '/'+ i, r"c:/users/gilberto.radecki_sol/.spyder-py3/emotion/test/" + new_path + '/' + i)
        print(len(os.listdir(path)))
    except:
        print("Pasta não encontrada. Informe outra pasta\n")
        new_path = input("Nova pasta:\n")
        os.rename(path + '/'+ i, r"c:/users/gilberto.radecki_sol/.spyder-py3/emotion/test/" + new_path + '/' + i)