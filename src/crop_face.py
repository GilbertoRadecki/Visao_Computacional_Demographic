# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 09:27:21 2024

@author: gilberto.radecki_sol
"""

import cv2
import os

# Carregar o classificador Haar para detecção de rosto
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Selecionar o diretório
path = r"C:\Users\gilberto.radecki_sol\.spyder-py3\emotion\train\sad_"

# Definir o novo diretório
newPath = r"C:\Users\gilberto.radecki_sol\.spyder-py3\emotion\train\sad"
moved = r"C:\Users\gilberto.radecki_sol\.spyder-py3\emotion\train\moved"

for img in os.listdir(path):
# Carregar a imagem
    image = cv2.imread(os.path.join(path, img))
    
    # Converter a imagem para escala de cinza (necessário para o classificador Haar)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detectar rostos na imagem
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3)
    #faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=6, minSize=(30, 30))
    # Verificar se algum rosto foi detectado
    if len(faces) == 0:
        print("Nenhum rosto detectado!")
    else:
        # Para cada rosto detectado, recortar e salvar a imagem do rosto
        for (x, y, w, h) in faces:
            # Recortar o rosto da imagem
            rosto = image[y:y+h, x:x+w]
            # Salvar a imagem do rosto
            cv2.imwrite(os.path.join(newPath, img), rosto)  # Substitua pelo nome desejado para o arquivo de saída
            os.rename(os.path.join(path,img), os.path.join(moved, img))
            # Exibir a imagem do rosto (opcional)
            # cv2.imshow('Rosto', rosto)
            # cv2.waitKey(0)
    
    # Fechar todas as janelas abertas
    # cv2.destroyAllWindows()