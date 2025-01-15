# -*- coding: utf-8 -*-
"""
A parte mais importante desse projeto é o arquivo haarcascade_frontalface_default.xml.
Haar Cascade é um algoritmo de detecção de objetos utilizado normalmente para identificar faces em imagens
ou vídeos em tempo real.
q
Este modelo foi desenvolvido para detecção de rostos humanos vistos de frente.

Pressione SHIFT + A para fechar a janela de detecção
"""
import cv2

a = cv2.CascadeClassifier(r"C:\Users\gilberto.radecki_sol\.spyder-py3\haarcascade_frontalface_default.xml")
#a = cv2.CascadeClassifier(r"C:\Users\gilberto.radecki_sol\.spyder-py3\haarcascade_frontalface_alt.xml")
#a = cv2.CascadeClassifier(r"C:\Users\gilberto.radecki_sol\.spyder-py3\haarcascade_frontalface_alt2.xml")
b = cv2.VideoCapture(0)

# c_rec se refere ao retângulo
# d_img se refere a  imagem
# e se refere a imagem em escala de cinza
while True:
    c_rec,d_img = b.read()
    e = cv2.cvtColor(d_img, cv2.COLOR_BGR2GRAY)
    f = a.detectMultiScale(e, 1.3, 3)
    
    for(x1, y1, w1, h1) in f:
        cv2.rectangle(d_img, (x1, y1), (x1+w1, y1+h1), (255,0,0), 5)
        
    cv2.imshow('Solvis Image Detector by Gilberto Radecki', d_img)
    h = cv2.waitKey(65)
    if h == 65:
        break
    
b.release()
cv2.destroyAllWindows()

