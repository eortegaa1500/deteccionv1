#Importamos librerias

import torch 
import cv2 
import numpy as np

#Leemos el modelo
model = torch.hub.load('ultralytics/yolov5', 'custom', path = "./best.pt")

#Realizamos videocaptura
cap = cv2.VideoCapture(1)

#Empezamos con el while true

while True:
    #Realizar la lectura de la videocaptura
    ret, frame = cap.read()
    #Realizamos detecciones
    detect = model(frame)
    #Mostramos fps
    cv2.imshow('Detector',np.squeeze(detect.render()))
    #Leer el teclado
    t = cv2.waitKey(5)
    if t == 27:
        break

cap.release()
cv2.destroyAllWindows()



