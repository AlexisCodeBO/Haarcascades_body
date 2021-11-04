import cv2
import numpy as np

body_cascade = cv2.CascadeClassifier("haarcascade_fullbody.xml")

# Case Sensitive letters
#vid = cv2.VideoCapture("H:\ALEXIS2\Inteligencia Artificial\Proyfinal_ia\Videos_pexels\Pexels_Videos_2670.mp4")
vid = cv2.VideoCapture("H:\ALEXIS2\Inteligencia Artificial\Proyfinal_ia\Videos_pexels\Pexels_Videos_3030.mp4")
# Captura de cámara
# Usar un dispositivo externo
# x vid= cv2.VideoCapture(0)
# x vid= cv2.VideoCapture(1)

while True:
    # Leer imagenes de la webcam
    response, color_image= vid.read()

    
    #Si la respuesta es falsa vamos a cortar el while loop( significa que el sistema no esta preparado para la R del video)
    if response==False:
        break
       
    # Convertir imagen a gray scale
    grey_image=cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    
    # Detectar el cuerpo
    body = body_cascade.detectMultiScale(grey_image, 1.3, 5)

    #coords=np.zeros((body,10), dtype=int)

    #1 es MinNeighbors
    # para mostrar ese rectángulo (cuerpo detectado)
    for (x, y, w, h) in body:
        cv2.rectangle(color_image, (x,y), (x+w, y+h), (0, 255, 255), 2 )
        
    #Dibujar punto (x,y) uniendolos por lineas
    #for i in range(0, body):
    #    coords[i] = (coords.part(i).x, coords.part(i).y)  

    #for (x, y) in coords:
    #	cv2.line(color_image, (x, y), 1, (0, 255, 255), 2)

    #x,y son las coordenadas
    #(x+w) , (y+h) es el rectángulo formado por x, y coordenadas
    # 0,255,255 es el color RGB, puede dar cualquier color
    # para mostrar ese rectángulo (cuerpo detectado) 2 es el ancho del rectángulo hueco que se formará
    cv2.imshow('Detector', color_image )
    # detector es el nombre que vendrá en la pantalla
    # y queremos que nuestra imagen se muestre como color, así que color_image
    # si quieres mostrar una imagen gris puede hacer grey_image
    if cv2.waitKey(1) & 0xFF== ord('q'):
        break
# 0xFF es el blanco y cuando se presiona Q  romperá
# FF es el blanco y cuando q se presiona se romperá
vid.release()
cv2.destroyAllWindows()

