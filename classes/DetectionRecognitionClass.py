from PIL import Image
import cv2 as cv
from keras.models import load_model
import numpy as np
import os

# Listar directorios
faces_path = "./faces/train"
class_names = os.listdir(faces_path)

# Cargar modelo entrenado
model = load_model('./models/VGG16_Final_Model_Face.keras')
model = load_model('./models/Sequential_Final_Model_Face.keras')

# Cargar clasificador en cascada de OpenCV para detección de rostros frontales
haar_cascade = cv.CascadeClassifier('./haarcascades/haarcascade_frontalface_default.xml')

img = cv.imread('./faces/Momoa.png')

faces_rect = haar_cascade.detectMultiScale(img, 1.3, 5)

for(x, y, w, h) in faces_rect:
    faces_roi = img[y:y+h, x:x+w]
    faces_roi = cv.resize(faces_roi, (224, 224))
    im = Image.fromarray(faces_roi, 'RGB')
    img_array = np.array(im)
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array)
    threshold = 0.4
    name = None

    for i in range(len(pred[0])):
        if pred[0][i] > threshold:
            name = class_names[i]
            break

    print(name)
    cv.rectangle(img, (x, y+h), (x+w+40, y+h+60), (0, 255, 0), -1)
    cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), thickness=2)
    cv.putText(img, name, (x, y+h+20), 2, 0.7, (0, 0, 0), thickness=1)

cv.imshow('Detected Face', img)
cv.waitKey(0)