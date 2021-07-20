

from keras.models import model_from_json

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()

loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

print( type(loaded_model) )

import cv2
import os

vidcap = cv2.VideoCapture('Payitaht Abdülhamid 1. Bölüm-MyNz3YvSEhk.mp4')

success,image = vidcap.read()
while success:
    success,image = vidcap.read()
    pred = loaded_model.predict_classes(image)
    print(pred)
