# Load and apply the trained model
import tensorflow as tf
import cv2


model_path = sys.argv[1]
loaded_model = tf.keras.models.load_model(model_path)
print("Loaded model from disk")
print( type(loaded_model) )


vidcap = cv2.VideoCapture('Payitaht Abdülhamid 1. Bölüm-MyNz3YvSEhk.mp4')

success,image = vidcap.read()
while success:
    success,image = vidcap.read()
    pred = loaded_model.predict_classes(image)
    print(pred)
