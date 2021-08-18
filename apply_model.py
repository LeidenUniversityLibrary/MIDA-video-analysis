# Load and apply the trained model
import tensorflow as tf
import numpy as np
from tensorflow import keras
# from sklearn.metrics import classification_report, confusion_matrix
import cv2
import os
# import pytube
import sys

import numpy as np

model_directory = sys.argv[1]
video_file = sys.argv[2]

loaded_model = keras.models.load_model(model_directory)




# predictions = loaded_model.predict( validation_ds )
#matrix = confusion_matrix( validation_ds , predictions )
#print(matrix)

# dir = 'Video'


# episodes = [ 'https://www.youtube.com/watch?v=2-ZHA6RAdDg',
# 'https://www.youtube.com/watch?v=ZP9s097xtpU',
# 'https://www.youtube.com/watch?v=5_uHLp1CU18',
# 'https://www.youtube.com/watch?v=IQrmmDSCw0s',
# 'https://www.youtube.com/watch?v=aX-JwSRsAS0',
# 'https://www.youtube.com/watch?v=j0IYIXafXzY',
# 'https://www.youtube.com/watch?v=2747ZKwXcfI',
# 'https://www.youtube.com/watch?v=PQo78fZuY3Y',
# 'https://www.youtube.com/watch?v=pVQd4Esf8zo',
# 'https://www.youtube.com/watch?v=MfOpBrYvjYQ' ]


# for episode in episodes:
#     #print(episode)
#     youtube = pytube.YouTube(episode)
#     video = youtube.streams.first()
#     video.download(dir)

def predict_video(video_file: str):
    frame_nr = 0
    vidcap = cv2.VideoCapture(video_file)
    success,image = vidcap.read()
    max_pred = 0
    while success:
        dim = (150, 150)
        resized = cv2.resize(image , dim, interpolation = cv2.INTER_AREA)
        resized_data = np.expand_dims( resized , axis=0)
        #print( resized.shape )
        predictions = loaded_model.predict( resized_data )
        # Class probabilities
        # classes = predictions.argmax(axis=-1)
        # print( classes, predictions )
        pred = predictions[0][0]
        max_pred = max(max_pred, pred)
        if pred == max_pred:
            out_file = f'./tmp/f{frame_nr}_{round(pred, 3)}.jpg'
            print(out_file, cv2.imwrite(out_file, resized))
        print( frame_nr, predictions )
        # populate a data frame: store frame number and classification
        frame_nr += 1
        success,image = vidcap.read()


# for vid in os.listdir(dir):
#     frame_nr = 0
#     vidcap = cv2.VideoCapture( os.path.join( dir, vid ))
#     success,image = vidcap.read()
#     while success:
#         frame_nr += 1
#         success,image = vidcap.read()
#         dim = (150, 150)
#         resized = cv2.resize(image , dim, interpolation = cv2.INTER_AREA)
#         resized = np.expand_dims( resized , axis=0)
#         #print( resized.shape )
#         predictions = loaded_model.predict( resized )
#         # Class probabilities
#         classes = predictions.argmax(axis=-1)
#         print( classes, predictions )
#         # populate a data frame: store frame number and classification
        
# predict_video(video_file)

validation_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "/Users/companjenba/surfdrive/Projecten/MIDA/Mustafa/episodes/49/frames_by_class", labels='inferred', label_mode='int',
    class_names=None, color_mode='rgb', image_size=(256,
    256), shuffle=True, seed= 42 , #validation_split= 0.1, subset= 'validation',
    interpolation='bilinear', follow_links=False
)
size = (150, 150)
validation_ds = validation_ds.map(lambda x, y: (tf.image.resize(x, size), y))

print(validation_ds.cardinality())


predictions = loaded_model.predict(validation_ds, steps=2)
print(predictions)
