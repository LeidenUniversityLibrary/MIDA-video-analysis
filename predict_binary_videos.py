


import tensorflow as tf
import numpy as np
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix
import cv2
import os
import pytube

import numpy as np

loaded_model = keras.models.load_model('symbolism')




predictions = loaded_model.predict( validation_ds )
#matrix = confusion_matrix( validation_ds , predictions )
#print(matrix)

dir = 'Video'


episodes = [ 'https://www.youtube.com/watch?v=2-ZHA6RAdDg',
'https://www.youtube.com/watch?v=ZP9s097xtpU',
'https://www.youtube.com/watch?v=5_uHLp1CU18',
'https://www.youtube.com/watch?v=IQrmmDSCw0s',
'https://www.youtube.com/watch?v=aX-JwSRsAS0',
'https://www.youtube.com/watch?v=j0IYIXafXzY',
'https://www.youtube.com/watch?v=2747ZKwXcfI',
'https://www.youtube.com/watch?v=PQo78fZuY3Y',
'https://www.youtube.com/watch?v=pVQd4Esf8zo',
'https://www.youtube.com/watch?v=MfOpBrYvjYQ' ]


for episode in episodes:
    #print(episode)
    youtube = pytube.YouTube(episode)
    video = youtube.streams.first()
    video.download(dir)


for vid in os.listdir(dir):
    frame_nr = 0
    vidcap = cv2.VideoCapture( os.path.join( dir, vid ))
    success,image = vidcap.read()
    while success:
        frame_nr += 1
        success,image = vidcap.read()
        dim = (150, 150)
        resized = cv2.resize(image , dim, interpolation = cv2.INTER_AREA)
        resized = np.expand_dims( resized , axis=0)
        #print( resized.shape )
        predictions = loaded_model.predict( resized )
        # Class probabilities
        classes = predictions.argmax(axis=-1)
        print( classes, predictions )
        # populate a data frame: store frame number and classification
        
