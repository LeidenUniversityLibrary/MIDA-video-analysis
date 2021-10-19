"""
Evaluate a model's predictions on labeled data.
"""
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.data import Dataset
import os
import sys
import pandas as pd
import pathlib


# Load parameters
model_directory = sys.argv[1]
image_directory = pathlib.Path(sys.argv[2])
metadata_path = sys.argv[3]
results_path = sys.argv[4]

loaded_model = keras.models.load_model(model_directory)
metadata = pd.read_csv(metadata_path)

validation_datagen = keras.preprocessing.image.ImageDataGenerator(
    # rescale=1./255,
    # validation_split=0.01
)

validation_generator = validation_datagen.flow_from_dataframe(
    dataframe=metadata,
    directory=image_directory,
    x_col='filename',
    y_col=['pentagram_visible', 'star_visible'],
    weight_col=None,
    target_size=(150, 150),
    color_mode='rgb',
    class_mode='raw',
    shuffle=False,
    # seed=42,
    # subset='validation',
    interpolation='bilinear'
)

predictions = loaded_model.predict(validation_generator, verbose=1)
predictions_df = pd.DataFrame(predictions, columns=['pentagram_visible', 'star_visible'])
results = metadata.join(predictions_df, rsuffix="_pred")
results = results.assign(pentagram_correct=1 - abs(results.pentagram_visible - results.pentagram_visible_pred.round()),
                         star_correct=1 - abs(results.star_visible - results.star_visible_pred.round()))
results.to_csv(results_path, index=False)
