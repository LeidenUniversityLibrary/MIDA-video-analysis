"""
Predict multiple labels for images in folders
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
images_path = pathlib.Path(sys.argv[2])
results_path = sys.argv[3]

print(f'Starting. Looking for images in {images_path}.')
print(f'Loading the model from {model_directory}.')
loaded_model = keras.models.load_model(model_directory)
loaded_model.summary()


filenames = Dataset.list_files(str(images_path/'*/*'), shuffle=False)


img_height = 150
img_width = 150


def get_label(file_path):
    # convert the path to a list of path components
    parts = tf.strings.split(file_path, os.path.sep)
    # The third to last is the class-directory
    one_hot = parts[-3] == class_names
    # Integer encode the label
    return tf.argmax(one_hot)


def decode_img(img):
    # convert the compressed string to a 3D uint8 tensor
    img = tf.io.decode_jpeg(img, channels=3)
    # resize the image to the desired size
    return tf.image.resize(img, [img_height, img_width])


def process_path(file_path):
    # label = get_label(file_path)
    # load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = tf.expand_dims( decode_img(img) , axis=0)
    return img, file_path

# Create Dataset of images from the filenames Dataset
images_ds = filenames.map(process_path, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)

# Create predictions using the loaded model
predictions = loaded_model.predict(images_ds, verbose=1)
# predictions is a list of ndarrays, one ndarray for each model output
print("Length of predictions:", len(predictions))
print("Shape of first item in predictions:", predictions[0].shape)
print("Type of first item in predictions:", type(predictions[0]))
# We don't need to convert the ndarrays to a dataset first,
# we can load them into a DataFrame directly.
# Load results into a DataFrame to save to CSV

data = {"filename": filenames.as_numpy_iterator()}
for c, p in zip(loaded_model.output_names, predictions):
    data[c] = p.ravel() # flatten the prediction arrays

results = pd.DataFrame(data)
results.loc[:, "filename"] = results.loc[:, "filename"].str.decode('utf-8').str.replace(str(images_path), "")
results.to_csv(results_path, index=False)
