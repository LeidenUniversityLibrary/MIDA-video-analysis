# Predict binary labels for images in folders
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.data import Dataset
import os
import click
import pandas as pd
import pathlib

img_height = 150
img_width = 150


def decode_img(img):
    # convert the compressed string to a 3D uint8 tensor
    img = tf.io.decode_jpeg(img, channels=3)
    # resize the image to the desired size
    return tf.image.resize(img, [img_height, img_width])


def process_path(file_path):
    # load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = tf.expand_dims( decode_img(img) , axis=0)
    return img

@click.command()
@click.option('-m', '--model-directory', type=click.Path(exists=True, dir_okay=True), required=True)
@click.option('-i', '--images-directory', type=click.Path(exists=True, dir_okay=True, path_type=pathlib.Path), required=True)
@click.option('-p', '--pattern', default='*/*/*/*.jpg', show_default=True)
@click.option('-o', '--output', type=click.Path(writable=True), required=True)
def predict(model_directory, images_directory, pattern, output):
    """
    Predict whether some class label applies to images in a directory.
    """
    # Load the filenames and the actual images
    filenames = Dataset.list_files(str(images_directory/pattern), shuffle=False)

    for f in filenames.take(5):
        print(f.numpy())

    images_ds = filenames.map(process_path, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
    
    # Load model
    loaded_model = keras.models.load_model(model_directory)
    # Make predictions
    predictions = loaded_model.predict(images_ds, verbose=1)
    # Convert predictions to DataFrame with filenames
    predictions_ds = Dataset.from_tensor_slices(predictions)
    predictions_ds = predictions_ds.flat_map(lambda x: Dataset.from_tensor_slices(x))
    pred_labels = Dataset.zip((filenames, predictions_ds))
    results = pd.DataFrame(pred_labels.as_numpy_iterator(), columns=["filename", "prediction"])
    results.loc[:, "filename"] = results.loc[:, "filename"].str.decode('utf-8').str.removeprefix(str(images_directory) + "/")
    results.to_csv(output, index=False)


if __name__ == '__main__':
    predict()
