# Predict binary labels for images in folders
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

loaded_model = keras.models.load_model(model_directory)

filenames = Dataset.list_files(str(images_path/'*/*/*'), seed=42)#.take(100)

for f in filenames.take(5):
    print(f.numpy())


class_names = np.array(sorted([item.name for item in images_path.glob('*')]))
print(class_names)

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
    label = get_label(file_path)
    # load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = tf.expand_dims( decode_img(img) , axis=0)
    return img, label


images_ds = filenames.map(process_path, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
for f in images_ds.take(5):
    print(f)


# label_batches = list(labels for images,labels in data_subset)
# all_labels = tf.concat(label_batches, 0)
# for images, labels in data_subset:
    # all_labels = np.concatenate([all_labels,labels.numpy()])
    # print(labels.numpy())
# print(all_labels.shape)

predictions = loaded_model.predict(images_ds, verbose=1)
# print(predictions, len(predictions))
# sig_predictions = tf.keras.activations.sigmoid(predictions)
# print(sig_predictions.shape)
# pred_labels = tf.stack([sig_predictions, all_labels], axis=1)
# pred_labels = tf.concat([sig_predictions, all_labels], axis=1)
# pred_labels = tf.concat([predictions, all_labels], axis=1)
predictions_ds = Dataset.from_tensor_slices(predictions)
predictions_ds = predictions_ds.flat_map(lambda x: Dataset.from_tensor_slices(x))
pred_labels = Dataset.zip((predictions_ds, filenames))
results = pd.DataFrame(pred_labels.as_numpy_iterator(), columns=["prediction", "filename"])
results.loc[:, "filename"] = results.loc[:, "filename"].astype("string").str.replace(str(images_path), "")
results.to_csv(results_path, index=False)
print(pred_labels)
# print(loaded_model.evaluate(data_subset, verbose=1))
