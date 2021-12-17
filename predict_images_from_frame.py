"""
Create predictions on files listed in a dataframe.

Usage::

    $ python predict_images_from_frame.py model_dir image_dir file_list.csv results.csv

"""
import tensorflow as tf
import numpy as np
from tensorflow import keras

import os
import sys
import pandas as pd
import pathlib


# Load parameters
model_directory = sys.argv[1]
image_directory = pathlib.Path(sys.argv[2])
metadata_path = sys.argv[3]
results_path = sys.argv[4]

print(f'Starting. Looking for images in {image_directory}, labels in {metadata_path}.')
print(f'Loading the model from {model_directory}.')
loaded_model = keras.models.load_model(model_directory)
metadata = pd.read_csv(metadata_path)

# Sample the metadata file to speed up the evaluation during development
# metadata = metadata.sample(100, ignore_index=True)

# Find the names of columns ending in '_visible'
label_columns = [col_name for col_name in metadata.columns if col_name.endswith('_visible')]
print(f'Found {label_columns} as columns with labels.')

validation_datagen = keras.preprocessing.image.ImageDataGenerator(
    # rescale=1./255,
)

validation_generator = validation_datagen.flow_from_dataframe(
    dataframe=metadata,
    directory=image_directory,
    x_col='filename',
    y_col=label_columns,
    weight_col=None,
    target_size=(150, 150),
    color_mode='rgb',
    class_mode='multi_output',
    shuffle=False,
    interpolation='bilinear'
)
# print(validation_generator.filenames, type(validation_generator.filenames), len(validation_generator.filenames))
# sys.exit(0)
# Make the predictions
predictions = loaded_model.predict(validation_generator, verbose=1)

# Convert predictions to a DataFrame
pred_dict = {}
for col_name, preds in zip(label_columns, predictions):
    pred_dict[col_name.replace('_visible', '_pred')] = preds.ravel()
pred_dict['filename'] = validation_generator.filenames
predictions_df = pd.DataFrame(pred_dict)
print(predictions_df.head())

# Add columns with rounded prediction values
rounded_pred_columns = [col + "_round" for col in predictions_df.columns]
rounded_predictions = predictions_df.apply(round, axis=1)
rounded_predictions.columns = rounded_pred_columns

# Join the predictions with the original data
results = pd.merge(metadata, predictions_df, on='filename')
results = results.join(rounded_predictions)
results.to_csv(results_path, index=False)
print(f'Saved the predictions to {results_path}.')

print('Counting predictions for each combination of predicted label value.')
totals = results.groupby(rounded_pred_columns)['filename']
totals_count = totals.size()
counts_path = results_path.replace('.csv', '_counts.csv')
totals_count.to_csv(counts_path)
print(totals_count)
print(f'Saved these counts to {counts_path}.')
