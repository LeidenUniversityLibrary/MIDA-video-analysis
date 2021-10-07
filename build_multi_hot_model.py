# coding: utf-8


import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
import sys


image_directory = sys.argv[1]
csv_filename    = sys.argv[2]
model_directory = sys.argv[3]

print(f'Starting. Looking for images in {image_directory}, labels in {csv_filename}.')
print(f'Will save the model in {model_directory}.')

metadata = pd.read_csv(csv_filename)

train_datagen = keras.preprocessing.image.ImageDataGenerator(
    # rescale=1./255,
    rotation_range=36,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2)

train_generator = train_datagen.flow_from_dataframe(
    dataframe=metadata,
    directory=image_directory,
    x_col='',
    y_col=[''],
    weight_col='',
    target_size=(150, 150),
    color_mode='rgb',
    class_mode='raw',
    shuffle=True,
    seed=42,
    subset='training',
    interpolation='bilinear'
)

validation_datagen = keras.preprocessing.image.ImageDataGenerator(
    # rescale=1./255,
    validation_split=0.2)

validation_generator = validation_datagen.flow_from_dataframe(
    dataframe=metadata,
    directory=image_directory,
    x_col='',
    y_col=[''],
    weight_col='',
    target_size=(150, 150),
    color_mode='rgb',
    class_mode='raw',
    shuffle=True,
    seed=42,
    subset='validation',
    interpolation='bilinear'
)


print("Number of training samples: %d" % tf.data.experimental.cardinality(train_ds))
print("Number of validation samples: %d" % tf.data.experimental.cardinality(validation_ds))




# Build a model

base_model = keras.applications.Xception(
    weights="imagenet",  # Load weights pre-trained on ImageNet.
    input_shape=(150, 150, 3),
    include_top=False,
)

# Freeze the base_model
base_model.trainable = False

# Create new model on top
inputs = keras.Input(shape=(150, 150, 3))

# Pre-trained Xception weights requires that input be normalized
# from (0, 255) to a range (-1., +1.), the normalization layer
# does the following, outputs = (inputs - mean) / sqrt(var)
norm_layer = keras.layers.experimental.preprocessing.Normalization()
mean = np.array([127.5] * 3)
var = mean ** 2
# Scale inputs to [-1, +1]
x = norm_layer(inputs)
norm_layer.set_weights([mean, var])

# The base model contains batchnorm layers. We want to keep them in inference mode
# when we unfreeze the base model for fine-tuning, so we make sure that the
# base_model is running in inference mode here.
x = base_model(x, training=False)
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dropout(0.2)(x)  # Regularize with dropout
outputs = keras.layers.Dense(3, activation='sigmoid')(x)
model = keras.Model(inputs, outputs)

model.summary()


# Train the top layer

model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.BinaryCrossentropy(from_logits=False),
    metrics=[[keras.metrics.BinaryAccuracy()],
        [keras.metrics.BinaryAccuracy()],
        [keras.metrics.BinaryAccuracy()]]
)

epochs = 20
model.fit(train_generator, epochs=epochs, validation_data=validation_generator)


## Save the model

model.save(model_directory)

print("Saved model to disk")
