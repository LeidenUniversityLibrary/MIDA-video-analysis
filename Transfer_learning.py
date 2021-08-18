# coding: utf-8


import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
# from keras.models import model_from_json
import sys
# The `tf.keras.preprocessing.image_dataset_from_directory` can be used to generate a labeled dataset objects from a set of images on disk filed into class-specific folders.

image_directory = sys.argv[1]
model_directory = sys.argv[2]

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    image_directory, labels='inferred', label_mode='int',
    class_names=None, color_mode='rgb', image_size=(256,
    256), shuffle=True, seed= 42 , validation_split= 0.2, subset= 'training',
    interpolation='bilinear', follow_links=False #, smart_resize=False
)

class_names = train_ds.class_names
print(class_names)



validation_ds = tf.keras.preprocessing.image_dataset_from_directory(
    image_directory, labels='inferred', label_mode='int',
    class_names=None, color_mode='rgb', image_size=(256,
    256), shuffle=True, seed= 42 , validation_split= 0.2, subset= 'validation',
    interpolation='bilinear', follow_links=False #, smart_resize=False
)

print("Number of training batches: %d" % tf.data.experimental.cardinality(train_ds).numpy())
print("Number of validation batches: %d" % tf.data.experimental.cardinality(validation_ds).numpy())

print("Resizing datasets")

size = (150, 150)

train_ds = train_ds.map(lambda x, y: (tf.image.resize(x, size), y))
validation_ds = validation_ds.map(lambda x, y: (tf.image.resize(x, size), y))

# Random data augmentation
data_augmentation = keras.Sequential(
    [
        layers.experimental.preprocessing.RandomFlip("horizontal"),
        layers.experimental.preprocessing.RandomRotation(0.1),
    ]
)


# Build a model
print("Loading base model")

base_model = keras.applications.VGG16(
    weights="imagenet",  # Load weights pre-trained on ImageNet.
    input_shape=(150, 150, 3),
    include_top=False,
)

# Freeze the base_model
base_model.trainable = False

# Create new model on top
inputs = keras.Input(shape=(150, 150, 3))
# x = data_augmentation(inputs)  # Apply random data augmentation
x = inputs  # Do not apply random data augmentation

# Pre-trained Xception weights requires that input be normalized
# from (0, 255) to a range (-1., +1.), the normalization layer
# does the following, outputs = (inputs - mean) / sqrt(var)
norm_layer = keras.layers.experimental.preprocessing.Normalization()
mean = np.array([127.5] * 3)
var = mean ** 2
# Scale inputs to [-1, +1]
x = norm_layer(x)
norm_layer.set_weights([mean, var])

# The base model contains batchnorm layers. We want to keep them in inference mode
# when we unfreeze the base model for fine-tuning, so we make sure that the
# base_model is running in inference mode here.
x = base_model(x, training=False)
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dropout(0.2)(x)  # Regularize with dropout
outputs = keras.layers.Dense(1)(x)
model = keras.Model(inputs, outputs)

model.summary()


# Train the top layer

print("Starting compilation")

model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=[keras.metrics.BinaryAccuracy()],
)

print("Starting fit")
epochs = 5
model.fit(train_ds, epochs=epochs, validation_data=validation_ds)


## Save the model
print("Saving model")
model.save("./{}/".format(model_directory))
print("Saved model to disk")
