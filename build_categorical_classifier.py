#!/usr/bin/env python
# coding: utf-8


import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers


image_directory = 'Image'

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    image_directory, labels='inferred', label_mode='categorical',
    class_names=None, color_mode='rgb', image_size=(150,
    150), shuffle=True, seed= 42 , validation_split= 0.2, subset= 'training',
    interpolation='bilinear', follow_links=False #, smart_resize=False
)

validation_ds = tf.keras.preprocessing.image_dataset_from_directory(
    image_directory, labels='inferred', label_mode='categorical',
    class_names=None, color_mode='rgb', image_size=(150,
    150), shuffle=True, seed= 42 , validation_split= 0.2, subset= 'validation',
    interpolation='bilinear', follow_links=False #, smart_resize=False
)

class_names = train_ds.class_names
nr_classes = len(class_names)

print("Number of training samples: %d" % tf.data.experimental.cardinality(train_ds))



data_augmentation = tf.keras.Sequential([
  layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
  layers.experimental.preprocessing.RandomRotation(0.2),
])



base_model = keras.applications.Xception(
    weights="imagenet",
    input_shape=(150, 150, 3),
    include_top=False,
)

base_model.trainable = False




inputs = keras.Input(shape=(150, 150, 3))
# Apply random data augmentation
x = data_augmentation(inputs)

norm_layer = keras.layers.experimental.preprocessing.Normalization()
mean = np.array([127.5] * 3)
var = mean ** 2
# Scale inputs to [-1, +1]
x = norm_layer(x)
norm_layer.set_weights([mean, var])
x = norm_layer(x)

x = base_model(x, training=False)
x = keras.layers.GlobalAveragePooling2D()(x)
# Regularize with dropout
x = keras.layers.Dropout(0.2)(x)
outputs = keras.layers.Dense(nr_classes, activation = 'sigmoid')(x)
model = keras.Model(inputs, outputs)

model.summary()




model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.CategoricalCrossentropy(
    from_logits=False,
    label_smoothing=0,
    reduction="auto",
    name="categorical_crossentropy"),
    metrics=[keras.metrics.CategoricalAccuracy(name="categorical_accuracy", dtype=None)],
)

epochs = 200
model.fit(train_ds, epochs=epochs, validation_data=validation_ds)



## Save the model
print("Saving model")
model.save('symbolism')
print("Saved model to disk")
