# coding: utf-8


import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import model_from_json

# The `tf.keras.preprocessing.image_dataset_from_directory` can be used to generate a labeled dataset objects from a set of images on disk filed into class-specific folders.



train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    'Image', labels='inferred', label_mode='int',
    class_names=None, color_mode='rgb', image_size=(256,
    256), shuffle=True, seed= 42 , validation_split= 0.1, subset= 'training',
    interpolation='bilinear', follow_links=False, smart_resize=False
)




validation_ds = tf.keras.preprocessing.image_dataset_from_directory(
    'Image', labels='inferred', label_mode='int',
    class_names=None, color_mode='rgb', image_size=(256,
    256), shuffle=True, seed= 42 , validation_split= 0.1, subset= 'validation',
    interpolation='bilinear', follow_links=False, smart_resize=False
)




print("Number of training samples: %d" % tf.data.experimental.cardinality(train_ds))
print("Number of validation samples: %d" % tf.data.experimental.cardinality(validation_ds))





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




import numpy as np


# Build a model



base_model = keras.applications.VGG16(
    weights="imagenet",  # Load weights pre-trained on ImageNet.
    input_shape=(150, 150, 3),
    include_top=False,
)

# Freeze the base_model
base_model.trainable = False

# Create new model on top
inputs = keras.Input(shape=(150, 150, 3))
x = data_augmentation(inputs)  # Apply random data augmentation

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
outputs = keras.layers.Dense(3, activation='softmax')(x)
model = keras.Model(inputs, outputs)

model.summary()


# Train the top layer



model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=[keras.metrics.BinaryAccuracy()],
)

epochs = 20
model.fit(train_ds, epochs=epochs, validation_data=validation_ds)

#predictions = model.predict(X_test)
#matrix = metrics.confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))


## Save the model

model.save("symbolism")

print("Saved model to disk")
