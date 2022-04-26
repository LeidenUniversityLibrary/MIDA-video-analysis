# coding: utf-8
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import pathlib
import click

@click.command()
@click.option('-i', '--image-dir', type=click.Path(exists=True, dir_okay=True), 
    help='Directory with images for training and evaluation')
@click.option('--model-name', help='Name of the model (must not include spaces)')
@click.option('--model-basedir', type=click.Path(path_type=pathlib.Path), help='Base directory to store all models')
@click.option('--epochs', type=int, default=50, show_default=True)
@click.option('--seed', type=int, default=42, show_default=True)
@click.option('--split', default=0.2, show_default=True)
def train(image_dir, model_name, model_basedir, epochs, seed, split):

    model_directory = model_basedir / model_name

    # The `tf.keras.preprocessing.image_dataset_from_directory` can be used to
    # generate a labeled dataset objects from a set of images on disk filed into
    # class-specific folders.
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        image_dir, labels='inferred', label_mode='binary',
        class_names=None, color_mode='rgb', image_size=(150,
        150), shuffle=True, seed=seed, validation_split=split, subset= 'training',
        interpolation='bilinear', follow_links=False #, smart_resize=False
    )

    class_names = train_ds.class_names
    print("Classes:", class_names)

    validation_ds = tf.keras.preprocessing.image_dataset_from_directory(
        image_dir, labels='inferred', label_mode='binary',
        class_names=None, color_mode='rgb', image_size=(150,
        150), shuffle=True, seed=seed, validation_split=split, subset= 'validation',
        interpolation='bilinear', follow_links=False #, smart_resize=False
    )

    print("Number of training batches:", tf.data.experimental.cardinality(train_ds).numpy())
    print("Number of validation batches:", tf.data.experimental.cardinality(validation_ds).numpy())

    # Random data augmentation
    # data_augmentation = keras.Sequential(
    #     [
    #         layers.experimental.preprocessing.RandomFlip("horizontal"),
    #         layers.experimental.preprocessing.RandomRotation(0.1),
    #     ]
    # )

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
    scale_layer = keras.layers.Rescaling(scale=1 / 127.5, offset=-1)
    x = scale_layer(inputs)

    # The base model contains batchnorm layers. We want to keep them in inference mode
    # when we unfreeze the base model for fine-tuning, so we make sure that the
    # base_model is running in inference mode here.
    x = base_model(x, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.2)(x)  # Regularize with dropout
    outputs = keras.layers.Dense(1, activation='sigmoid')(x)
    model = keras.Model(inputs, outputs, name=model_name)

    model.summary()

    # Train the top layer
    print("Starting compilation")
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.BinaryCrossentropy(from_logits=False),
        metrics=[keras.metrics.BinaryAccuracy(),
            tf.keras.metrics.FalsePositives(),
            tf.keras.metrics.FalseNegatives()],
    )

    print("Starting fit")
    model.fit(train_ds, epochs=epochs, validation_data=validation_ds)

    ## Save the model
    print("Saving model")
    model.save(model_directory)
    print("Saved model to disk")


if __name__ == "__main__":
    train()
