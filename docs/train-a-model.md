---
title: Training a model
---

# Overview

We have worked on various Keras-based models to predict the visibility of
symbols in frames from the TV series.
If there is only one symbol of interest, we could put the images with the
symbol in one directory and the images without the symbol in another.
The script `build_binary_classifier.py` works that way.

When we want a model to predict the appearance of multiple different symbols,
potentially in the same frame, using directories to separate the classes
becomes intractible.
In that case we have to create a CSV file that contains the filenames of the
images and for each symbol a column that indicates the visibility of the symbol
for each image.

# Train a binary classifier from image directories

## Image directory setup

## Training the binary classifier

# Train a multi-label model from images and metadata

## Create training data

To train a model, we need images with labels, and a name for the model.
We expect images to be in subdirectories of a given directory.
The filenames and corresponding labels should be in a CSV file.
Filenames must be in a column `filename` and must be relative to the given
directory.
Names of columns with (numeric 0 or 1) labels must end in `_visible`.

Minimal example of a labels file:


| filename      | pentagram_visible | star_visible |
|---------------|-------------------|--------------|
| image001.jpg  | 1                 | 0            |
| image002.jpg  | 1                 | 1            |
| image003.jpg  | 0                 | 1            |
| image004.jpg  | 0                 | 0            |

A model trained on these images and labels will have two outputs: `pentagram`
and `star`.

## Run the training

To train a model, run:

```console
$ python build_multi_hot_model.py image_dir labels.csv output_model_dir
```

In this command:

- `image_dir` is the directory that contains subdirectories with the images;
- `labels.csv` is the CSV file holding the filename and label(s) for each image;
- `output_model_dir` is the name of the model and the directory that it will
  be saved to.
