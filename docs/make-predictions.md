---
title: Make predictions
---

When you have a model that is good enough, you can use it to predict whether
symbols are visible on other unlabeled frames.

# Predict a single label with a binary classification model

Use `predict_binary_images.py`.

# Make predictions from a directory of (directories of) images

To make predictions using a model, run:

```console
$ python predict_images.py model_dir image_dir predictions.csv
```

In this command:

- `model_dir` is the directory that contains the model;
- `image_dir` is the directory that contains subdirectories with the images;
- `predictions.csv` is the filename in which predictions will be saved.

# Make predictions for files in a CSV

## Create frame list

The other script that runs the model is based on the script for evaluation.
It reads the filenames of the frames, as well as the expected labels from a CSV
file.
We therefore need to create the CSV file first.

## Make predictions from the frame list

Use `predict_images_from_frame.py`.

## Look at the statistics

