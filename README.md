# MIDA-video-analysis
Computer vision in television

## Install on Apple M1

Because the M1 chipset has an architecture based on ARM instead of x86_64,
we cannot use Python and Tensorflow from the default channels.

In a Terminal, run `conda env create -f environment_m1.yml -n mida_video`
to create a conda environment with basic dependencies for running the scripts
in this repository.
Afterwards, run `conda activate mida_video` to activate the environment.

If you want more manual control, install Tensorflow following instructions at
<https://developer.apple.com/metal/tensorflow-plugin/>.
Then install the other dependencies listed in `environment_m1.yml`.

## Usage

The provided scripts help us train, evaluate and use computer-vision models
based on Tensorflow and Keras.

### Train a model

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

To train a model, run:

```console
$ python build_multi_hot_model.py image_dir labels.csv output_model_dir
```

In this command:

- `image_dir` is the directory that contains subdirectories with the images;
- `labels.csv` is the CSV file holding the filename and label(s) for each image;
- `output_model_dir` is the name of the model and the directory that it will
  be saved to.

### Evaluate a model

To evaluate a model, run:

```console
$ python eval_multi_hot_model.py model_dir image_dir labels.csv results.csv
```

In this command:

- `model_dir` is the directory that contains the model;
- `image_dir` is the directory that contains subdirectories with the images;
- `labels.csv` is the CSV file holding the filename and label(s) for each
  image, which should be structured the same as the file used for training;
- `results.csv` is a filename that the summary of predictions will be in.

### Use a model for making predictions

To make predictions using a model, run:

```console
$ python predict_images.py model_dir image_dir predictions.csv
```

In this command:

- `model_dir` is the directory that contains the model;
- `image_dir` is the directory that contains subdirectories with the images;
- `predictions.csv` is the filename in which predictions will be saved.
