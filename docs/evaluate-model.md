---
title: Evaluating a model
---

Use already labeled frames to see how many predictions are correct.

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
