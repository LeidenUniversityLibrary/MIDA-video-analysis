---
title: Train a YOLOv5 model
---

# Collect training data

Training data consist of images and rectangular annotations of objects.
The object labels are in a text file that is linked to the image through the
file name.
Multiple objects in a single image go on separate lines.

It is advisable to use a tool like <https://makesense.ai/>.

!!! warning

    If you create various datasets, make sure to use the same labels in each of
    them.
    YOLO expects numbers as labels and training will not give expected results
    if in one dataset the symbol A maps to 2 in another dataset it maps to 0.

Various tips are available in the [YOLOv5 custom data guide][customdata].

[customdata]: https://docs.ultralytics.com/tutorials/train-custom-datasets/

# Prepare training configuration

The [YOLOv5 custom data guide][customdata] has the most important information.

If you want to adjust training parameters like random rotation, you have to
provide your own hyperparameters in a YAML file.

# Run the training

Again, see the [YOLOv5 custom data guide][customdata] for the current
instructions.

See our [instructions on running the training on the ALICE HPC](run-on-alice.md)
to understand how we can speed up the process.
