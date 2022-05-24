---
title: Detect symbols with YOLO model
---

# Prepare data to run on



# Run YOLO model

Command goes here.

While it is possible to run YOLOv5 locally, it is highly advisable to use a
machine with a GPU.
See our [instructions on running detection on the ALICE HPC](run-on-alice.md)
to understand how we used ALICE.

# Filter false positives using a binary classifier

Our YOLO model may predict a lot of false positives, even after increasing the
number of training examples and epochs.
We can try to remove false positives using a Keras-based binary classifier
that works on the cropped images that YOLO thinks are symbols.

## Collect training data

In our case only one symbol had many false positives.
That means we only need to collect a sample of true and false positives for
that symbol and we can train a binary classifier on this sample.

Copy a set of correctly identified crops into one directory and a set of
incorrectly identified crops into another.
It helps if the directory names are such that the false positives come first,
so that they internally get mapped to `0`, while true positives get `1`.
When we then predict the correctness, the rounded confidence matches false and
true positive, respectively.

## Train binary classifier

See [Train a model](train-a-model.md) for instructions.

## Predict whether cropped images are the expected symbol

Run the new classifier on each crop directory that YOLO made for the symbol.
You could use `predict_binary_images.py` for this.

```sh
cd yolo_model_5-recognitions
for I in {1..54}; do
mkdir -p ${I}/crops
unzip -u ${I}/ep${I}_211984_recognitions.zip '*.jpg' -d ${I}/crops
python ~/git/MIDA-video-analysis/predict_binary_images.py --model-directory ~/surfdrive/Projecten/MIDA/Mustafa/models/star_verifier --images-directory ${I}/crops --pattern '*/*/star_of_david/*.jpg' --output ${I}/star_results.csv
done
```

## Remove incorrectly identified symbols

We need to remove the lines from the predictions text files that correspond to
images that our classifier thinks are not the symbol we are looking for.

The `yolo_zip_summary.py` script provides the `-a` (or `--aux-results`) option that expects a tuple
of class label (as text) and corresponding results file.
The threshold for including images based on the auxiliary classification is set
to 0.3.
This option can be repeated, so you could use auxiliary classification results
for multiple symbols.

```sh
cd yolo_model_5-recognitions
for I in {1..54}; do
python ~/git/MIDA-video-analysis/yolo_zip_summary.py -a star_of_david ${I}/star_results.csv --min-confidence 0.4 --output-csv ${I}/ep${I}-0_4-counts.csv --delete-labels pentagram,keys_of_heaven ${I}/ep${I}_211984_recognitions.zip
done
```

# Summarise results

## Run yolo summarise

Run `yolo_zip_summary.py` for zipped results (such as the results from ALICE).

```sh
cd yolo_model_5-recognitions
for I in {1..54}; do
python ~/git/MIDA-video-analysis/yolo_zip_summary.py --min-confidence 0.4 --output-csv ${I}/ep${I}-0_4-counts.csv --delete-labels pentagram,keys_of_heaven ${I}/ep${I}_211984_recognitions.zip
done
```

## Combine with detected scenes

Run `match_symbols_to_scenes.py`.

```sh
cd yolo_model_5-recognitions/..
for I in {1..54}; do
python ~/git/MIDA-video-analysis/match_symbols_to_scenes.py \
 --symbols yolo_model_5-recognitions/${I}/ep${I}-0_4-counts.csv \
 --scenes scene-detection-results/${I}/scenes.csv \
 --output yolo_model_5-recognitions/${I}/ep${I}-0_4-with-scenes.csv
done
```

## Concatenate files per episode into one file

We used [csvstack].

```bash
cd yolo_model_5-recognitions
csvstack --filenames {1..54}/*0_4-with-scenes.csv > all_recognitions-0_4-with-scenes.csv
```

[csvstack]: https://csvkit.readthedocs.io/en/latest/scripts/csvstack.html

## Count symbols per scene

Not implemented yet, but you could load the combined results into a spreadsheet
application and create a pivot table that shows the number of symbols by
episode and scene.
