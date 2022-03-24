---
title: Video analysis
---

To use computer vision for video analysis, we have to train a model with
annotated video frames.
Next to the scripts in this repository, we use the scripts for creating and
working with annotations from [MIDA-scene-detection].
In the future we may merge these repositories.

[MIDA-scene-detection]: https://github.com/LeidenUniversityLibrary/MIDA-scene-detection

This guide covers the following steps of the process:

1. [Training a model](train-a-model.md) using labeled (i.e. annotated) frames;
1. [Evaluating a model](evaluate-model.md) on annotated frames;
1. [Making predictions](make-predictions.md) for not yet annotated frames;
1. [Improving the annotations](improve-annotations.md) using ELAN, starting
   from predictions.

Additionally, there are [DIANNA installation instructions](install-dianna.md).
[DIANNA] is a tool that tries to explain how a model classifies data.

[DIANNA]: https://github.com/dianna-ai/dianna
