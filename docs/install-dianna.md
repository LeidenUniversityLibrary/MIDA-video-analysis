---
title: Installing DIANNA
---

Installing [DIANNA], especially on an M1 Apple computer, requires some thought.

[DIANNA]: https://github.com/dianna-ai/dianna

## Installing on an Apple M1 computer

Because not all packages in [PyPI] have been built for the M1's ARM
architecture yet, we start with the [miniforge] distribution of Conda.
From here, these instructions may get outdated, as many packages are becoming
available for ARM in PyPI.
Some DIANNA dependencies, however, still need to be installed from
[conda-forge].

[PyPI]: https://pypi.org/
[miniforge]: https://github.com/conda-forge/miniforge
[conda-forge]: https://conda-forge.org/

These instructions have worked with MacOS 12 (Monterey), but may not work with
MacOS 11, because Apple's M1-optimised version of Tensorflow is only supported
on MacOS 12.

Create and activate a conda environment for DIANNA:

```console
$ conda create -n dianna
$ conda activate dianna
```

Install Python 3.9 (per the [DIANNA installation instructions][dii]):

```console
$ conda install python=3.9
```

[dii]: https://github.com/dianna-ai/dianna#installation

Install Tensorflow (per the [Apple instructions][ati]):

```console
$ conda install -c apple tensorflow-deps
$ pip install tensorflow-macos
$ pip install tensorflow-metal
```

[ati]: https://developer.apple.com/metal/tensorflow-plugin/

Install other dependencies:

```console
$ conda install onnxruntime onnx numba llvmlite
$ pip install jupyter
```

Comment out the `tensorflow` dependency from DIANNA's `setup.cfg` and then
install DIANNA itself from the cloned git repository:

```console
$ cd path/to/dianna
$ pip install -e .
```
