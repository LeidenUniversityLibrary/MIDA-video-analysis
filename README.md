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

Please see the [documentation](docs/index.md) in this repository, or the
[rendered online documentation][online_docs] for usage instructions.

[online_docs]: https://leidenuniversitylibrary.github.io/MIDA-video-analysis/

## Authors and license

These scripts have been created by Peter Verhaar and Ben Companjen at
Leiden University Libraries' Centre for Digital Scholarship, in collaboration
with Mustafa Çolak and the Netherlands eScience Center.
Some scripts are heavily based on existing tutorials, for which we do not claim
authorship.

The license is to be determined.
