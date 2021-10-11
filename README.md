# MIDA-video-analysis
Computer vision in television

## Run on Apple M1

In a Terminal, run `conda env create -f environment_m1.yml -n mida_video`
to create a conda environment with basic dependencies for running the scripts
in this repository.
Afterwards, run `conda activate mida_video` to activate the environment.

If you want more manual control, install Tensorflow following instructions at
<https://developer.apple.com/metal/tensorflow-plugin/>.
Then install the other dependencies listed in `environment_m1.yml`.
