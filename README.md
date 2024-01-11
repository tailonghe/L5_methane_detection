# Landsat 5 methane super emitter detection using deep learning
This repository contains scripts used in our paper entitled "[Space-borne assessment of the Soviet Union collapse on the methane slowdown](https://doi.org/10.31223/X5G67G)".

# Description
* Folder `backend` contains scripts used by the detection of methane plumes and the information about Landsats/Sentinel-2 ImageCollections from Google Earth Engine.
* Folder `figure_reproduce` contains auxiliary data files generaed by `figure3.ipynb` that are used to reproduce figures in the paper.
* `landsat5_environment.yml` is the list of Python packages for setting up the conda environment. The Python environment could be set up using conda as `conda env create -f landsat5_environment.yml`.
* `PlumeNet.py` is the module to build U-net deep learning models and to define functions for training.
* `ensemble_train.ipynb` shows the construction and training process of the ensemble system.
* `demonstration.ipynb` shows some examples of methane plume detections and quantifications directly from Google Earth Engine.
* Notebooks `figure1.ipynb`, `figure2.ipynb`, and `figure3.ipynb` generate figures in the paper, using functions from `plot_functions.py`.
* Results and datasets required for the reproduction of the results have been deposited in the Dryad Data Repository (https://doi.org/10.5061/dryad.4mw6m90hp).
* `figure_reproduce/combined_data_set.csv` is the dataset integrating gas production, bottom-up emissions, number of detections, and annual CH4 emissions from the detected point sources.

# Examples of detected methane plumes
