# Generic Object ZSL Dataset (GOZ)

This repository references the dataset described in the paper "On Zero-Shot Learning of generic objects" to be presented at CVPR 2019. [arxiv link]
In addition to the instructions to download the dataset, it also provides code to reproduce the different experiments presented in the paper, 
as well as code to manipulate the Wordnet hierarchy and Imagenet metadata in order to experiment with different training/test class splits 
of the Imagenet dataset subject to different constraints.

# About this repository

This repository is organized into four folders.
Each folder has its own readme and its own python package requirements (requirements.txt)

To start experimenting with this dataset, first clone the repo

```
git clone https://github.com/TristHas/GOZ.git
```

and refer to the folder you are interested in, as described below:

## Download

People interested in evaluating their ZSL model on the GOZ dataset should refer to the Download folder.
You can either download Resnet extracted feature representations of the dataset or the original images.

In addition to the download of the ZSL dataset, we also provide scripts for downloading arbitrary splits of the Imagenet dataset, 
as well as scripts to extract visual features from the downloaded raw images.

See the readme of the download folder for more precise instructions.

## Benchmark

This folder contains the notebooks used to evaluate the few baseline models on the GOZ dataset as presented in Table of the paper.
You can use these notebooks as a starting point to develop new ideas/models.

## Experiments

This folder is intended for people interested in either 
 - the dataset generation process
 - experiments presented in the paper
 - experimenting with different training/test splits than the one proposed in the paper 

WARNING: Some of the experiments require you to download the full Imagenet dataset and to extract Resnet features from the full dataset.
Downloading the 13M images of the Imagenet dataset can be time-consuming, even with parallel downloads, and need large storage capacity. 

For researchers interested in experimenting on different test splits with limited resurces, we propose a trick we have been using:
Instead of working on individual images, we have been working on the mean visual feature activations.
This drastically speeds up processing time at the cost of ignoring the intra-class variance of the visual feature distribution.
Please refer to the Experiments folder's readme for more detailed explanations

## Data

This folder serializes the data needed for either downloading, benchmarking or experimentuing with new training/test splits.

# Citation

If you use this dataset for your research, please cite the following paper:
