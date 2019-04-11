# Generic Object ZSL Dataset (GOZ)

This repository references the dataset described in the paper "On Zero-Shot Learning of generic objects" to be presented at CVPR 2019. (http://arxiv.org/abs/1904.04957)

In addition to the instructions to download the dataset, it also provides code to:
 - benchmark the few baseline models evaluated in the final section of the paper
 - manipulate the Wordnet hierarchy and Imagenet metadata in order to generate training/test class splits subject to different constraints from the Imagenet  dataset.
 - reproduce the different experiments presented in the paper

# About this repository

This repository is organized into four folders.
Each folder has its own readme and its own python package requirements (requirements.txt)

To get started, first clone the repo

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

WARNING: Some of the experiments in this folder require you to download the full Imagenet dataset and to extract Resnet features from the full dataset.
Downloading the 13M images of the Imagenet dataset can be time-consuming, even with parallel downloads, and need large storage capacity. 


## Data

This folder provides the data needed for either downloading, benchmarking or experimenting with new training/test splits.

# Citation

If you use this dataset for your research, please cite the following paper:

```
@article{hascoet2019goz,
  title={On Zero-Shot Learning of generic objects},
  author={Hascoet, Tristan and Ariki, Yasuo and Takiguchi, Tetsuya},
  journal={arXiv preprint arXiv:1904.04957},
  year={2019}
}
```