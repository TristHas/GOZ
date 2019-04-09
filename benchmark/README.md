# Benchmark

This folder contains notebooks to repoduce the evaluation of different models on the GOZ dataset.

## Prerequisite

To run the notebooks, you will need to download the Resnet visual features following the instructions of the download folder.
In addition, you will need to install the packages listed in the requirements.txt file of this folder.

## Linear models

The ConSE, DeViSe and ESZSL notebooks contain the code for training and evaluating the respective models.

## GCN models

The GCN notebook contains the code for testing different GCN-based models (GCN-6, GCN-2 and ADGPM)
This notebook uses classifier weights learned using the excellent repo: https://github.com/cyvius96/DGP.git and stored in the folder data/models/.
The classifier weights used in our benchmark were learned without the fine-tuning step of https://arxiv.org/abs/1805.11724.

## Contribution

Anyone is welcome to contribute their model.
Feel free to contact me to include your model in this repository.
