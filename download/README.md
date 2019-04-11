# Download Features

The Resnet feature representations are currently hosted on a personal Google Drive.
We will try to move these files to a more permanent storage solutions so this is likely to change.
As of today, you can download these features from the following url:

- training features: https://drive.google.com/open?id=1_XFAswLqvcH2FkO_aa3CDpiRTcUXM1gm
- test features: https://drive.google.com/open?id=1Ru3nd9ZsHF_TzPorLackI83dWfe98u4s

To evaluate the baseline linear and gcn models using the jupyter notebooks in the benchmark folder,
make sure you download these files to the data/visuals/features/ folder or edit the corresponding pathes in the  data/visuals/features/helpers.py file.

The datasets are h5py.File objects contain two subgroups images and idx:
- The images group contains one HDF5 dataset per class with size (N samples, 2048) and type float32 containing the extracted visual features.
- The images group contains one HDF5 dataset per class with size (N samples,) and type int8 containing the id of the original imageof each extracted features. 

For example:

```
- images
-- <HDF5 dataset 'n00141669': shape (100, 2048), type "<f4">
...
-- <HDF5 dataset 'n00440747': shape (100, 2048), type "<f4">

- idx
-- <HDF5 dataset "n00141669": shape (100,), type "<i8">
...
-- <HDF5 dataset "n00440747": shape (100,), type "<i8">
```

# Download Images

## GOZ training images

The training split consists of the original ILSVRC 2012 classification challenge dataset.
You can directly download it from http://image-net.org/download-images
In addition to the training images, the validatition set of the ILSVRC dataset are used as test samples of the training classes in the Generalized ZSL setting.

## GOZ test images

To download the test images, please refer to the Download Images notebook.

## Full Imagenet images

In order to run the code of the experiments folder, you will need to download the full Imagenet dataset.
To download the full images of the Imagenet dataset, you will need to first download the list of image urls.
To do so, execute the  imagenet_urls.sh script from this repository:

cd /path/to/GOZ/download
bash ./imagenet_urls.sh

You can then download the full Imagenet dataset using the Download Images notebook by changing the url_path variable to ../data/downloads/full_imagenet_urls.txt

# Extract visual features

The notebook "Extract visual.ipynb" shows code snippets to extract resnet features from the downloaded raw images.

Features are saved as h5py datasets in the location (data/visuals/features/) used by the benchmark notebooks.

The structure of the resulting h5py.File datasets is the same as the downloadable features.

# Caution

Many of the Imagenet provided URLS seem to be broken.
This is also true of the GOZ testsplit images. 
You will not be able to download the full (100 images per test class) test split as many download URLs seem to have gone down since the time we downloaded these.
We will provide a link to download the images as soon as possible (the time to figure out a long term storage solution), and the full test split original images will be made available by the time of the CVPR conference. In the mean time, we encourage researcher to use the feature representation made available.

Researchers interested in experimenting with different test split using the code provided in the experiments folder might need to download the full Imagenet dataset.
The full Imagenet dataset is huge, you will need almost 1TB of storage and it may take up to one week to download.

# Aknowledgement

Much of the code used to download the raw images was adapted from https://github.com/akshaychawla/ImageNet-downloader