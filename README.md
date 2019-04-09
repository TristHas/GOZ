# Generic Object ZSL Datast

This repository references the dataset described in the paper "On Zero-Shot Learning of generic objects" to be presented at CVPR 2019.
It also provides code to reproduce the different experiments presented in the paper, as well as code to manipulate the Wordnet hierarchy and Imagenet metadata to generate new training/test class splits of the Imagenet dataset subject to different constraints.

## About this repository

People interested in downloading the dataset can follow the instructions in XXX. 
People interested in the dataset generation process and experiments presented in the paper should refer to XXX.

### Dependencies

### Download

Downloading the GOZ dataset:

If you are only interested in downloading the dataset, you do not need to install the full set of dependices listed in the requirements.txt.
You can just clone the repo and check out the code and instructions of the download folder.
You can either work on raw images or download the Resnet-50 feature representations of the original images used in the paper.

To download the original images, see the notebook download/original.ipynb
We only provide code to download the images of the test.
The training set is the usual training set of the ILSVRC 2012 classification challenge. 
Please directly download it from the imagenet website (http://image-net.org/download-images).

Alternatively, you can work on the extracted resnet features used in the paper.
Features are provided for both the training and test sets. 
To download the visual features, see the notebook download/features.ipynb.

### Benchmark

What things you need to install the software and how to install them

```
Give examples
```

### Experiments

Dataset Creation:
To either reproduce some of the experiments of the paper or to experiment with test splits different from the proposed one, you will need to:
 - Install the dependencies defined in the requirements.txt
 - Download the full Imagenet dataset (>13M images, XXX GB) or a subset of interest to you.

We share scripts to efficiently download Imagenet images in parallel.
Downloading the 13M images of the Imagenet dataset can be time-consuming, even with parallel accesses, and need large storage capacity. 
For researchers interested in experimenting on different test splits with limited resurces, we propose a trick we have been using:
Instead of working on individual images, we have been working on the mean visual feature activations.
This drastically speeds up processing time at the cost of ignoring the intra-class variance of visual feature distribution.


Dataset creation/manipulation:
The bulk of the code used to create the dataset is contained in the src folder.
The folder dataset features the notebooks used to create the dataset.

Experiments reproduction:
The folder experiments features notebooks to reproduce the experiments presented in the paper.

Data and dependencies
Dependencies on python modules are listed in the requirements.txt file at the root of the repository.
Additional data needed for the experiments such as the Wikipedia word counts can be found in the data folder.


### Citation

If you use this dataset for your research, please cite the following paper:



### Rest of the README template


```
Give the example
```

And repeat

```
until finished
```

End with an example of getting some data out of the system or using it for a little demo

## Running the tests

Explain how to run the automated tests for this system

### Break down into end to end tests

Explain what these tests test and why

```
Give an example
```

### And coding style tests

Explain what these tests test and why

```
Give an example
```

## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **Billie Thompson** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc




People interested in downloading the dataset can follow the instructions in XXX. 
People interested in the dataset generation process and experiments presented in the paper should refer to XXX.

Downloading the GOZ dataset:

If you are only interested in downloading the dataset, you do not need to install the full set of dependices listed in the requirements.txt.
You can just clone the repo and check out the code and instructions of the download folder.
You can either work on raw images or download the Resnet-50 feature representations of the original images used in the paper.

To download the original images, see the notebook download/original.ipynb
We only provide code to download the images of the test.
The training set is the usual training set of the ILSVRC 2012 classification challenge. 
Please directly download it from the imagenet website (http://image-net.org/download-images).

Alternatively, you can work on the extracted resnet features used in the paper.
Features are provided for both the training and test sets. 
To download the visual features, see the notebook download/features.ipynb.



Dataset Creation:
To either reproduce some of the experiments of the paper or to experiment with test splits different from the proposed one, you will need to:
 - Install the dependencies defined in the requirements.txt
 - Download the full Imagenet dataset (>13M images, XXX GB) or a subset of interest to you.

We share scripts to efficiently download Imagenet images in parallel.
Downloading the 13M images of the Imagenet dataset can be time-consuming, even with parallel accesses, and need large storage capacity. 
For researchers interested in experimenting on different test splits with limited resurces, we propose a trick we have been using:
Instead of working on individual images, we have been working on the mean visual feature activations.
This drastically speeds up processing time at the cost of ignoring the intra-class variance of visual feature distribution.


Dataset creation/manipulation:
The bulk of the code used to create the dataset is contained in the src folder.
The folder dataset features the notebooks used to create the dataset.

Experiments reproduction:
The folder experiments features notebooks to reproduce the experiments presented in the paper.

Data and dependencies
Dependencies on python modules are listed in the requirements.txt file at the root of the repository.
Additional data needed for the experiments such as the Wikipedia word counts can be found in the data folder.


