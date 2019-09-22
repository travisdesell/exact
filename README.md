
# Getting Started and Prerequisites

EXONA has been developed to compile using CMake, which should be installed before attempting to compile. To use the MPI version, a version of MPI (such as OpenMPI) should be installed. EXACT currently requires libtiff and libpng
The EXACT algorithm can also checkpoint to a database, however this is not required.  To enable this I recommend installing libmysql-dev via apt-get on Linux systems, or mysql via [homebrew](https://brew.sh) on OSX.  Other than that, EXACT/EXALT/EXAMM has no prerequesites other than c++11 compatible compiler.

To build:

```
~/exact $ mkdir build
~/exact $ cd build
~/exact/build $ cmake ..
~/exact/build $ make
```

# EXAMM: Evolutionary eXploration of Augmenting Memory Models and EXALT: Evolutionary eXploration of Augmenting LSTM Topologies

Source code for EXALT/EXAMM can be found in the rnn subdirectory. EXALT has been enhanced with the ability to utilize more recurrent memory cells and has been renamed EXAMM.  The memory cells currently implemented are Delta-RNN, GRU, LSTM, MGU, and UGRNNs. Some example time series data has been provided as part of two publications on EXALT and EXAMM, which also provide implementation details:

1. Alex Ororbia, AbdElRahman ElSaid, and Travis Desell. **Investigating Recurrent Neural Network Memory Structures using Neuro-Evolution.** *The Genetic and Evolutionary Computation Conference (GECCO 2019).* Prague, Czech Republic. July 8-12, 2019. [link](https://dl.acm.org/citation.cfm?id=3321795)

2. AbdElRahman ElSaid, Steven Benson, Shuchita Patwardhan, David Stadem and Travis Desell. **Evolving Recurrent Neural Networks for Time Series Data Prediction of Coal Plant Parameters.** *The 22nd International Conference on the Applications of Evolutionary Computation (EvoStar: EvoApps 2019).* Leipzig, Germany. April 24-26, 2019. [link](https://link.springer.com/chapter/10.1007/978-3-030-16692-2_33)




# EXACT: Evolutionary Exploration of Augmenting Convolutional Topologies

This repository provides source code for the Evolutionary Exploration of Augmenting Convolutional Topologies algorithm.  This algorithm progressively evolves convolutional neural networks for image classification problems.  The algorithm is asychronous, which allows for easy multithreading and parallelization. Code is provided for running EXACT as a BOINC volunteer computing project, on a cluster or supercomputer using MPI or on a desktop or laptop using threads.

We've built and run EXACT on both an Ubuntu Linux high performance computing cluser and OSX laptops and desktops. We have not tried it on Windows.  If there are issues with the CMake scripts please let us know and we'll update them.

If you want to set this up on your own BOINC volunteer computing project, we recommend sending us an email as this is a rather complicated process.

## Setting up Training and Testing Data

This version EXACT is set up to run using the [MNIST Handwritten Digits Dataset](http://yann.lecun.com/exdb/mnist/). However it expects a different data format where the images and labels are combined.  You can download the data and convert it as follows:

```
~/exact $ mkdir datasets
~/exact $ cd datasets

~/exact/datasets $ wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
~/exact/datasets $ wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
~/exact/datasets $ wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
~/exact/datasets $ wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz

~/exact/datasets $ unzip train-images-idx3-ubyte.gz
~/exact/datasets $ unzip train-labels-idx1-ubyte.gz
~/exact/datasets $ unzip t10k-images-idx3-ubyte.gz
~/exact/datasets $ unzip t10k-labels-idx1-ubyte.gz

~/exact/datasets $ cd ../build
~/exact/build $ ./image_tools/convert_mnist_data ../datasets/train-images-idx3-ubyte ../datasets/train-labels-idx1-ubyte ../datasets/mnist_training_data.bin 60000
~/exact/build $ ./image_tools/convert_mnist_data ../datasets/t10k-images-idx3-ubyte ../datasets/t10k-labels-idx1-ubyte ../datasets/mnist_testing_data.bin 10000
```

## Example Genomes from GECCO 2017

Our submission to GECCO describes a set of best found genomes for the MNIST handwritten digits dataset.  These can be found in the genomes subdirectory of the project. Please checkout the tag for the GECCO paper to use the version of EXACT these CNN genome files were generated with:

```
git checkout -b exact_gecco gecco_2017
```

After compiling this version and setting up the MNIST training and testing data as described in the previous section, these genomes can be run over the training and testing data for validation as follows:

```
~/exact/build/ $ ./tests/evaluate_cnn --training_data ../datasets/mnist_training_data.bin --testing_data ../datasets/mnist_testing_data.bin --genome_file ../genomes/genome_46823
~/exact/build/ $ ./tests/evaluate_cnn --training_data ../datasets/mnist_training_data.bin --testing_data ../datasets/mnist_testing_data.bin --genome_file ../genomes/genome_57302
~/exact/build/ $ ./tests/evaluate_cnn --training_data ../datasets/mnist_training_data.bin --testing_data ../datasets/mnist_testing_data.bin --genome_file ../genomes/genome_59455
~/exact/build/ $ ./tests/evaluate_cnn --training_data ../datasets/mnist_training_data.bin --testing_data ../datasets/mnist_testing_data.bin --genome_file ../genomes/genome_59920
```
