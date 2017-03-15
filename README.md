# EXACT: Evolutionary Exploration of Augmenting Convolutional Topologies

This repository provides source code for the Evolutionary Exploration of Augmenting Convolutional Topologies algorithm.  This algorithm progressively evolves convolutional neural networks for image classification problems.  The algorithm is asychronous, which allows for easy multithreading and parallelization. Code is provided for running EXACT as a BOINC volunteer computing project, on a cluster or supercomputer using MPI or on a desktop or laptop using threads.

## Getting Started and Prerequisites

EXACT has been developed to compile using CMake, which should be installed before attempting to compile. To use the MPI version, a version of MPI (such as OpenMPI) should be installed.  The EXACT algorithm can also checkpoint to a database.  To enable this I recommend installing libmysql-dev via apt-get on Linux systems, or mysql via [homebrew](https://brew.sh) on OSX.  Other than that, EXACT has no prerequesites other than c++11 compatible compiler.

To build:

```
~/exact $ mkdir build
~/exact $ cd build
~/exact/build $ cmake ..
~/exact/build $ make
```

I've built and run EXACT on both an Ubuntu Linux high performance computing cluser and OSX laptops and desktops. I have not tried it on Windows.  If there are issues with the CMake scripts please let me know and I'll update them.

If you want to set this up on your own BOINC volunteer computing project, I recommend sending me an email as this is a rather complicated process.

## Setting up Training and Testing Data

This version EXACT is set up to run using the [MNIST Handwritten Digits Dataset](http://yann.lecun.com/exdb/mnist/). However it expects a different data format where the images and labels are combined.  You can download the data and convert it as follows:

```
~/exact $ mkdir datasets
~/exact $ cd datasets
~/exact/datasets $ wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
~/exact/datasets $ wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
~/exact/datasets $ wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
~/exact/datasets $ wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
~/exact/datasets $ cd ../build
~/exact/build $ ./image_tools/convert_mnist_data ../datasets/train-images-idx3-ubyte.gz ../datasets/train-labels-idx1-ubyte.gz ../datasets/mnist_training_data.bin 60000
~/exact/build $ ./image_tools/convert_mnist_data ../datasets/t10k-images-idx3-ubyte.gz ../datasets/t10k-labels-idx1-ubyte.gz ../datasets/mnist_testing_data.bin 10000
```

## Example Genomes from GECCO 2017

Our submission to GECCO describes a set of best found genomes for the MNIST handwritten digits dataset.  These can be found in the genomes subdirectory of the project. After setting up the MNIST training and testing data as described in the previous section, these genomes can be run over the training and testing data for validation as follows:

```
~/exact/build/ $ ./tests/evaluate_cnn --training_data ../datasets/mnist_training_data.bin --testing_data ../datasets/mnist_testing_data.bin --genome_file ../genomes/genome_46823
~/exact/build/ $ ./tests/evaluate_cnn --training_data ../datasets/mnist_training_data.bin --testing_data ../datasets/mnist_testing_data.bin --genome_file ../genomes/genome_57302
~/exact/build/ $ ./tests/evaluate_cnn --training_data ../datasets/mnist_training_data.bin --testing_data ../datasets/mnist_testing_data.bin --genome_file ../genomes/genome_59455
~/exact/build/ $ ./tests/evaluate_cnn --training_data ../datasets/mnist_training_data.bin --testing_data ../datasets/mnist_testing_data.bin --genome_file ../genomes/genome_59920
```
