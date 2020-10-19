
# Getting Started and Prerequisites

EXONA has been developed to compile using CMake, which should be installed before attempting to compile. To use the MPI version, a version of MPI (such as OpenMPI) should be installed. EXACT currently requires libtiff and libpng
The EXACT algorithm can also checkpoint to a database, however this is not required.  To enable this I recommend installing libmysql-dev via apt-get on Linux systems, or mysql via [homebrew](https://brew.sh) on OSX.  Other than that, EXACT/EXALT/EXAMM has no prerequesites other than c++11 compatible compiler.

If you are using OSX, to set up the environment:

```
brew install cmake
brew install mysql
brew install open-mpi
brew install libtiff
brew install libpng
xcode-select --install
```

To build:

```
~/exact $ mkdir build
~/exact $ cd build
~/exact/build $ cmake ..
~/exact/build $ make
```

You may also want to have graphviz installed so you can generate images of the evolved neural networks.  EXACT/EXALT/EXAMM will write out evolved genomes in a .gv (graphviz) format for this. For example, can generate a pdf from a gv file (assuming graphviz is installed with):

```
$ dot -Tpdf genome.gv -o genome.pdf
```

# EXAMM: Evolutionary eXploration of Augmenting Memory Models and EXALT: Evolutionary eXploration of Augmenting LSTM Topologies

Source code for EXALT/EXAMM can be found in the rnn subdirectory. EXALT has been enhanced with the ability to utilize more recurrent memory cells and has been renamed EXAMM.  The memory cells currently implemented are Delta-RNN, GRU, LSTM, MGU, and UGRNNs. Some example time series data has been provided as part of two publications on EXALT and EXAMM, which also provide implementation details:

1. Alex Ororbia, AbdElRahman ElSaid, and Travis Desell. **[Investigating Recurrent Neural Network Memory Structures using Neuro-Evolution](https://dl.acm.org/citation.cfm?id=3321795).** *The Genetic and Evolutionary Computation Conference (GECCO 2019).* Prague, Czech Republic. July 8-12, 2019.

2. AbdElRahman ElSaid, Steven Benson, Shuchita Patwardhan, David Stadem and Travis Desell. **[Evolving Recurrent Neural Networks for Time Series Data Prediction of Coal Plant Parameters](https://link.springer.com/chapter/10.1007/978-3-030-16692-2_33).** *The 22nd International Conference on the Applications of Evolutionary Computation (EvoStar: EvoApps 2019).* Leipzig, Germany. April 24-26, 2019.

These datasets can be found in the *datasets* directory, and provide example CSV files which you can use with EXAMM. EXAMM can be run in two different ways, a multithreaded version:

```
./multithreaded/examm_mt --number_threads 9 --training_filenames ../datasets/2018_coal/burner_[0-9].csv --test_filenames ../datasets/2018_coal/burner_1[0-1].csv --time_offset 1 --input_parameter_names Conditioner_Inlet_Temp Conditioner_Outlet_Temp Coal_Feeder_Rate Primary_Air_Flow Primary_Air_Split System_Secondary_Air_Flow_Total Secondary_Air_Flow Secondary_Air_Split Tertiary_Air_Split Total_Comb_Air_Flow Supp_Fuel_Flow Main_Flm_Int --output_parameter_names Main_Flm_Int --number_islands 10 --population_size 10 --max_genomes 2000 --bp_iterations 10 --output_directory "./test_output" --possible_node_types simple UGRNN MGU GRU delta LSTM --std_message_level INFO --file_message_level INFO
```

And a parallel version using MPI:

```
~/exact/build/ $ mpirun -np 9 ./mpi/examm_mpi --training_filenames ../datasets/2018_coal/burner_[0-9].csv --test_filenames ../datasets/2018_coal/burner_1[0-1].csv --time_offset 1 --input_parameter_names Conditioner_Inlet_Temp Conditioner_Outlet_Temp Coal_Feeder_Rate Primary_Air_Flow Primary_Air_Split System_Secondary_Air_Flow_Total Secondary_Air_Flow Secondary_Air_Split Tertiary_Air_Split Total_Comb_Air_Flow Supp_Fuel_Flow Main_Flm_Int --output_parameter_names Main_Flm_Int --number_islands 10 --population_size 10 --max_genomes 2000 --bp_iterations 10 --output_directory "./test_output" --possible_node_types simple UGRNN MGU GRU delta LSTM --std_message_level INFO --file_message_level INFO
```

Which will run EXAMM with 9 threads or 9 processes, respectively. Note that EXAMM uses one thread/process as the master and this typically just waits on the results of backprop so you if you have 8 processors/cores available you can usually run EXAMM with 9 processes/threads for better performance. A performance log of RNN fitnesses will be exported into fitness_log.csv, as well as the best found RNNs into the specified output directory, in this case *./test_output*.  You can control the level of message logging for standard output with *--std_message_level* (options are NONE, FATAL, ERROR, WARNING, INFO, DEBUG, TRACE and ALL) and message logging to files (which will be placed in the output directory) with *--file_message_level*. Separate logging files will be made for each thread/process.

The aviation data can be run similarly, however it the data should be normalized first (which can be done with the *--normalize* command line parameter), e.g.:

```
./multithreaded/examm_mt --number_threads 9 --training_filenames ../datasets/2018_ngafid/flight_[0-7].csv --test_filenames ../datasets/2018_ngafid/flight_[8-9].csv --time_offset 1 --input_parameter_names "AltAGL" "E1 CHT1" "E1 CHT2" "E1 CHT3" "E1 CHT4" "E1 EGT1" "E1 EGT2" "E1 EGT3" "E1 EGT4" "E1 OilP" "E1 OilT" "E1 RPM" "FQtyL" "FQtyR" "GndSpd" "IAS" "LatAc" "NormAc" "OAT" "Pitch" "Roll" "TAS" "volt1" "volt2" "VSpd" "VSpdG" --output_parameter_names Pitch --number_islands 10 --population_size 10 --max_genomes 2000 --bp_iterations 10 --output_directory "./test_output" --possible_node_types simple UGRNN MGU GRU delta LSTM --normalize --std_message_level INFO --file_message_level INFO
```

The *--time_offset* parameter specifies how many time steps in the future EXAMM should predict for the output parameter(s). The *--number_islands* is the number of islands of populations that EXAMM will use, and the *--population_size* parameter specifies how many individuals/genomes are in each island. The *--bp_iterations* specifies how many epochs/iterations backpropagation should be run for each generated RNN genome.

The 

# EXACT: Evolutionary Exploration of Augmenting Convolutional Topologies

This repository provides source code for the Evolutionary Exploration of Augmenting Convolutional Topologies algorithm.  This algorithm progressively evolves convolutional neural networks for image classification problems.  The algorithm is asychronous, which allows for easy multithreading and parallelization. Code is provided for running EXACT as a BOINC volunteer computing project, on a cluster or supercomputer using MPI or on a desktop or laptop using threads.

We've built and run EXACT on both an Ubuntu Linux high performance computing cluser and OSX laptops and desktops. We have not tried it on Windows.  If there are issues with the CMake scripts please let us know and we'll update them.

If you want to set this up on your own BOINC volunteer computing project, we recommend sending us an email as this is a rather complicated process.

For more information on EXACT please see our following publications:

1. Travis Desell. **[Accelerating the Evolution of Convolutional Neural Networks with Node-Level Mutations and Epigenetic Weight Initialization](https://arxiv.org/abs/1811.08286).** *arXiv: Neural and Evolutionary Computing (cs.NE).* November, 2018.

2. Travis Desell. **[Developing a Volunteer Computing Project to Evolve Convolutional Neural Networks and Their Hyperparameters](https://ieeexplore.ieee.org/document/8109119).** *The 13th IEEE International Conference on eScience (eScience 2017).* Auckland, New Zealand. October 24-27 2017.


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

You can also split up either the training or test data to add a third dataset of validation images for more robust analysis, e.g., so you can train on one set, use the validation set to determine the best CNNs for EXACT to keep, and then after the evolution process is completed you can test the best found genome(s) on the test set which has not been seen before. You can split up the MNIST training data into a 50k training set and 10k validation set as follows:

```
~/exact/build/ $ ./image_tools/split_mnist_data ../datasets/train-images-idx3-ubyte ../datasets/train-labels-idx1-ubyte ../datasets/mnist_train_50k.bin ../datasets/mnist_validation_10k.bin 60000 1000
```

Where the usage is:

```
./image_tools/split_mnist_data <mnist image file> <mnist label file> <output training file> <output validation file> <expected number of images> <validation images per label>
```

So this will take 1000 images of each type from the training data and put them into the specified validation file (*mnist_validation_10k.bin*) and the rest of the images will be in the training set (*mnist_train_50k.bin*).

## Running EXACT

You can then run EXACT in a similar manner to EXAMM, either using threads or MPI. For the threaded version:

```
~/exact/build/ $ ./multithreaded/exact_mt --number_threads 9 --padding 2 --training_file ../datasets/mnist_train_50k.bin --validation_file ../datasets/mnist_validation_10k.bin  --testing_file ../datasets/mnist_testing_data.bin --population_size 30 --max_epochs 5 --use_sfmp 1 --use_node_operations 1 --max_genomes 1000 --output_directory ./test_exact --search_name "test" --reset_edges 0 --images_resize 50000

```

And for the MPI version:

```
~/exact/build/ $ mpirun -np 9 ./mpi/exact_mpi --padding 2 --training_file ../datasets/mnist_train_50k.bin --validation_file ../datasets/mnist_validation_10k.bin  --testing_file ../datasets/mnist_testing_data.bin --population_size 30 --max_epochs 5 --use_sfmp 1 --use_node_operations 1 --max_genomes 1000 --output_directory ./test_exact --search_name "test" --reset_edges 0 --images_resize 50000
```

Which will run EXACT with 9 threads or processes, respectively. The *--use_sfmp* argument turns on or off scaled fractional max pooling (which allows for pooling operations between feature maps of any size), the *--use_node_operations* argument turns on or off node level mutations (see the EXACT and EXAMM papers), the *--reset_edges* parameter turns on or off Lamarckian weight evolution (turning it on will evolve and train networks faster) and the *--images_resize* parameter allows EXACT to train CNNs on a subset of the training data to speed the evolution process (e.g., --images_resize 5000 would train each CNN on a different subset of 5k images from the training data, as opposed to the full 50k).

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
