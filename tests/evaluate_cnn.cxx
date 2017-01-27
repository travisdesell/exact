#include <iomanip>
using std::setw;

#include <iostream>
using std::cerr;
using std::cout;
using std::endl;

#include <string>
using std::string;

#include <vector>
using std::vector;

#include "common/arguments.hxx"

#include "strategy/exact.hxx"
#include "strategy/cnn_genome.hxx"
#include "strategy/cnn_edge.hxx"
#include "strategy/cnn_node.hxx"

int main(int argc, char **argv) {
    vector<string> arguments = vector<string>(argv, argv + argc);

    int genome_id;
    get_argument(arguments, "--genome_id", true, genome_id);

    string training_data;
    get_argument(arguments, "--training_data", true, training_data);

    string testing_data;
    get_argument(arguments, "--testing_data", true, testing_data);

    Images training_images(training_data);
    Images testing_images(testing_data, training_images.get_average(), training_images.get_std_dev());

    CNN_Genome *genome = new CNN_Genome(genome_id);
    genome->evaluate(training_images);

    genome->evaluate(testing_images);
}
