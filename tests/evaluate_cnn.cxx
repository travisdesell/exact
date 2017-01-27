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

    //string testing_data;
    //get_argument(arguments, "--testing_data", true, testing_data);

    Images images(training_data);

    CNN_Genome *genome = new CNN_Genome(genome_id);
    genome->evaluate(images);
}
