#include <chrono>
#include <fstream>
using std::getline;
using std::ifstream;

#include <random>
using std::minstd_rand0;
using std::uniform_real_distribution;

#include <string>
using std::string;
using std::to_string;

#include <vector>
using std::vector;

#include<iostream>
using std::cout;
using std::endl;

#include "common/arguments.hxx"
#include "common/log.hxx"
#include "gradient_test.hxx"
#include "rnn/generate_nn.hxx"
#include "rnn/rnn_edge.hxx"
#include "rnn/rnn_genome.hxx"
#include "rnn/rnn_node.hxx"
#include "rnn/rnn_node_interface.hxx"
#include "rnn/multiply_node.hxx"
#include "time_series/time_series.hxx"
#include "weights/weight_rules.hxx"

int main(int argc, char** argv) {
    vector<string> arguments = vector<string>(argv, argv + argc);

    Log::initialize(arguments);
    Log::set_id("main");

    RNN_Genome* genome;
    string genome_binary;
    get_argument(arguments, "--genome_binary", true, genome_binary);

    genome = new RNN_Genome(genome_binary);
    genome->set_weights(genome->get_best_parameters());
    genome->get_equations();
    cout << "best_validation_mse: " << to_string(genome->get_best_validation_mse()) << endl;
    
    delete genome; 
}
