#ifndef EXAMM_GRADIENT_TEST
#define EXAMM_GRADIENT_TEST

#include <chrono>

#include <fstream>
using std::getline;
using std::ifstream;

#include <random>
using std::minstd_rand0;
using std::uniform_real_distribution;

#include <string>
using std::string;

#include <vector>
using std::vector;

#include "common/arguments.hxx"

#include "rnn/lstm_node.hxx"
#include "rnn/rnn_edge.hxx"
#include "rnn/rnn_genome.hxx"
#include "rnn/rnn_node.hxx"
#include "rnn/rnn_node_interface.hxx"

#include "rnn/generate_nn.hxx"

#include "time_series/time_series.hxx"

void initialize_generator();
void generate_random_vector(int number_parameters, vector<double> &v);

void gradient_test(string name, RNN_Genome *genome, const vector< vector<double> > &inputs, const vector< vector<double> > &outputs);

#endif
