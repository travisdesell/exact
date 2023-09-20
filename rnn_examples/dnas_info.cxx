#include <chrono>
#include <fstream>
using std::getline;
using std::ifstream;
using std::ofstream;

#include <random>
using std::minstd_rand0;
using std::uniform_real_distribution;

#include <string>
using std::string;

#include <vector>
using std::vector;

#include "common/arguments.hxx"
#include "common/files.hxx"
#include "common/log.hxx"
#include "rnn/generate_nn.hxx"
#include "rnn/gru_node.hxx"
#include "rnn/lstm_node.hxx"
#include "rnn/rnn_edge.hxx"
#include "rnn/rnn_genome.hxx"
#include "rnn/rnn_node.hxx"
#include "rnn/rnn_node_interface.hxx"
#include "time_series/time_series.hxx"
#include "weights/weight_rules.hxx"
#include "weights/weight_update.hxx"

vector<vector<vector<double> > > training_inputs;
vector<vector<vector<double> > > training_outputs;
vector<vector<vector<double> > > test_inputs;
vector<vector<vector<double> > > test_outputs;

bool random_sequence_length;
int32_t sequence_length_lower_bound = 30;
int32_t sequence_length_upper_bound = 100;

RNN_Genome* genome;
RNN* rnn;
WeightUpdate* weight_update_method;
int32_t bp_iterations;
bool using_dropout;
double dropout_probability;

ofstream* log_file;
string output_directory;

double objective_function(const vector<double>& parameters) {
    rnn->set_weights(parameters);

    double error = 0.0;

    for (int32_t i = 0; i < (int32_t) training_inputs.size(); i++) {
        error += rnn->prediction_mae(training_inputs[i], training_outputs[i], false, true, 0.0);
    }

    return -error;
}

double test_objective_function(const vector<double>& parameters) {
    rnn->set_weights(parameters);

    double total_error = 0.0;

    for (int32_t i = 0; i < (int32_t) test_inputs.size(); i++) {
        double error = rnn->prediction_mse(test_inputs[i], test_outputs[i], false, true, 0.0);
        total_error += error;

        Log::info("output for series[%d]: %lf\n", i, error);
    }

    return -total_error;
}

int main(int argc, char** argv) {
    vector<string> arguments = vector<string>(argv, argv + argc);

    Log::initialize(arguments);
    Log::set_id("main");

    string filename;
    get_argument(arguments, "--filename", true, filename);

    RNN_Genome genome(filename);

    for (auto node : genome.get_nodes()) {
        if (DNASNode *d = dynamic_cast<DNASNode*>(node)) {
          std::cout << "'" << filename << "': ";
          d->print_info();
        }
    }

    Log::release_id("main");
}
