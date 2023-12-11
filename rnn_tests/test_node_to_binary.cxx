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

#include <iostream>

#include "common/arguments.hxx"
#include "common/log.hxx"
#include "gradient_test.hxx"
#include "rnn/generate_nn.hxx"
#include "rnn/rnn_edge.hxx"
#include "rnn/rnn_genome.hxx"
#include "rnn/rnn_node.hxx"
#include "rnn/rnn_node_interface.hxx"
#include "rnn/sin_node.hxx"
#include "rnn/sum_node.hxx"
#include "time_series/time_series.hxx"
#include "weights/weight_rules.hxx"

int main(int argc, char** argv) {
    vector<string> arguments = vector<string>(argv, argv + argc);

    Log::initialize(arguments);
    Log::set_id("main");

    initialize_generator();

    RNN_Genome* genome_original = nullptr;

    vector<vector<double> > inputs;
    vector<vector<double> > outputs;

    int input_length = 10;
    string hidden_node_type;
    get_argument(arguments, "--hidden_node_type", true, hidden_node_type);

    WeightRules* weight_rules = new WeightRules();

    int32_t max_recurrent_depth = 3;
    Log::info("testing with max recurrent depth: %d\n", max_recurrent_depth);

    inputs.resize(3);
    outputs.resize(3);

    vector<string> inputs3{"input 1", "input 2", "input 3"};
    vector<string> outputs3{"output 1", "output 2", "input 3"};
    Log::info("testing with 3 input nodes, 3 output nodes\n");

    generate_random_vector(input_length, inputs[0]);
    generate_random_vector(input_length, outputs[0]);
    generate_random_vector(input_length, inputs[1]);
    generate_random_vector(input_length, outputs[1]);
    generate_random_vector(input_length, inputs[2]);
    generate_random_vector(input_length, outputs[2]);

    if (hidden_node_type.compare("sin") == 0) {
        Log::info("TESTING SIN!!!\n");
        genome_original = create_sin(inputs3, 1, 5, outputs3, max_recurrent_depth, weight_rules);
        Log::info("testing with 1 hidden layer, 5 sin nodes\n");
    } else if (hidden_node_type.compare("sum") == 0){
        Log::info("TESTING SUM!!!\n");
        genome_original = create_sum(inputs3, 1, 5, outputs3, max_recurrent_depth, weight_rules);
        Log::info("testing with 1 hidden layer, 5 sum nodes\n");
    }

    int32_t num_weights = genome_original->get_number_weights();
    vector<double> best_parameters_original;
    vector<double> initial_parameters_original;
    generate_random_vector(num_weights, best_parameters_original);
    generate_random_vector(num_weights, initial_parameters_original);
    genome_original->set_best_parameters(best_parameters_original);
    genome_original->set_initial_parameters(initial_parameters_original);

    string path = "./genome_original.bin";
    genome_original->write_to_file(path);
    RNN_Genome* genome_file = new RNN_Genome(path);
    vector<double> best_parameters_file = genome_file->get_best_parameters();
    vector<double> initial_parameters_file = genome_file->get_initial_parameters();

    if (best_parameters_original == best_parameters_file) {
        Log::info("PASS: BEST PARAMETERS ARE EQUAL!!!\n");
    } else {
        Log::fatal("FAILURE: BEST PARAMETERS ARE NOT EQUAL!!!\n");
        exit(1);
    }

    if (initial_parameters_original == initial_parameters_file) {
        Log::info("PASS: INITIAL PARAMETERS ARE EQUAL!!!\n");
    } else {
        Log::fatal("FAILURE: INITIAL PARAMETERS ARE NOT EQUAL!!!\n");
        exit(1);
    }

    RNN* rnn_original = genome_original->get_rnn();
    RNN* rnn_file = genome_file->get_rnn();

    if (rnn_original->get_number_nodes() == rnn_file->get_number_nodes()) {
        Log::info("PASS: RNN NODE COUNT EQUAL!!!\n");
    } else {
        Log::fatal("FAILURE: RNN NODE COUNT NOT EQUAL!!!\n");
        exit(1);
    }
    if (rnn_original->get_number_edges() == rnn_file->get_number_edges()) {
        Log::info("PASS: RNN EDGE COUNT EQUAL!!!\n");
    } else {
        Log::fatal("FAILURE: RNN EDGE COUNT NOT EQUAL!!!\n");
        exit(1);
    }

    double analytic_mse_original, analytic_mse_file;
    vector<double> analytic_gradient_original, analytic_gradient_file;
    Log::info("getting analytic gradient\n");
    rnn_original->get_analytic_gradient(
        best_parameters_original, inputs, outputs, analytic_mse_original, analytic_gradient_original, false, true, 0.0
    );
    rnn_file->get_analytic_gradient(
        best_parameters_file, inputs, outputs, analytic_mse_file, analytic_gradient_file, false, true, 0.0
    );

    vector<double> weights_original, weights_file;
    rnn_original->get_weights(weights_original);
    rnn_file->get_weights(weights_file);
    if (weights_original == weights_file) {
        Log::info("PASS: RNN WEIGHTS EQUAL!!!\n");
    } else {
        Log::fatal("FAILURE: RNN WEIGHTS NOT EQUAL!!!\n");
        exit(1);
    }

    int output_node_count = 1;
    for (int i = 0; i < rnn_original->get_number_nodes(); i++) {
        RNN_Node_Interface* test_node_original = rnn_original->get_node(i);
        RNN_Node_Interface* test_node_file = rnn_file->get_node(i);
        if ((test_node_original->layer_type == OUTPUT_LAYER) && (test_node_file->layer_type == OUTPUT_LAYER)) {
            if (test_node_original->output_values == test_node_file->output_values) {
                Log::info("PASS: RNN FORWARD PASS OUTPUT NODE %d - OUTPUT EQUAL!!!\n", output_node_count);
            } else {
                Log::fatal("FAILURE: RNN FORWARD PASS OUTPUT NODE %d - OUTPUT NOT EQUAL!!!\n", output_node_count);
                exit(1);
            }
            ++output_node_count;
        }
    }

    if (analytic_gradient_original == analytic_gradient_file) {
        Log::info("PASS: RNN ANALYTIC GRADIENTS ARE EQUAL!!!\n");
    } else {
        Log::fatal("FAILURE: RNN ANALYTIC GRADIENTS ARE NOT EQUAL!!!\n");
        exit(1);
    }

    generate_random_vector(input_length, inputs[0]);
    generate_random_vector(input_length, outputs[0]);
    generate_random_vector(input_length, inputs[1]);
    generate_random_vector(input_length, outputs[1]);
    generate_random_vector(input_length, inputs[2]);
    generate_random_vector(input_length, outputs[2]);
    Log::info("new inputs/outputs generated\n");

    double empirical_mse_original, empirical_mse_file;
    vector<double> empirical_gradient_original, empirical_gradient_file;
    Log::info("getting empirical gradient\n");
    rnn_original->get_empirical_gradient(
        best_parameters_original, inputs, outputs, empirical_mse_original, empirical_gradient_original, false, true, 0.0
    );
    rnn_file->get_empirical_gradient(
        best_parameters_file, inputs, outputs, empirical_mse_file, empirical_gradient_file, false, true, 0.0
    );

    if (empirical_gradient_original == empirical_gradient_file) {
        Log::info("PASS: RNN EMPIRICAL GRADIENTS ARE EQUAL!!!\n");
    } else {
        Log::fatal("FAILURE: RNN EMPIRICAL GRADIENTS ARE NOT EQUAL!!!\n");
        exit(1);
    }

    delete genome_original;
    delete genome_file;
}
